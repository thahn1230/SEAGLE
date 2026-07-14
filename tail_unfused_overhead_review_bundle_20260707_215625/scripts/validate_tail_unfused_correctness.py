#!/usr/bin/env python
"""Correctness of the tail variants vs the true unrotated model (T0).

(A) tiny fp32: random x, orthogonal R1, positive gamma, random W_lm; build the
    fused head W_fused = W_lm @ diag(gamma) @ R1; run T0-T4; compare logits.
    Separates math from fp16 roundoff.
(B) real model: capture the unrotated model's final-norm INPUT x, its hidden h
    and its logits (ground truth); capture the rotated model's norm INPUT x_R;
    recompute every tail from the REAL x_R with stash R1/gamma/W_lm/W_fused and
    compare hidden vs h, logits vs ground-truth logits (fp16, the deployed dtype).

Outputs runs/<run>/tail_correctness.{json,csv}.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/validate_tail_unfused_correctness.py \
      --out-dir runs/tail_unfused_<ts> --num-prompts 2
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             rotation_aware as ra, study, tail_unfused as tu)

DEV = "cuda:0"


def logit_metrics(lg, ref):
    lg = lg.float(); ref = ref.float()
    d = (lg - ref)
    top1 = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
    k = min(5, lg.shape[-1])
    a5 = lg.topk(k, -1).indices; b5 = ref.topk(k, -1).indices
    ov = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / k
                       for a, b in zip(a5.reshape(-1, k), b5.reshape(-1, k))]).mean().item()
    pr = F.log_softmax(lg, -1); pref = F.log_softmax(ref, -1)
    kl = F.kl_div(pr, pref, log_target=True, reduction="batchmean").item()
    return {"max_abs_logit_error": d.abs().max().item(),
            "mean_abs_logit_error": d.abs().mean().item(),
            "relative_l2_logit_error": (d.norm() / (ref.norm() + 1e-9)).item(),
            "top1_agreement": top1, "top5_overlap": ov, "kl_divergence": kl}


def hidden_metrics(h, ref):
    h = h.float().flatten(); ref = ref.float().flatten()
    return {"hidden_rel_l2": ((h - ref).norm() / (ref.norm() + 1e-9)).item(),
            "hidden_cosine": F.cosine_similarity(h, ref, dim=0).item()}


def tiny_fp32():
    g = torch.Generator().manual_seed(0)
    D, V, M = 4096, 32000, 8
    x = torch.randn(M, D, generator=g, dtype=torch.float64)
    R1, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    gamma = torch.rand(D, generator=g, dtype=torch.float64) + 0.25
    W_lm = torch.randn(V, D, generator=g, dtype=torch.float64) * 0.02
    x_R = x @ R1
    W_fused = ra.in_fold(W_lm, R1, gamma)            # W_lm @ diag(gamma) @ R1
    eps = 1e-5
    R1t = R1.t()
    h0, lg0 = tu.tail_T0(x, gamma, W_lm, eps)
    outs = {"T1": tu.tail_T1(x_R, W_fused, eps),
            "T2": tu.tail_T2(x_R, R1t, gamma, W_lm, eps),
            "T3": tu.tail_T3(x_R, R1t, gamma, W_lm, eps),
            "T4": tu.tail_T4(x_R, gamma, W_lm, eps)}
    res = {}
    for k, (h, lg) in outs.items():
        res[k] = {**logit_metrics(lg, lg0), **hidden_metrics(h, h0)}
    return res


@torch.no_grad()
def capture_norm_io(model, ids_list, is_rotated):
    """Return per-prompt (norm_input, norm_output, logits) for the last token
    row-block (all positions) of each prompt."""
    cap = {}
    norm = model.model.norm

    def hook(mod, inp, out):
        cap["in"] = inp[0].detach()
        cap["out"] = out.detach()
    h = norm.register_forward_hook(hook)
    res = []
    for ids in ids_list:
        out = model.model(input_ids=ids.to(DEV))
        hid = out[0]
        logits = model.lm_head(hid)
        res.append((cap["in"].float().cpu(), cap["out"].float().cpu(),
                    logits.float().cpu()))
    h.remove()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    out = {"gpu": gpu, "tiny_fp32": tiny_fp32()}
    print("[tiny fp32]", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()
          if kk in ("relative_l2_logit_error", "hidden_cosine", "top1_agreement")}
          for k, v in out["tiny_fp32"].items()}, indent=2))

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"])
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    ids_list = [eagle_bridge.PROMPT_BUILDERS[tmpl](tok, p["text"]) for p in prompts]

    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    # (1) unrotated ground truth
    m0 = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                 low_cpu_mem_usage=True).to(DEV).eval()
    gt = capture_norm_io(m0, ids_list, is_rotated=False)
    del m0; torch.cuda.empty_cache()

    # (2) rotated/fused target + stash
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             cfg.get("paths", {}).get("rotations_root"))
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True).eval()
    stash = study.apply_rotation_quant(rot, "full", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    rot.to(DEV)
    rr = capture_norm_io(rot, ids_list, is_rotated=True)
    W_fused_live = rot.lm_head.weight.detach().float().cpu()
    eps = float(rot.model.norm.variance_epsilon)
    del rot; torch.cuda.empty_cache()

    R1 = stash["R1"].float(); R1t = R1.t()
    gamma = stash["gamma_f"].float()
    W_lm = stash["lm_head_weight"].float()

    # sanity: x_R ~= x @ R1 (residual stream really is rotated)
    xr_check = []
    for (x_in, _, _), (xR_in, _, _) in zip(gt, rr):
        pred = x_in @ R1
        xr_check.append(((pred - xR_in).norm() / (xR_in.norm() + 1e-9)).item())
    out["residual_is_rotated_rel_l2"] = sum(xr_check) / len(xr_check)

    rows, real = [], {"T1": [], "T2": [], "T3": [], "T4": []}
    # fp16 real-tensor tails from the REAL captured x_R
    for pi in range(len(ids_list)):
        x_in, h_gt, lg_gt = gt[pi]
        xR = rr[pi][0].to(DEV).half()
        g16 = gamma.to(DEV).half(); Wlm16 = W_lm.to(DEV).half()
        Wf16 = W_fused_live.to(DEV).half(); R1t16 = R1t.to(DEV).half()
        outs = {
            "T1": tu.tail_T1(xR, Wf16, eps),
            "T2": tu.tail_T2(xR, R1t16, g16, Wlm16, eps),
            "T3": tu.tail_T3(xR, R1t16, g16, Wlm16, eps),
            "T4": tu.tail_T4(xR, g16, Wlm16, eps)}
        for k, (h, lg) in outs.items():
            m = {**logit_metrics(lg.cpu(), lg_gt),
                 **hidden_metrics(h.cpu(), h_gt)}
            real[k].append(m)
    out["real_model_fp16"] = {k: {kk: sum(d[kk] for d in v) / len(v)
                                  for kk in v[0]} for k, v in real.items()}
    for k, v in out["real_model_fp16"].items():
        rows.append({"variant": k, "source": "real_model_fp16", **v})
    for k, v in out["tiny_fp32"].items():
        rows.append({"variant": k, "source": "tiny_fp32", **v})

    # verdicts
    r = out["real_model_fp16"]; t = out["tiny_fp32"]
    out["verdicts"] = {
        "H1_T2_matches_original":
            bool(r["T2"]["top1_agreement"] > 0.99 and r["T2"]["hidden_cosine"] > 0.999),
        "H2_T3_matches_T2":
            bool(abs(r["T3"]["relative_l2_logit_error"] - r["T2"]["relative_l2_logit_error"]) < 5e-3
                 and r["T3"]["hidden_cosine"] > 0.999),
        "H3_T4_fails":
            bool(t["T4"]["hidden_cosine"] < 0.99 or t["T4"]["top1_agreement"] < 0.9),
        "T1_logits_match_original_fused": bool(r["T1"]["top1_agreement"] > 0.99),
        "T1_hidden_is_rotated_not_original": bool(r["T1"]["hidden_cosine"] < 0.5),
        "gamma_f_nonuniform_std": gamma.std().item(),
    }
    logging_utils.write_csv(os.path.join(args.out_dir, "tail_correctness.csv"), rows)
    with open(os.path.join(args.out_dir, "tail_correctness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["verdicts"], indent=2))
    print("real_model_fp16:", json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()
          if kk in ("relative_l2_logit_error", "top1_agreement", "hidden_cosine",
                    "hidden_rel_l2")} for k, v in r.items()}, indent=2))
    print(f"-> {args.out_dir}/tail_correctness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
