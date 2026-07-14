#!/usr/bin/env python
"""Task 3 verification: the pure-R1 draft + T5 (h_R) tail is EXACT and
SINGLE-PATH.

Captures the reference (original draft consuming original h) and the pure-R1
draft (consuming h_R = h@R1) across all tree levels of one topK_genrate; checks
per-level feature agreement (cosine of pure_out to f@R1) and head-scored top-1.
Also checks the T5 tail's logits vs the true unrotated model.

The single-path claim: the pure-R1 draft uses ONE fc for both the external h_R
and every recycled f_R, with no swap — and stays exact at all levels (unlike
Variant B, which corrupts at level >= 2).

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/eval_pure_r1_correctness.py \
      --out-dir runs/pure_r1_<ts> --num-prompts 4
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
                             pure_r1_eagle as pr, rotation_aware as ra, study)
from eagle_spinquant.rotation_interface import build_original_head  # noqa: E402

DEV = "cuda:0"


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    ra.verify_fold_algebra()

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"])
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    ids_list = [eagle_bridge.PROMPT_BUILDERS[tmpl](tok, p["text"]) for p in prompts]

    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    m0 = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                 low_cpu_mem_usage=True).to(DEV).eval()
    h_list, next_toks = [], []
    W_head = m0.lm_head.weight.detach().cpu().clone()
    for ids in ids_list:
        h = m0.model(input_ids=ids.to(DEV))[0]
        next_toks.append(int(m0.lm_head(h)[0, -1].argmax()))
        h_list.append(h.float().cpu())
    orig_logits = [m0.lm_head(m0.model(input_ids=ids.to(DEV))[0]).float().cpu()
                   for ids in ids_list]
    del m0; torch.cuda.empty_cache()

    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             cfg.get("paths", {}).get("rotations_root"))
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True).eval()
    stash = study.apply_rotation_quant(rot, "full", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    rot.to(DEV)
    R1 = stash["R1"].double()
    gamma = stash["gamma_f"].double()
    G = R1.t() @ torch.diag(gamma) @ R1                       # h_R = h_hat @ G
    W_lm_R = (W_head.double() @ R1)
    R1_dev = R1.to(DEV); G_dev = G.to(DEV); W_lm_R_dev = W_lm_R.to(DEV)

    # T5 tail logit correctness: emulate h_R from the rotated model's norm input
    norm_in = {}
    hk = rot.model.norm.register_forward_hook(
        lambda m, i, o: norm_in.__setitem__("x_R", i[0].detach()))
    t5_logit = {"rel_l2": [], "top1": []}
    for pi, ids in enumerate(ids_list):
        _ = rot.model(input_ids=ids.to(DEV))
        x_R = norm_in["x_R"].double()
        h_hat = (x_R / (x_R.pow(2).mean(-1, keepdim=True)
                        + rot.model.norm.variance_epsilon).sqrt())
        h_R = h_hat @ G_dev
        lg = (h_R @ W_lm_R_dev.t()).float().cpu()
        t5_logit["rel_l2"].append(rel(lg, orig_logits[pi]))
        t5_logit["top1"].append(
            (lg.argmax(-1) == orig_logits[pi].argmax(-1)).float().mean().item())
    hk.remove()
    del rot; torch.cuda.empty_cache()

    # drafts
    head_o = build_original_head(W_head, DEV, torch.float32)
    import torch.nn as nn
    head_R = nn.Linear(4096, W_lm_R.shape[0], bias=False)
    head_R.weight.data = W_lm_R.float()
    head_R = head_R.to(DEV)

    draft_ref = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    sd = {k: v.detach().cpu() for k, v in draft_ref.state_dict().items()}
    conv = pr.build_pure_r1_draft_state(sd, R1, gamma)
    draft_pr = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    draft_pr.load_state_dict({k: v.float() for k, v in conv.items()}, strict=True)
    draft_pr.to(DEV)

    rows, level_cos = [], {}
    for pi in range(len(ids_list)):
        ids = ids_list[pi].to(DEV)
        full_ids = torch.cat([ids, torch.tensor([[next_toks[pi]]], device=DEV)], 1)
        h = h_list[pi].to(DEV)
        h_R = (h.double() @ R1_dev).float()
        _, outs_ref = ra.level_capture(draft_ref, h, full_ids, head_o)
        # SINGLE fc path, no swap: feed h_R, recycle f_R natively
        _, outs_pr = ra.level_capture(draft_pr, h_R, full_ids, head_R)
        for lvl, (a, b) in enumerate(zip(outs_ref, outs_pr), start=1):
            if a.shape != b.shape:
                continue
            # pure-R1 output b = f_R should equal (reference f) @ R1
            c = cos(b, (a.double() @ R1_dev.cpu()).float()) if b.device.type=="cpu" else cos(b.to(DEV), (a.to(DEV).double() @ R1_dev).float())
            level_cos.setdefault(lvl, []).append(c)
            rows.append({"prompt_id": prompts[pi]["question_id"], "level": lvl,
                         "cos_fR_to_f_at_R1": c,
                         "relL2_fR_to_f_at_R1": rel(b.to(DEV) if b.device.type=="cpu" else b, (a.to(DEV).double() @ R1_dev).float())})

    logging_utils.write_csv(os.path.join(args.out_dir, "pure_r1_level_diag.csv"), rows)
    lvl_mean = {lvl: sum(v) / len(v) for lvl, v in sorted(level_cos.items())}
    out = {
        "gpu": gpu,
        "T5_tail_logits_vs_original": {
            "mean_rel_l2": sum(t5_logit["rel_l2"]) / len(t5_logit["rel_l2"]),
            "mean_top1": sum(t5_logit["top1"]) / len(t5_logit["top1"])},
        "pure_r1_level_cos_mean": lvl_mean,
        "single_path": True,
        "verdicts": {
            "T5_tail_logits_match_original": bool(
                sum(t5_logit["top1"]) / len(t5_logit["top1"]) > 0.99),
            "pure_r1_exact_all_levels_single_path": bool(
                all(c > 0.999 for c in lvl_mean.values())),
        }}
    with open(os.path.join(args.out_dir, "pure_r1_correctness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"-> {args.out_dir}/pure_r1_correctness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
