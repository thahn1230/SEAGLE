#!/usr/bin/env python
"""Depth-wise acceptance-surrogate metrics (k=1..4) for the R6 study.

For each arm (R5-only, R5+R6_PTQ, QAT-weights, QAT-weights+R6_QAT)
computes per-depth alpha_k (full-vocab overlap), KL_k, TV_k and the
expected-tau summary on the SAME held-out validation windows the
trainer used (last 10% of the canonical corpus, teacher-forced).

Usage: _r6_depthwise.py <run_dir> --r5-ckpt <RD_HYB_s2.pt>
         [--r6-ptq <ck>] [--r6-qat <ck>] [--qat-sd <ck>]
Writes <run>/geometry/r6_depthwise.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import SharedRotation
from train_eagle_lk_rotation import load_corpus  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_G = D ** 0.42
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--r5-ckpt", required=True)
    ap.add_argument("--r6-ptq", default=None)
    ap.add_argument("--r6-qat", default=None)
    ap.add_argument("--qat-sd", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    import glob as _g
    from safetensors.torch import load_file, safe_open
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(paths["draft_path"], fn)
        if os.path.exists(p):
            sd = load_file(p) if fn.endswith("safetensors") else \
                torch.load(p, map_location="cpu", weights_only=True)
            break
    gamma = W_lm = None
    for p in sorted(_g.glob(os.path.join(paths["target_path"],
                                         "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    R5 = torch.load(args.r5_ckpt, map_location="cpu",
                    weights_only=False)["R_D"].float()

    corpus = os.path.join(args.run_dir, "manifests",
                          "lkcorpus__rd_t4_all.json")
    windows, _man = load_corpus(corpus, args.run_dir)
    n_val = max(len(windows) // 10, 16)
    W_va = windows[-n_val:][:64]

    qat_sd = None
    if args.qat_sd:
        q = torch.load(args.qat_sd, map_location="cpu",
                       weights_only=False)
        qat_sd = q.get("draft_state_dict", q.get("model", q))

    def make_sd(use_qat):
        base = {k: v.clone() for k, v in sd.items()}
        if use_qat:
            for k in list(base.keys()):
                if k in qat_sd:
                    base[k] = qat_sd[k].to(base[k].dtype)
        return base

    def load_r6(p):
        return (torch.load(p, map_location="cpu",
                           weights_only=False)["R6"].float()
                if p else None)

    arms = [("R5_only", False, None),
            ("R5_plus_R6ptq", False, load_r6(args.r6_ptq))]
    if qat_sd is not None:
        arms += [("QATw_R5", True, None),
                 ("QATw_R5_plus_R6qat", True, load_r6(args.r6_qat))]

    out = {}
    for name, use_qat, R6 in arms:
        if name.endswith("ptq") and R6 is None:
            continue
        rot2 = SharedRotation(R6).to(dev) if R6 is not None else None
        core = ExactQATRotatedDraft(
            make_sd(use_qat), R_T, gamma, W_lm,
            SharedRotation(R5).to(dev), alpha_init=ALPHA_G,
            w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
            first_fold_R=R_T, rot2=rot2)
        agg = {k: dict(alpha=0.0, kl=0.0, tv=0.0, n=0)
               for k in range(K)}
        taus = []
        for i0 in range(0, len(W_va), 8):
            ws = W_va[i0:i0 + 8]
            tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
            a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
            teach = torch.stack([w["teacher_tokens"][:K].long()
                                 for w in ws]).to(dev)
            zT = torch.stack([w["teacher_logits"][:K].float()
                              for w in ws]).to(dev)
            outs = core.forward_chain(tok, a, K, recur_tokens=teach)
            alphas = []
            for k, (lg, _h, _c) in enumerate(outs):
                zTk, zDk = zT[:, k], lg
                al = L.overlap_alpha(zTk, zDk)
                agg[k]["alpha"] += float(al.sum())
                agg[k]["kl"] += float(L.kl_full(zTk, zDk).sum())
                agg[k]["tv"] += float(L.tv(zTk, zDk).sum())
                agg[k]["n"] += al.numel()
                alphas.append(al)
            taus.append(L.expected_tau(torch.stack(alphas, dim=-1)))
        res = {}
        for k in range(K):
            n = max(agg[k]["n"], 1)
            res[f"depth_{k+1}"] = dict(
                alpha=round(agg[k]["alpha"] / n, 5),
                kl=round(agg[k]["kl"] / n, 5),
                tv=round(agg[k]["tv"] / n, 5))
        res["val_expected_tau"] = round(
            float(torch.cat(taus).mean()), 5)
        out[name] = res
        print(f"[depthwise] {name}: {json.dumps(res)}", flush=True)
        del core
        torch.cuda.empty_cache()

    dst = os.path.join(args.run_dir, "geometry", "r6_depthwise.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(dict(n_val_windows=len(W_va), K=K, alpha_gs=ALPHA_G,
                   arms=out), open(dst, "w"), indent=1)
    print(f"[depthwise] -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
