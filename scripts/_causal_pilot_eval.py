#!/usr/bin/env python
"""Held-out cross-domain diagnostic for a QAT (pilot) checkpoint.

Computes per-DOMAIN (wiki/c4/sharegpt/gsm8k/code) validation metrics on
the canonical corpus val windows (teacher-forced, K=4): alpha_k, KL,
expected tau, plus the QAT losses themselves (SmoothL1 feature loss and
SoftCE) against the same W4A4 teacher hidden/logits. No test prompts
are touched — selection-safe.

Usage: _causal_pilot_eval.py <run_dir> --qat-sd <ck> --r5-ckpt <R5>
        --out-tag <name>
Appends one record to <run>/tables/pilot_eval.jsonl.
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
    ap.add_argument("--qat-sd", default=None)
    ap.add_argument("--r5-ckpt", required=True)
    ap.add_argument("--out-tag", required=True)
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
    if args.qat_sd:
        q = torch.load(args.qat_sd, map_location="cpu",
                       weights_only=False)
        q = q.get("draft_state_dict", q.get("model", q))
        sd = {k: (q[k].to(v.dtype) if k in q else v.clone())
              for k, v in sd.items()}

    corpus = os.path.join(args.run_dir, "manifests",
                          "lkcorpus__rd_t4_all.json")
    windows, _man = load_corpus(corpus, args.run_dir)
    n_val = max(len(windows) // 10, 16)
    W_va = windows[-n_val:]

    core = ExactQATRotatedDraft(
        sd, R_T, gamma, W_lm, SharedRotation(R5).to(dev),
        alpha_init=ALPHA_G, w_bits=4, a_bits=4, draft_kv_bits=16,
        device=dev, first_fold_R=R_T)
    by_dom = {}
    for i0 in range(0, len(W_va), 8):
        ws = W_va[i0:i0 + 8]
        tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
        a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
        teach = torch.stack([w["teacher_tokens"][:K].long()
                             for w in ws]).to(dev)
        zT = torch.stack([w["teacher_logits"][:K].float()
                          for w in ws]).to(dev)
        outs = core.forward_chain(tok, a, K, recur_tokens=teach)
        alphas, kls, ces = [], [], []
        for k, (lg, _h, _c) in enumerate(outs):
            alphas.append(L.overlap_alpha(zT[:, k], lg))
            kls.append(L.kl_full(zT[:, k], lg))
            lp = torch.log_softmax(lg.float(), dim=-1)
            tp = torch.softmax(zT[:, k].float(), dim=-1)
            ces.append(-(tp * lp).sum(-1))
        A = torch.stack(alphas, dim=-1)     # (B, K)
        Kl = torch.stack(kls, dim=-1)
        Ce = torch.stack(ces, dim=-1)
        Et = L.expected_tau(A)
        for b, w in enumerate(ws):
            dom = w.get("domain", "?")
            r = by_dom.setdefault(dom, dict(n=0, alpha=[0.0] * K,
                                            kl=0.0, ce=0.0, et=0.0))
            r["n"] += 1
            for k in range(K):
                r["alpha"][k] += float(A[b, k])
            r["kl"] += float(Kl[b].mean())
            r["ce"] += float(Ce[b].mean())
            r["et"] += float(Et[b])
    rec = dict(tag=args.out_tag, qat_sd=args.qat_sd, domains={})
    for dom, r in sorted(by_dom.items()):
        n = max(r["n"], 1)
        rec["domains"][dom] = dict(
            n=r["n"],
            alpha_k=[round(x / n, 4) for x in r["alpha"]],
            kl=round(r["kl"] / n, 4), softce=round(r["ce"] / n, 4),
            expected_tau=round(r["et"] / n, 4))
    allrec = [v for v in by_dom.values()]
    ntot = sum(v["n"] for v in allrec)
    rec["overall_expected_tau"] = round(
        sum(v["et"] for v in allrec) / max(ntot, 1), 4)
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables",
                           "pilot_eval.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
