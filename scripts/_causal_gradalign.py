#!/usr/bin/env python
"""Gradient alignment between the QAT objective and the acceptance-aware
LK objective, at the SAME R5-PTQ initialization, no optimizer updates.

g_QAT = grad[SmoothL1(pred_hidden, teacher_hidden) + 0.1*SoftCE]
g_ACC = grad[hybrid LK surrogate (lambda KL + (1-lambda) TV, K=4,
        w_k = 0.8^(k-1))]

Both computed on identical held-out corpus val windows (diagnostic set;
never test prompts), through the same W4A4 exact core. Reports per-CORE-
tensor cosine/dot/norm-ratio distributions over batches, overall and by
domain.

Usage: _causal_gradalign.py <run_dir> --r5-ckpt <R5> [--batches 24]
Writes <run>/geometry/grad_alignment.json.
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
CORE = ("W_e", "W_h", "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown")
GROUPS = dict(projection=("W_e", "W_h"),
              attention=("Wq", "Wk", "Wv", "Wo"),
              mlp=("Wgate", "Wup", "Wdown"),
              ar=("Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown"),
              full=CORE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--r5-ckpt", required=True)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = "cuda:0"
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
    core = ExactQATRotatedDraft(
        sd, R_T, gamma, W_lm, SharedRotation(R5).to(dev),
        alpha_init=ALPHA_G, w_bits=4, a_bits=4, draft_kv_bits=16,
        train_draft_core=True, device=dev, first_fold_R=R_T)
    core.log_alpha.requires_grad_(False)
    params = {n: getattr(core, n) for n in CORE}

    corpus = os.path.join(args.run_dir, "manifests",
                          "lkcorpus__rd_t4_all.json")
    windows, _man = load_corpus(corpus, args.run_dir)
    n_val = max(len(windows) // 10, 16)
    W_va = windows[-n_val:]
    wk = L.depth_weights(K, 0.8, device=dev)

    def grads_for(loss):
        for p in params.values():
            p.grad = None
        loss.backward()
        return {n: (p.grad.detach().clone() if p.grad is not None
                    else torch.zeros_like(p)) for n, p in params.items()}

    def batch(ws):
        tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
        a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
        teach = torch.stack([w["teacher_tokens"][:K].long()
                             for w in ws]).to(dev)
        zT = torch.stack([w["teacher_logits"][:K].float()
                          for w in ws]).to(dev)
        return tok, a, teach, zT

    recs = []
    for b0 in range(0, min(len(W_va), args.batches * args.bs), args.bs):
        ws = W_va[b0:b0 + args.bs]
        if len(ws) < 2:
            break
        tok, a, teach, zT = batch(ws)
        dom = [w.get("domain", "?") for w in ws]

        # g_ACC: hybrid LK over K depths
        outs = core.forward_chain(tok, a, K, recur_tokens=teach)
        acc_loss = 0.0
        h_pred_k1 = None
        for k, (lg, h, _c) in enumerate(outs):
            lk_, _lam, _al = L.hybrid_lk(zT[:, k], lg, eta=3.0)
            acc_loss = acc_loss + wk[k] * lk_.mean()
            if k == 0:
                h_pred_k1 = h
        gA = grads_for(acc_loss)

        # g_QAT: SmoothL1 on depth-1 hidden vs the R5-basis teacher
        # hidden (= a_seq rotated: teacher-forced next-feature target,
        # matching the canonical trainer's regression target semantics)
        outs = core.forward_chain(tok, a, K, recur_tokens=teach)
        lg1, h1, _ = outs[0]
        R5d = core.rot.R().to(h1.dtype)
        tgt_h = (a @ R5d)[:, -h1.shape[1]:, :] if h1.dim() == 3 else \
            (a @ R5d)
        vloss = torch.nn.functional.smooth_l1_loss(
            h1.float(), tgt_h.float().detach())
        tp = torch.softmax(zT[:, 0].float(), dim=-1)
        lp = torch.log_softmax(lg1.float(), dim=-1)
        ploss = -(tp * lp).sum(-1).mean()
        gQ = grads_for(1.0 * vloss + 0.1 * ploss)

        rec = dict(domains=dom, per_tensor={})
        for n in CORE:
            a_, q_ = gA[n].flatten(), gQ[n].flatten()
            na, nq = float(a_.norm()), float(q_.norm())
            cos = float((a_ @ q_) / (na * nq + 1e-12))
            rec["per_tensor"][n] = dict(cos=round(cos, 4),
                                        norm_acc=na, norm_qat=nq)
        for gname, members in GROUPS.items():
            a_ = torch.cat([gA[n].flatten() for n in members])
            q_ = torch.cat([gQ[n].flatten() for n in members])
            cos = float((a_ @ q_) /
                        (a_.norm() * q_.norm() + 1e-12))
            rec[f"group_{gname}_cos"] = round(cos, 4)
        recs.append(rec)
        print(f"[gradalign] batch {len(recs)}: full-cos "
              f"{rec['group_full_cos']:+.4f}", flush=True)

    import statistics as st
    summary = {}
    for gname in GROUPS:
        xs = [r[f"group_{gname}_cos"] for r in recs]
        summary[gname] = dict(mean=round(st.mean(xs), 4),
                              median=round(st.median(xs), 4),
                              min=round(min(xs), 4),
                              max=round(max(xs), 4),
                              frac_negative=round(
                                  sum(x < 0 for x in xs) / len(xs), 3))
    per_t = {}
    for n in CORE:
        xs = [r["per_tensor"][n]["cos"] for r in recs]
        per_t[n] = dict(mean=round(st.mean(xs), 4),
                        frac_negative=round(
                            sum(x < 0 for x in xs) / len(xs), 3))
    # domain-conditional full-model cosine (batch majority domain)
    by_dom = {}
    for r in recs:
        d = max(set(r["domains"]), key=r["domains"].count)
        by_dom.setdefault(d, []).append(r["group_full_cos"])
    dom_sum = {d: dict(mean=round(st.mean(v), 4), n=len(v))
               for d, v in by_dom.items()}

    dst = os.path.join(args.run_dir, "geometry", "grad_alignment.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(dict(n_batches=len(recs), groups=summary,
                   per_tensor=per_t, by_batch_majority_domain=dom_sum,
                   caveat="g_QAT feature target approximates the "
                          "trainer's masked hidden regression with the "
                          "corpus-window rotated next-feature; SoftCE "
                          "term exact"),
              open(dst, "w"), indent=1)
    print("[gradalign] ->", dst)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
