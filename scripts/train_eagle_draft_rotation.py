#!/usr/bin/env python
"""Phase G trainer: learn a draft rotation R_D against cached deployed-
target teachers (teacher-forced trajectories; optional free-running stage
uses the live deployed target).

Objectives (spec §18): deployKL (top-64 restricted, temperature tau,
tau^2-scaled), targetCE, rank margin, feature NMSE (target hidden in the
draft gauge, gamma-correct), selfKL (live FP16 draft on the same windows),
FPtargetKL (from the fp16 cache when present). Depth weights gamma^k.

Optimizer: Adam on R_D + QR retraction after every step (repository-
equivalent orthogonal optimization; ||R^TR-I||_F tracked, Gate G abort at
1e-3). Draft/target weights frozen. Optional trainable folded alpha.
"""
import argparse, json, math, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F
from eagle_spinquant import experiment, study
from eagle_spinquant.draft_rotation import RotatedDraftTrainer

KIND = "learned_chat_w4a4kv16"


def topk_kl(logits, tv, ti, tau):
    """KL(p_T || p_D) restricted to the teacher's top-K support (renormed),
    temperature tau, scaled by tau^2."""
    pT = F.softmax(tv.float() / tau, dim=-1)
    zD = torch.gather(logits, -1, ti.long())
    logqD = F.log_softmax(zD / tau, dim=-1)
    return (tau ** 2) * (pT * (pT.clamp_min(1e-9).log() - logqD)) \
        .sum(-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fp16-cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="RT",
                    choices=["RT", "identity", "hadamard", "random"])
    ap.add_argument("--objective", default="deployKL+rank+feature+self",
                    help="'+'-joined of deployKL,targetCE,rank,feature,"
                         "self,fptarget,selfrecon")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--alpha-init", type=float, default=32.0)
    ap.add_argument("--train-alpha", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    torch.manual_seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    W = cache["windows"]
    K = cache["meta"]["K"]
    n_val = max(len(W) // 10, 8)
    W_tr, W_va = W[:-n_val], W[-n_val:]
    fp_cache = None
    if args.fp16_cache and os.path.exists(args.fp16_cache):
        fp_cache = torch.load(args.fp16_cache, map_location="cpu",
                              weights_only=False)["windows"]
    print(f"[train] {len(W_tr)} train / {len(W_va)} valid windows; "
          f"objective={args.objective}", flush=True)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    from safetensors.torch import load_file
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(paths["draft_path"], fn)
        if os.path.exists(p):
            sd = load_file(p) if fn.endswith("safetensors") else \
                torch.load(p, map_location="cpu", weights_only=True)
            break
    import glob
    from safetensors.torch import safe_open
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    model = RotatedDraftTrainer(sd, R["R1"], gamma, W_lm, w_bits=4,
                                a_bits=4, alpha_init=args.alpha_init,
                                train_alpha=args.train_alpha,
                                init=args.init, device=dev)
    # FP16 reference draft (identity gauge, no quant) for self/feature refs
    ref = RotatedDraftTrainer(sd, R["R1"], gamma, W_lm, w_bits=16,
                              a_bits=16, alpha_init=1.0, init="identity",
                              device=dev)
    ref.R_D.data = torch.eye(model.D, device=dev)
    for p_ in ref.parameters():
        p_.requires_grad_(False)

    params = [model.R_D] + ([model.log_alpha] if args.train_alpha else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    obj = set(args.objective.split("+"))
    LAM = dict(deployKL=1.0, targetCE=0.5, rank=0.3, feature=0.5,
               self=0.2, fptarget=0.3, selfrecon=1.0)
    wk = torch.tensor([args.gamma ** k for k in range(K)], device=dev)
    wk = wk / wk.sum()

    def batch(ws, idxs):
        tok = torch.stack([ws[i]["tok_ids"] for i in idxs]).long().to(dev)
        a = torch.stack([ws[i]["a_seq"] for i in idxs]).float().to(dev)
        tv = torch.stack([ws[i]["deploy_topk_v"] for i in idxs]) \
            .float().to(dev)
        ti = torch.stack([ws[i]["deploy_topk_i"] for i in idxs]).to(dev)
        hn = torch.stack([ws[i]["h_next"] for i in idxs]).float().to(dev)
        return tok, a, tv, ti, hn

    def losses(tok, a, tv, ti, hn, fp_tv=None, fp_ti=None):
        outs = model.unroll(tok, a, K)
        with torch.no_grad():
            refs = ref.unroll(tok, a @ torch.eye(model.D, device=dev), K) \
                if False else None
        # target hidden in draft gauge: h_T·R_D with gamma semantics:
        # cached a = n·R_T (exposed);  h_T = n·D_γ  =>
        # h_T·R_D = a·R_Tᵀ·D_γ·R_D
        RtT = model.R_T.t()
        total = 0.0
        parts = {}
        for k, (lg, hg) in enumerate(outs):
            if "deployKL" in obj:
                l = topk_kl(lg, tv[:, k], ti[:, k], args.tau)
                total = total + LAM["deployKL"] * wk[k] * l
                parts[f"deployKL{k}"] = float(l)
            if "targetCE" in obj:
                y = ti[:, k, 0].long()
                l = F.cross_entropy(lg, y)
                total = total + LAM["targetCE"] * wk[k] * l
            if "rank" in obj:
                y = ti[:, k, 0].long()
                zy = lg.gather(1, y[:, None]).squeeze(1)
                lg_m = lg.scatter(1, y[:, None], float("-inf"))
                l = F.relu(args.margin - zy + lg_m.max(1).values).mean()
                total = total + LAM["rank"] * wk[k] * l
            if "feature" in obj:
                hT_D = (hn[:, k] @ RtT) * model.gamma
                hT_D = hT_D @ model.R_D
                e = hg.squeeze(1) - hT_D
                l = (e ** 2).sum() / (hT_D ** 2).sum().clamp_min(1e-9)
                total = total + LAM["feature"] * wk[k] * l
                parts[f"featNMSE{k}"] = float(l)
            if "fptarget" in obj and fp_tv is not None:
                l = topk_kl(lg, fp_tv[:, k], fp_ti[:, k], args.tau)
                total = total + LAM["fptarget"] * wk[k] * l
        if "self" in obj or "selfrecon" in obj:
            with torch.no_grad():
                fouts = ref.unroll(tok, a, K)
            lam = LAM["selfrecon"] if "selfrecon" in obj else LAM["self"]
            for k, ((lg, hg), (flg, fhg)) in enumerate(zip(outs, fouts)):
                pF = F.softmax(flg / args.tau, -1)
                l = (args.tau ** 2) * F.kl_div(
                    F.log_softmax(lg / args.tau, -1), pF,
                    reduction="batchmean")
                total = total + lam * wk[k] * l
        return total, parts

    g = torch.Generator().manual_seed(args.seed)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        idxs = torch.randint(0, len(W_tr), (args.batch,), generator=g) \
            .tolist()
        tok, a, tv, ti, hn = batch(W_tr, idxs)
        fp_tv = fp_ti = None
        if fp_cache is not None and "fptarget" in obj:
            fp_tv = torch.stack([fp_cache[i]["deploy_topk_v"]
                                 for i in idxs]).float().to(dev)
            fp_ti = torch.stack([fp_cache[i]["deploy_topk_i"]
                                 for i in idxs]).to(dev)
        total, parts = losses(tok, a, tv, ti, hn, fp_tv, fp_ti)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        model.retract()
        oerr = model.orthogonality_error()
        assert oerr < 1e-3, f"Gate G: orthogonality error {oerr}"
        if step % 20 == 0 or step == args.steps - 1:
            log.append(dict(step=step, loss=float(total), orth=oerr,
                            alpha=float(model.log_alpha.exp()), **parts))
            print(f"[train] step {step}: loss={float(total):.4f} "
                  f"orth={oerr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    # validation quick metrics
    with torch.no_grad():
        tok, a, tv, ti, hn = batch(W_va, list(range(len(W_va))))
        outs = model.unroll(tok, a, K)
        top1 = [float((lg.argmax(-1) == ti[:, k, 0].long()).float().mean())
                for k, (lg, _h) in enumerate(outs)]
    import hashlib
    torch.save(dict(R_D=model.R_D.data.cpu(),
                    alpha=float(model.log_alpha.exp()),
                    meta=dict(init=args.init, objective=args.objective,
                              steps=args.steps, lr=args.lr, tau=args.tau,
                              gamma=args.gamma, seed=args.seed,
                              cache=os.path.basename(args.cache),
                              val_top1_by_depth=top1,
                              orth_final=model.orthogonality_error()),
                    log=log), args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    open(args.out + ".sha256", "w").write(sha + "\n")
    print(f"[train] saved {args.out} sha={sha[:16]} "
          f"val_top1={ [round(t,3) for t in top1] }", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
