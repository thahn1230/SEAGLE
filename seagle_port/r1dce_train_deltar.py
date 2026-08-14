"""R1DCE §13-14 — C8 residual context rotation:  R_context = R1_D @ DeltaR.

DeltaR: identity-init nn.Linear(4096,4096) wrapped in
nn.utils.parametrizations.orthogonal(orthogonal_map="cayley") — the SAME
mechanism as the validated R1_D/R_C trainers (vsq_train_draft_rot.py,
train_rc.py).  NOT official SpinQuant SGDG/Stiefel: documented deviation —
official SpinQuant optimizes rotations with SGDG on the Stiefel manifold;
this repo's rotation-training infrastructure is Cayley-parametrized Adam,
and C8 stays consistent with it (identical to how R1_D itself was trained).

Objective (deployed-failure proxy, calibration rows only, §14):
    L = mean_layers[ NMSE(K_q, K_fp) + NMSE(V_q, V_fp) ]
      (+ --lambda-h * NMSE(Q_A4(Ht R), Ht R), default 0)
with K_q = A4(Ht R) @ W4((W_K gamma_h) R)^T (STE), K_fp the frozen stock FP
projection.  Everything frozen except DeltaR.  Val = last 15 hcache rows
(same split convention as the R1_D trainer); best-val checkpoint.

Saves {"R_C": R1_D@DeltaR, "DeltaR": ..., "meta", "log"} to --out.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch import nn

from . import interfaces
from . import spinquant_target as sq
from .rc import rtn_sym_perchannel, act_fake_ste

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"
VAL_ROWS = 15


def rms_bare(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def build_data(dev):
    """All H_t tokens (train, val) + frozen ctx views + FP targets."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft
    d0 = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    R1T = sq.load_rbin(S1RBIN)["R1"].float().to(dev)
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1D = ck["R1_D"].float().to(dev)
    R2D = [t.float().to(dev) for t in ck["R2_D"]]
    base = interfaces.fold_wc(d0, R1T)
    rq = RotQuantDraft(base, w_bits=16, a_bits=16, use_r2=True,
                       train_rotations=False, device=dev)
    Wfc = rq.fc_w                                     # folded fc fp32
    files = sorted(glob.glob(f"{VSQ_RD}/hcache_w4a4_s1/row*.npz"))
    tr, va = [], []
    for i, f in enumerate(files):
        H = torch.tensor(np.load(f)["hidden"], dtype=torch.float32,
                         device=dev)
        Ht = rms_bare(H @ Wfc.t())
        (va if i >= len(files) - VAL_ROWS else tr).append(Ht)
    Ht_tr, Ht_va = torch.cat(tr), torch.cat(va)
    views = []
    for i in range(rq.n_layers):
        wk = getattr(rq, f"wk_ctx_{i}").float()
        wv = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(), R2D[i], "out")
        views.append((wk, wv))
    del d0, base, rq
    torch.cuda.empty_cache()
    return Ht_tr, Ht_va, views, R1D


def ctx_loss(Ht, R, views, lam_h=0.0, chunks=1):
    """W4A4 ctx K/V reconstruction NMSE under rotation R (STE grads)."""
    loss = 0.0
    X = Ht @ R
    Xq = act_fake_ste(X, 4)
    for wk, wv in views:
        for W in (wk, wv):
            Wr = rtn_sym_perchannel(W @ R, 4)
            Yq = Xq @ Wr.t()
            Yfp = Ht @ W.t()
            loss = loss + ((Yq - Yfp) ** 2).sum() / ((Yfp ** 2).sum()
                                                     + 1e-12)
    loss = loss / (2 * len(views))
    if lam_h:
        loss = loss + lam_h * ((Xq - X) ** 2).sum() / ((X ** 2).sum()
                                                       + 1e-12)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--lambda-h", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)

    Ht_tr, Ht_va, views, R1D = build_data(dev)
    print(f"[c8] train tokens={Ht_tr.shape[0]} val={Ht_va.shape[0]}",
          flush=True)

    lin = nn.Linear(4096, 4096, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4096))
    delta = nn.utils.parametrizations.orthogonal(
        lin, orthogonal_map="cayley").to(dev)
    opt = torch.optim.Adam(delta.parameters(), lr=args.lr)

    def R_ctx():
        return R1D @ delta.weight

    @torch.no_grad()
    def val_loss():
        return ctx_loss(Ht_va, R_ctx().detach(), views,
                        args.lambda_h).item()

    v0 = val_loss()
    with torch.no_grad():
        base_r1d = ctx_loss(Ht_va, R1D, views).item()
    best, best_step, log = v0, -1, []
    best_state = {k: v.detach().cpu().clone()
                  for k, v in delta.state_dict().items()}
    print(f"[c8] init val ctx-NMSE={v0:.5f} (pure R1_D: {base_r1d:.5f})",
          flush=True)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, Ht_tr.shape[0], (args.batch_tokens,),
                            generator=g)
        loss = ctx_loss(Ht_tr[idx.to(dev)], R_ctx(), views, args.lambda_h)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(delta.parameters(), 1.0)
        opt.step()
        if step % args.val_every == 0 or step == args.steps:
            v = val_loss()
            D = delta.weight.detach()
            dist = (torch.linalg.norm(D - torch.eye(4096, device=dev),
                                      ord="fro") / 64).item()
            log.append({"step": step, "train": loss.item(), "val": v,
                        "deltaR_dist": dist})
            mark = ""
            if v < best:
                best, best_step = v, step
                best_state = {k: t.detach().cpu().clone()
                              for k, t in delta.state_dict().items()}
                mark = " *best"
            print(f"[c8] step {step:4d} train={loss.item():.5f} "
                  f"val={v:.5f} ||dR-I||_F/sqrt(d)={dist:.4f}{mark}",
                  flush=True)

    delta.load_state_dict(best_state)
    D = delta.weight.detach()
    Rfinal = (R1D @ D).cpu()
    ev = torch.linalg.eigvals(D.cpu().double())
    ang = torch.atan2(ev.imag, ev.real).abs()
    stats = {"val_init": v0, "val_pure_r1d": base_r1d, "val_best": best,
             "best_step": best_step,
             "deltaR_dist_fro_over_sqrtd":
                 (torch.linalg.norm(D.cpu() - torch.eye(4096), ord="fro")
                  / 64).item(),
             "rot_angle_max_rad": ang.max().item(),
             "rot_angle_mean_rad": ang.mean().item(),
             "rot_angle_p99_rad": np.percentile(ang.numpy(), 99).item()}
    torch.save({"R_C": Rfinal.float(), "DeltaR": D.cpu().float(),
                "meta": {**vars(args), "optimizer":
                         "Adam+Cayley (repo infra; NOT SpinQuant SGDG)"},
                "log": log, "stats": stats}, args.out)
    json.dump(stats, open(f"{args.run_dir}/tables/"
                          "residual_rotation_stats.json", "w"), indent=1)
    print(f"[c8] saved {args.out}; stats={stats}")


if __name__ == "__main__":
    main()
