"""R1DCE §27 — W4A4-produced context K/V CACHE error (stored-tensor level).

K cache = k_norm(view(K_proj)) then RoPE; RoPE is an orthogonal 2D-pair
rotation per position, so it preserves error norms exactly — the reported
post-k_norm NMSE IS the stored-K-cache NMSE.  V cache = raw projection
(covered by §8 too; repeated here at head granularity for completeness).
Projection accuracy (§8) and cache distribution are distinct quantities —
this table is the cache-error one.  Writes tables/ctx_cache_error.csv.
"""
import argparse
import csv
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .rc import rtn_sym_perchannel, act_fake_ste
from .r1dce_ht_stats import candidates

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"
HD = 128


def rms_head(x, gam, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gam


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft

    z = np.load(f"{args.run_dir}/captures/ht_sample.npz")
    Ht = torch.tensor(z["rows"], dtype=torch.float32, device=dev)
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
    rows = []
    for name, R in candidates(args.run_dir, dev).items():
        X = Ht if R is None else Ht @ R
        Xq = act_fake_ste(X, 4)
        for i in range(rq.n_layers):
            kn = getattr(rq, f"kn_{i}")
            wk = getattr(rq, f"wk_ctx_{i}").float()
            wv = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                              R2D[i], "out")
            if R is not None:
                wk, wv = wk @ R, wv @ R
            Kfp = (Ht @ (getattr(rq, f"wk_ctx_{i}").float()).t()
                   ).view(-1, 8, HD)
            Kq = (Xq @ rtn_sym_perchannel(wk, 4).t()).view(-1, 8, HD)
            ck_fp = rms_head(Kfp, kn)
            ck_q = rms_head(Kq, kn)
            k_nmse = (((ck_q - ck_fp) ** 2).sum()
                      / ((ck_fp ** 2).sum() + 1e-12)).item()
            Vfp0 = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                                R2D[i], "out")
            Vfp = (Ht @ Vfp0.t()).view(-1, 8, HD)
            Vq = (Xq @ rtn_sym_perchannel(wv, 4).t()).view(-1, 8, HD)
            v_nmse = (((Vq - Vfp) ** 2).sum()
                      / ((Vfp ** 2).sum() + 1e-12)).item()
            rows.append({"candidate": name, "layer": i,
                         "Kcache_nmse_postknorm": k_nmse,
                         "Vcache_nmse": v_nmse})
        ks = [r["Kcache_nmse_postknorm"] for r in rows
              if r["candidate"] == name]
        vs = [r["Vcache_nmse"] for r in rows if r["candidate"] == name]
        print(f"[{name:16s}] K-cache NMSE mean={np.mean(ks):.5f} "
              f"V-cache NMSE mean={np.mean(vs):.5f}", flush=True)
    with open(f"{args.run_dir}/tables/ctx_cache_error.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote tables/ctx_cache_error.csv "
          "(RoPE is norm-preserving: post-k_norm == stored-K-cache NMSE)")


if __name__ == "__main__":
    main()
