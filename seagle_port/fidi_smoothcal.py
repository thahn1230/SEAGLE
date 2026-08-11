"""FIDI I7: calibrate a per-channel SmoothQuant-like diagonal scale S for the
H_t -> ctx K/V interface (study §10 A/C: can scaling replace or augment R_C?).

s_j = absmax_X(j)^alpha / absmax_W(j)^(1-alpha)  (clamped, geomean-normalized
— the global factor is a proven quantization gauge, see
tables/scale_invariance_test.csv). X = H_t on calib hcache rows; W-column
absmax is the max over all 10 ctx views (5 layers x {K,V}) since H_t is
shared. Two bases: M3 (no R_C) and M5 (post R_C = R1_T). Selection of alpha
by mean ctx K/V output NMSE on the calib rows (proxy only — final claims use
validation AL through eval_al --vsq-ctx-smooth). Writes
tables/i7_smooth_proxy.csv + i7_smooth_{m3,m5}.pt into --run-dir.
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .rc import rtn_sym_perchannel, act_fake_ste

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
ALPHAS = [0.25, 0.5, 0.75, 0.9]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--calib-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel

    d0 = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sq.load_rbin(args.rbin)["R1"]
    dF = interfaces.fold_wc(d0, R1)
    fc_fold = dF.fc.weight.data.float().to(dev)
    gam = d0.hidden_norm.weight.data.float().to(dev)
    R1d = R1.float().to(dev)
    kv = []
    for lyr in d0.layers:
        kv.append((lyr.self_attn.k_proj.weight.data.float().to(dev)
                   * gam[None, :],
                   lyr.self_attn.v_proj.weight.data.float().to(dev)
                   * gam[None, :]))
    del d0, dF

    files = sorted(glob.glob(f"{args.cache_dir}/row*.npz"))[:args.calib_rows]
    Hs = []
    for f in files:
        z = np.load(f)
        Hrot = torch.tensor(z["hidden"], dtype=torch.float32, device=dev)
        Zt = Hrot @ fc_fold.t()
        Hs.append(Zt * torch.rsqrt(Zt.pow(2).mean(-1, keepdim=True) + 1e-6))
    Ht = torch.cat(Hs)
    print(f"[i7] calib H_t {tuple(Ht.shape)}")

    rows, best = [], {}
    for basis, Rc in (("m3", None), ("m5", R1d)):
        X = Ht @ Rc if Rc is not None else Ht
        views = [(wk @ Rc if Rc is not None else wk,
                  wv @ Rc if Rc is not None else wv) for wk, wv in kv]
        w_absmax = torch.stack(
            [w.abs().amax(0) for pair in views for w in pair]).amax(0)
        x_absmax = X.abs().amax(0)

        def nmse_with(s):
            Xq = act_fake_ste(X * s if s is not None else X, 4)
            tot_k = tot_v = 0.0
            for wk, wv in views:
                wkq = rtn_sym_perchannel(
                    wk / s[None, :] if s is not None else wk, 4)
                wvq = rtn_sym_perchannel(
                    wv / s[None, :] if s is not None else wv, 4)
                yk, yv = X @ wk.t(), X @ wv.t()
                tot_k += ((Xq @ wkq.t() - yk).pow(2).sum()
                          / yk.pow(2).sum()).item()
                tot_v += ((Xq @ wvq.t() - yv).pow(2).sum()
                          / yv.pow(2).sum()).item()
            return tot_k / len(views), tot_v / len(views)

        k0, v0 = nmse_with(None)
        rows.append({"basis": basis, "alpha": "none",
                     "k_nmse": k0, "v_nmse": v0})
        print(rows[-1], flush=True)
        cand = []
        for a in ALPHAS:
            s = (x_absmax.clamp(min=1e-6) ** a
                 / w_absmax.clamp(min=1e-6) ** (1 - a)).clamp(1e-3, 1e3)
            s = s / torch.exp(torch.log(s).mean())      # gauge-fix geomean=1
            kn, vn = nmse_with(s)
            rows.append({"basis": basis, "alpha": a,
                         "k_nmse": kn, "v_nmse": vn})
            print(rows[-1], flush=True)
            cand.append((kn + vn, a, s))
        cand.sort(key=lambda t: t[0])
        _, a_best, s_best = cand[0]
        best[basis] = {"alpha": a_best,
                       "improves": cand[0][0] < k0 + v0}
        torch.save({"s": s_best.cpu(), "alpha": a_best, "basis": basis},
                   os.path.join(args.run_dir, "tables",
                                f"i7_smooth_{basis}.pt"))
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables", "i7_smooth_proxy.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[i7] DONE", best)


if __name__ == "__main__":
    main()
