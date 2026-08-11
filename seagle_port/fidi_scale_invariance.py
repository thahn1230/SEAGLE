"""FIDI §10: prove/refute quantization-gauge invariance of a GLOBAL scalar at H_t.

Claim (preregistered, manifests/PREREGISTRATION.md): under the deployed
per-token asym A4 (rc.act_fake_ste formula) and per-row sym W4
(rc.rtn_sym_perchannel), the transform

    H_t' = s * H_t,   W_K' = W_K / s,   W_V' = W_V / s      (s > 0)

leaves all integer codes identical (bit-exact for power-of-two s, since both
scales are linear in the extremal values), hence relative NMSE / SQNR / output
error are invariant and a scalar search would burn GPU on a gauge direction.

Setup mirrors the deployed M3/M5 constructions (vsq_mech.py conventions):
H_t built from hcache rows with the R1_T-folded fc, bare RMS norm (γ_hidden is
folded into the ctx K/V weight views exactly as RotQuantDraft does), optional
R_C = R1_T arm. Writes tables/scale_invariance_test.csv into --run-dir.
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .dkva_capture import act_quant_detail

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]


def w_codes(w, bits=4):
    maxq = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq
    q = torch.clamp(torch.round(w / scale), -maxq - 1, maxq)
    return q.to(torch.int8), scale


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-rows", type=int, default=8)
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
    # deployed ctx K/V views: hidden_norm gamma folded into weight columns
    kv = []
    for li, lyr in enumerate(d0.layers):
        wk = lyr.self_attn.k_proj.weight.data.float().to(dev) * gam[None, :]
        wv = lyr.self_attn.v_proj.weight.data.float().to(dev) * gam[None, :]
        kv.append((wk, wv))
    del d0, dF

    files = sorted(glob.glob(f"{args.cache_dir}/row*.npz"))[:args.n_rows]
    Hs = []
    for f in files:
        z = np.load(f)
        Hrot = torch.tensor(z["hidden"], dtype=torch.float32, device=dev)
        Zt = Hrot @ fc_fold.t()
        Hs.append(Zt * torch.rsqrt(Zt.pow(2).mean(-1, keepdim=True) + 1e-6))
    Ht = torch.cat(Hs)                       # bare-normed (gamma in weights)
    print(f"[gauge] H_t {tuple(Ht.shape)} from {len(files)} rows")

    rows = []
    for arm, Rc in (("M3_noRC", None), ("M5_RC_rt", R1d)):
        X = Ht @ Rc if Rc is not None else Ht
        base_act = act_quant_detail(X)
        for li, (wk0, wv0) in enumerate(kv):
            wk = wk0 @ Rc if Rc is not None else wk0
            wv = wv0 @ Rc if Rc is not None else wv0
            bk, bks = w_codes(wk)
            bv, bvs = w_codes(wv)
            y0 = (base_act["dequant"] @ (bk.float() * bks).t())
            for s in SCALES:
                a = act_quant_detail(X * s)
                qk, qks = w_codes(wk / s)
                qv, qvs = w_codes(wv / s)
                y1 = (a["dequant"] @ (qk.float() * qks).t())
                rows.append({
                    "arm": arm, "layer": li, "s": s,
                    "act_codes_equal": bool((a["codes"] == base_act["codes"]).all()),
                    "wk_codes_equal": bool((qk == bk).all()),
                    "wv_codes_equal": bool((qv == bv).all()),
                    "act_nmse": float(a["tok_nmse"].mean()),
                    "act_nmse_base": float(base_act["tok_nmse"].mean()),
                    "y_max_rel_dev": float(((y1 - y0).abs().max()
                                            / (y0.abs().max() + 1e-12))),
                    "scale_clamped_frac": float((a["scale"] <= 1e-8).float().mean()),
                })
                print(rows[-1], flush=True)
    out = os.path.join(args.run_dir, "tables", "scale_invariance_test.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    inv = all(r["act_codes_equal"] and r["wk_codes_equal"]
              and r["wv_codes_equal"] for r in rows if r["s"] != 1.0)
    print(f"[gauge] VERDICT: global scalar is "
          f"{'a pure quantization gauge (CLOSE I5 arm)' if inv else 'NOT gauge-invariant'}")


if __name__ == "__main__":
    main()
