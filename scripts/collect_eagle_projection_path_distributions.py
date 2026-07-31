#!/usr/bin/env python
"""First/recurrent projection-input distribution statistics (study §7).

Consumes tables/p3exp_capture_<tgt>.pt (raw UNSCALED concat inputs
[N, 8192] for the first and recurrent paths plus unquantized folded
weights) and emits channel/group statistics used by the 3D figures and
the first-vs-recurrent verdict:

  per path (first, recurrent) x slice (e=[:D], h=[D:]) :
    per-channel absmax / rms / p99.9  (saved full-resolution in NPZ)
    64-channel group means of each    (JSON + NPZ)
  weight block grids (64x64 blocks of W_first / W_rec): rms, absmax
  summary ratios: e/h rms ratio per path, first/rec e-slice rms ratio

Writes tables/path_distributions_<tgt>.json + npz alongside.
"""
import argparse, json, os, sys

import numpy as np
import torch

D = 4096
G = 64


def ch_stats(x):
    """x: [N, C] fp32 -> per-channel absmax, rms, p99.9."""
    ax = x.abs().amax(0)
    rms = x.pow(2).mean(0).sqrt()
    p999 = torch.quantile(
        x.abs().float(), 0.999, dim=0) if x.shape[0] < 4000 else \
        torch.quantile(x.abs()[:4000].float(), 0.999, dim=0)
    return ax.numpy(), rms.numpy(), p999.numpy()


def grp(a):
    return a.reshape(-1, G).mean(1)


def blocks(w, fn):
    """w: [R, C] -> [R//G, C//G] blockwise stat."""
    r, c = w.shape
    v = w.reshape(r // G, G, c // G, G).permute(0, 2, 1, 3) \
        .reshape(r // G, c // G, -1)
    return fn(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["fp16", "int4"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    cap = torch.load(os.path.join(
        rd, "tables", f"p3exp_capture_{args.target}.pt"),
        map_location="cpu", weights_only=False)
    out_npz, out_json = {}, dict(target=args.target)
    for path in ("first", "rec"):
        x = cap[path].float()
        for sl, name in ((slice(0, D), "e"), (slice(D, 2 * D), "h")):
            ax, rms, p999 = ch_stats(x[:, sl])
            out_npz[f"{path}_{name}_absmax"] = ax
            out_npz[f"{path}_{name}_rms"] = rms
            out_npz[f"{path}_{name}_p999"] = p999
            out_npz[f"{path}_{name}_absmax_g64"] = grp(ax)
            out_npz[f"{path}_{name}_rms_g64"] = grp(rms)
            out_npz[f"{path}_{name}_p999_g64"] = grp(p999)
            out_json[f"{path}_{name}"] = dict(
                absmax=float(ax.max()), rms=float(np.sqrt(
                    (rms.astype(np.float64) ** 2).mean())),
                p999_mean=float(p999.mean()))
        out_json[f"{path}_e_over_h_rms"] = (
            out_json[f"{path}_e"]["rms"] / out_json[f"{path}_h"]["rms"])
    out_json["first_over_rec_e_rms"] = (
        out_json["first_e"]["rms"] / out_json["rec_e"]["rms"])
    out_json["first_over_rec_h_rms"] = (
        out_json["first_h"]["rms"] / out_json["rec_h"]["rms"])
    for wk in ("W_first", "W_rec"):
        w = cap[wk].float()
        out_npz[f"{wk}_block_rms"] = blocks(
            w, lambda v: v.pow(2).mean(-1).sqrt()).numpy()
        out_npz[f"{wk}_block_absmax"] = blocks(
            w, lambda v: v.abs().amax(-1)).numpy()
        out_json[wk] = dict(
            e_rms=float(w[:, :D].pow(2).mean().sqrt()),
            h_rms=float(w[:, D:].pow(2).mean().sqrt()))
        out_json[f"{wk}_e_over_h_rms"] = (
            out_json[wk]["e_rms"] / out_json[wk]["h_rms"])
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    np.savez(os.path.join(
        rd, "tables", f"path_distributions_{args.target}.npz"),
        **out_npz)
    json.dump(out_json, open(os.path.join(
        rd, "tables", f"path_distributions_{args.target}.json"), "w"),
        indent=1)
    print(f"[pathdist] {args.target}: "
          f"first e/h rms ratio={out_json['first_e_over_h_rms']:.4f} "
          f"rec e/h rms ratio={out_json['rec_e_over_h_rms']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
