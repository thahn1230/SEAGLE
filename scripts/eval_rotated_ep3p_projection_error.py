#!/usr/bin/env python
"""Detailed projection metrics for finalists (spec §10-§11):
per-config activation/weight distribution stats, code stats, output
error, and — for cross/full rotations — the embedding/hidden
CONTRIBUTION decomposition per rotated channel (never mislabeling
rotated coordinates as e/h regions).

Writes tables/rep3p_projection_metrics.json (+NPZ contributions).
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))
from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 transform_xw,
                                                 scale_vec)

D = 4096


def nmse(y, ref):
    y = y.double(); ref = ref.double()
    return float(((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-30))


def kurt(v):
    v = v.double().flatten()
    m = v.mean(); s = v.std().clamp_min(1e-12)
    return float((((v - m) / s) ** 4).mean())


def act_stats(X):
    a = X.abs()
    pt = a.amax(dim=1)
    q = a.flatten()[:: max(1, a.numel() // 4_000_000)].float()
    return dict(
        rms=float(X.double().pow(2).mean().sqrt()),
        absmax=float(a.max()), meanabs=float(a.mean()),
        p90=float(torch.quantile(q, 0.90)),
        p95=float(torch.quantile(q, 0.95)),
        p99=float(torch.quantile(q, 0.99)),
        p999=float(torch.quantile(q, 0.999)),
        kurtosis=kurt(X),
        per_token_absmax_mean=float(pt.mean()),
        max_over_rms=float(a.max() / X.double().pow(2).mean()
                           .sqrt()))


def act_codes(X):
    mn = X.min(dim=1, keepdim=True).values
    mx = X.max(dim=1, keepdim=True).values
    s = (mx - mn).clamp_min(1e-12) / 15.0
    code = torch.round((X - mn) / s).clamp(0, 15)
    zc = torch.round((0 - mn) / s).clamp(0, 15)
    util = [torch.unique(code[i]).numel() / 16.0
            for i in range(0, code.shape[0],
                           max(1, code.shape[0] // 64))]
    return dict(saturation=float(((code == 0) | (code == 15))
                                 .float().mean()),
                zero_code=float((code == zc).float().mean()),
                utilization=float(sum(util) / len(util)))


def w_row_stats(W, Wq):
    a = W.abs()
    rr = W.double().pow(2).mean(dim=1).sqrt()
    ram = a.amax(dim=1)
    s = ram.clamp_min(1e-12) / 7.0
    code = torch.round(Wq / s[:, None]).clamp(-8, 7)
    util = [torch.unique(code[i]).numel() / 16.0
            for i in range(0, code.shape[0], 64)]
    return dict(row_rms_mean=float(rr.mean()),
                row_absmax_mean=float(ram.mean()),
                row_p99_mean=float(torch.quantile(
                    a.float(), 0.99, dim=1).mean()),
                row_max_over_rms=float((ram / rr.clamp_min(1e-12))
                                       .mean()),
                saturation=float(((code == 7) | (code == -8))
                                 .float().mean()),
                zero_code=float((code == 0).float().mean()),
                utilization=float(sum(util) / len(util)),
                w4_nmse=nmse(Wq, W))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    dev = "cuda:0"
    from eagle_spinquant import fake_w4a4_draft as fq
    tens = torch.load(os.path.join(rd, "tensors", "calib_int4.pt"),
                      map_location="cpu", weights_only=False)
    W0 = tens["W"].float().to(dev)
    bias = tens.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    s5 = json.load(open(os.path.join(rd, "candidates",
                                     "s5_mtbench.json")))
    best = s5["best"]
    out = {}
    npz = {}
    for path, key, ep3p_beta in (("first", "X_first", 0.40),
                                 ("rec", "X_rec_all", 0.45)):
        X = tens[key].float().to(dev)
        cfg = best[path if path == "first" else "rec"]
        rot = StructuredRotation({k: cfg[k] for k in
                                  ("family", "block", "seed",
                                   "interleave_chunk")
                                  if k in cfg}, device=dev)
        ident = StructuredRotation(dict(family="identity"))
        Yref = X @ W0.t() + (b_t if b_t is not None else 0)
        for mname, r, bb in (("original", ident, 0.0),
                             ("ep3p", ident, ep3p_beta),
                             ("rep3p", rot, cfg["beta"])):
            m = float(D ** bb) if bb > 0 else 1.0
            Xt, Wt = transform_xw(X, W0, m, r, "SR")
            Wq = fq._weight_fake_quant(Wt.half(), 4).float()
            aq = fq._act_quantizer(4)
            aq.find_params(Xt.half())
            Xq = aq(Xt.half()).float()
            aq.free()
            Y = Xq @ Wq.t() + (b_t if b_t is not None else 0)
            out[f"{path}_{mname}"] = dict(
                beta=bb, m=m, act=act_stats(Xt),
                act_codes=act_codes(Xt),
                a4_nmse=nmse(Xq, Xt),
                w=w_row_stats(Wt, Wq),
                out_nmse=nmse(Y, Yref),
                out_cos=float(torch.nn.functional.cosine_similarity(
                    Y.flatten(), Yref.flatten(), dim=0)))
        # contribution decomposition for the rotated config
        m = float(D ** cfg["beta"])
        s = scale_vec(2 * D, m, dev, torch.float32)
        Xe = X.clone(); Xe[:, D:] = 0
        Xh = X.clone(); Xh[:, :D] = 0
        Ze = rot.apply(Xe * s)
        Zh = rot.apply(Xh * s)
        ce = Ze.double().pow(2).mean(0).sqrt()
        ch = Zh.double().pow(2).mean(0).sqrt()
        frac = (ce ** 2 / (ce ** 2 + ch ** 2 + 1e-30)).float()
        npz[f"{path}_contrib_e_rms"] = ce.float().cpu().numpy()
        npz[f"{path}_contrib_h_rms"] = ch.float().cpu().numpy()
        npz[f"{path}_contrib_e_frac"] = frac.cpu().numpy()
        out[f"{path}_contribution"] = dict(
            e_frac_mean=float(frac.mean()),
            e_frac_std=float(frac.std()),
            e_frac_min=float(frac.min()),
            e_frac_max=float(frac.max()),
            interpretation="e_frac ~const across rotated channels "
                           "=> energy fully mixed")
    np.savez_compressed(os.path.join(
        rd, "tables", "rep3p_contributions.npz"), **npz)
    json.dump(out, open(os.path.join(
        rd, "tables", "rep3p_projection_metrics.json"), "w"),
        indent=1)
    for k in out:
        if k.endswith("_rep3p"):
            print(f"[projmet] {k}: out_nmse="
                  f"{out[k]['out_nmse']:.4f} "
                  f"a4={out[k]['a4_nmse']:.4f} "
                  f"w4={out[k]['w']['w4_nmse']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
