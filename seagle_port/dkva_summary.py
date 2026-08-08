"""DKVA §18 + §21 + §28: representative tokens, rotation-effect summary,
final-question table.

- §18: 3 tokens auto-selected by H_t A4 NMSE rank (median / p99 / worst)
  from the C2 mtbench qparam arrays; channel plots of H_t vs H_t R_C and
  their dequants, with per-token qparams annotated. Selection rule is
  fixed BEFORE looking at results: rank by tok_nmse; median = floor(n/2),
  p99 = floor(0.99 n), worst = argmax.
- §21: rotation_effect_summary.csv from capture summaries + aw table.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .dkva_capture import act_quant_detail


def merged(rd, cfg, key):
    out = []
    for p in sorted(glob.glob(f"{rd}/tables/capture__{cfg}__*.json")):
        d = json.load(open(p))
        if key in d["stats"]:
            out.append(d["stats"][key])
    return out


def wavg(items, field):
    n = sum(i["count_rows"] for i in items)
    return sum(i[field] * i["count_rows"] for i in items) / max(n, 1)


def qmerged(rd, cfg, key):
    out = []
    for p in sorted(glob.glob(f"{rd}/tables/capture__{cfg}__*.json")):
        d = json.load(open(p))
        if key in d["qparams"]:
            out.append(d["qparams"][key])
    return out


def qavg(items, field):
    n = sum(i["n_tokens"] for i in items)
    return sum(i[field] * i["n_tokens"] for i in items) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir

    # ---- §21 rotation effect summary
    b = merged(rd, "C2", "C_Ht")
    a = merged(rd, "C3", "C_Ht_rot")
    qb = qmerged(rd, "C2", "C_Ht")
    qa = qmerged(rd, "C3", "C_Ht_rot")
    rows = []

    def add(metric, vb, va):
        rows.append({"metric": metric, "Ht": round(vb, 6),
                     "Ht_RC": round(va, 6),
                     "relative_change": round((va - vb) / (abs(vb) + 1e-12),
                                              4)})
    add("absmax", wavg(b, "absmax"), wavg(a, "absmax"))
    add("abs_p99_9", wavg(b, "abs_p99_9"), wavg(a, "abs_p99_9"))
    add("absmax_over_rms", wavg(b, "absmax_over_rms"),
        wavg(a, "absmax_over_rms"))
    add("kurtosis", wavg(b, "kurtosis"), wavg(a, "kurtosis"))
    add("A4_scale_median", qavg(qb, "scale_median"),
        qavg(qa, "scale_median"))
    add("A4_zero_int_code_ratio", qavg(qb, "zero_int_code_ratio"),
        qavg(qa, "zero_int_code_ratio"))
    add("A4_exact_zero_dequant_ratio",
        qavg(qb, "exact_zero_dequant_ratio"),
        qavg(qa, "exact_zero_dequant_ratio"))
    add("A4_code_entropy_bits", qavg(qb, "code_entropy_bits"),
        qavg(qa, "code_entropy_bits"))
    add("A4_nmse_mean", qavg(qb, "nmse_mean"), qavg(qa, "nmse_mean"))
    aw = f"{rd}/tables/kv_projection_error.csv"
    if os.path.exists(aw):
        r = list(csv.DictReader(open(aw)))
        for pname in ("K", "V"):
            off = np.mean([float(x["AW_nmse"]) for x in r
                           if x["proj"] == pname
                           and x["branch"] == "ctx_RCoff"])
            on = np.mean([float(x["AW_nmse"]) for x in r
                          if x["proj"] == pname
                          and x["branch"] == "ctx_RC1"])
            add(f"{pname}_ctx_AW_NMSE_mean", off, on)
    at = f"{rd}/tables/attention_error.csv"
    if os.path.exists(at):
        r = list(csv.DictReader(open(at)))
        off = np.mean([float(x["score_nmse"]) for x in r
                       if x["arm"] == "w4a4"])
        on = np.mean([float(x["score_nmse"]) for x in r
                      if x["arm"] == "rc1"])
        add("attention_score_NMSE_mean", off, on)
    with open(f"{rd}/tables/rotation_effect_summary.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "Ht", "Ht_RC",
                                          "relative_change"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)

    # ---- §18 representative tokens (rule fixed a priori)
    Ht = np.load(glob.glob(
        f"{rd}/raw/activations/C2__mtbench__C_Ht.npz")[0])["rows"]
    RC1 = torch.load(
        "runs/dflash_seagle_transfer_20260807_180238/rotations/"
        "RC1_reuseRT.pt", weights_only=False)["R_C"].float()
    X = torch.from_numpy(Ht.astype(np.float32))
    det = act_quant_detail(X)
    nm = det["tok_nmse"].numpy()
    order = np.argsort(nm)
    picks = {"median": order[len(order) // 2],
             "p99": order[int(0.99 * len(order))],
             "worst": order[-1]}
    side = []
    for name, idx in picks.items():
        x = X[idx]
        xr = x @ RC1
        d1 = act_quant_detail(x.unsqueeze(0))
        d2 = act_quant_detail(xr.unsqueeze(0))
        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
        for ax, (v, lab) in zip(axes.flat, (
                (x, "H_t"), (xr, "H_t R_C"),
                (d1["dequant"][0], "A4 dequant H_t"),
                (d2["dequant"][0], "A4 dequant H_t R_C"))):
            ax.plot(v.numpy(), lw=0.4)
            ax.set_title(lab, fontsize=9)
        fig.suptitle(f"token[{name}] idx={idx} | "
                     f"NMSE {float(d1['tok_nmse']):.4f} -> "
                     f"{float(d2['tok_nmse']):.4f}", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(f"{rd}/plots/qparams/token_{name}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        for tag, xx, dd in (("orig", x, d1), ("rc", xr, d2)):
            side.append({"token": name, "form": tag, "idx": int(idx),
                         "min": float(xx.min()), "max": float(xx.max()),
                         "absmax": float(xx.abs().max()),
                         "rms": float(xx.pow(2).mean().sqrt()),
                         "scale": float(dd["scale"][0]),
                         "zero_point": float(dd["zero"][0]),
                         "zero_code_frac": dd["zero_int_ratio"],
                         "nmse": float(dd["tok_nmse"])})
    with open(f"{rd}/tables/representative_tokens.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(side[0].keys()))
        w.writeheader(); w.writerows(side)
    print("[dkva_summary] DONE")


if __name__ == "__main__":
    main()
