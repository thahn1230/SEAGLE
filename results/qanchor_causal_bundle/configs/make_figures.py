#!/usr/bin/env python3
"""Pareto + revert-curve figures for the qanchor study.
Reads tables/final_taus.json, tables/final_drift_metrics.json,
tables/gateB_curves.json. Writes figures/*.png + figures/*.csv (raw
data backing every figure, per contract)."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QRUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(QRUN, "figures")
os.makedirs(FIG, exist_ok=True)

taus = json.load(open(os.path.join(QRUN, "tables",
                                   "final_taus.json")))["taus"]
drift = json.load(open(os.path.join(QRUN, "tables",
                                    "final_drift_metrics.json")))

PTQ = taus["CANON_B9"]["mean4"]


def drift_key(base, st):
    return f"{base}.pt.step{st}.pt"


# ---- Pareto: mean4 tau vs H_Q / D_Q / down-flip ----------------------------
rows = []
# anchored candidates
for tag, d in taus.items():
    if "mean4" not in d:
        continue
    if tag.startswith("FIN_HB_B_"):
        base = tag[4:].rsplit("_st", 1)[0]
        st = tag.rsplit("_st", 1)[1]
        fam = "hard-budget"
    elif tag.startswith("FIN_DP_B_"):
        base = tag[4:].rsplit("_st", 1)[0]
        st = tag.rsplit("_st", 1)[1]
        fam = "soft-cell-600"
    elif tag.startswith("FIN_FD_B_"):
        base = tag[4:].rsplit("_st", 1)[0]
        st = tag.rsplit("_st", 1)[1]
        fam = "soft-cell-3000"
    else:
        continue
    dk = drift_key(base, st)
    if dk not in drift:
        continue
    m = drift[dk]
    rows.append(dict(tag=tag, family=fam, mean4=d["mean4"],
                     H_Q=m["H_Q"], D_Q=m["D_Q"],
                     down_flip=m["per_site"]["down"]["flip"]))
# reference points
rows.append(dict(tag="PTQ (anchor c0)", family="ptq", mean4=PTQ,
                 H_Q=0.0, D_Q=0.0, down_flip=0.0))
agg = drift.get("CAN_P2_B_conv_s0.pt.step3000.pt")
if agg:
    p2 = taus.get("CFIN_P2conv_gsr5_s0", {})
    if "mean4" in p2:
        rows.append(dict(tag="plain QAT 1e-5 (s0)", family="plain-qat",
                         mean4=p2["mean4"], H_Q=agg["H_Q"],
                         D_Q=agg["D_Q"],
                         down_flip=agg["per_site"]["down"]["flip"]))

with open(os.path.join(FIG, "pareto_data.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

COL = {"hard-budget": "tab:red", "soft-cell-600": "tab:blue",
       "soft-cell-3000": "tab:cyan", "ptq": "black",
       "plain-qat": "tab:gray"}
MRK = {"hard-budget": "o", "soft-cell-600": "s",
       "soft-cell-3000": "^", "ptq": "*", "plain-qat": "X"}
for xkey, xlabel, fname in [
        ("H_Q", "code flip rate H_Q vs anchor", "pareto_tau_vs_HQ.png"),
        ("D_Q", "dequantized drift D_Q (frozen scales)",
         "pareto_tau_vs_DQ.png"),
        ("down_flip", "down_proj flip rate", "pareto_tau_vs_downflip.png")]:
    plt.figure(figsize=(7, 5))
    seen = set()
    for r in rows:
        lb = r["family"] if r["family"] not in seen else None
        seen.add(r["family"])
        plt.scatter(r[xkey], r["mean4"], c=COL[r["family"]],
                    marker=MRK[r["family"]], s=70, label=lb, zorder=3)
    plt.axhline(PTQ, color="k", lw=0.8, ls="--", alpha=0.6)
    plt.xscale("symlog", linthresh=1e-3)
    plt.xlabel(xlabel)
    plt.ylabel("official micro-tau mean4 (4 datasets)")
    plt.title("Anchored-QAT Pareto frontier (W4A4 KV16, GS+R5)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close()

# ---- Gate B revert curve ---------------------------------------------------
gb = json.load(open(os.path.join(QRUN, "tables", "gateB_curves.json")))
pts = []
for v in gb:
    if isinstance(v, dict) and "mean4" in v and "H_Q" in v:
        pts.append((v["H_Q"], v["mean4"], v.get("name", "")))
if pts:
    pts.sort()
    plt.figure(figsize=(7, 5))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.scatter(xs, ys, c=["tab:red" if "nondown" in p[2]
                           else ("tab:orange" if "down" in p[2]
                                 else "tab:blue") for p in pts], s=45)
    plt.axhline(PTQ, color="k", lw=0.8, ls="--", alpha=0.6,
                label=f"PTQ {PTQ}")
    plt.xlabel("retained flip rate H_Q")
    plt.ylabel("mean4 micro-tau")
    plt.title("Gate B: constructed-code revert/interpolation curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "gateB_revert_curve.png"), dpi=150)
    plt.close()
    with open(os.path.join(FIG, "gateB_curve_data.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "H_Q", "mean4"])
        for x, y, n in pts:
            w.writerow([n, x, y])
print(f"figures -> {FIG}: " + ", ".join(sorted(os.listdir(FIG))))
