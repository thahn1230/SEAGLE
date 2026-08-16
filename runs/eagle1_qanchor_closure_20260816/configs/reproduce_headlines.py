#!/usr/bin/env python3
"""Closure section 3: recompute headline micro-taus from RAW acceptance
lists (not summary JSON) and compare against the reported values.
Writes results/qanchor_closure/reproduction/current_headline_reproduction.csv
STOPS (exit 1) if any headline mean4 deviates beyond roundoff."""
import ast
import csv
import json
import os
import sys

REPO = "/home/thahn1230/SEAGLE"
QSH = os.path.join(REPO, "runs/eagle1_qat_qanchor_causal_20260814/shards")
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")

# method -> (shard filename pattern per ds, reported per-ds, reported mean4)
REP = json.load(open(os.path.join(
    REPO, "runs/eagle1_qat_qanchor_causal_20260814/tables/final_taus.json")))["taus"]

METHODS = {
    "GS+R5 PTQ": ("al__CANON_B9__int4__{ds}.csv", "CANON_B9"),
    "plain QAT 1e-5 (s0)": ("al__CFIN_P2conv_gsr5_s0__int4__{ds}.csv",
                            "CFIN_P2conv_gsr5_s0"),
    "HB conv b2% (s0)": (
        "al__FIN_HB_B_conv_b0.02_s0_st3000__{ds}__int4__{ds}.csv",
        "FIN_HB_B_conv_b0.02_s0_st3000"),
    "HB hybrid b0.5% (s0)": (
        "al__FIN_HB_B_hybrid_b0.005_s0_st3000__{ds}__int4__{ds}.csv",
        "FIN_HB_B_hybrid_b0.005_s0_st3000"),
    "code-frozen LoRA r16": (
        "al__FIN_PC_B_r16_lr1e-4_s0_st1500__{ds}__int4__{ds}.csv",
        "FIN_PC_B_r16_lr1e-4_s0_st1500"),
    "best soft-cell (DP hybrid b0.1 s1)": (
        "al__FIN_DP_B_hybrid_beta0.1_s1_st0600__{ds}__int4__{ds}.csv",
        "FIN_DP_B_hybrid_beta0.1_s1_st0600"),
}

out = os.path.join(REPO,
                   "results/qanchor_closure/reproduction/"
                   "current_headline_reproduction.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
rows = []
fail = False
for name, (pat, rtag) in METHODS.items():
    m4 = []
    for ds in DS:
        p = os.path.join(QSH, pat.format(ds=ds))
        tot = cyc = 0
        with open(p) as fh:
            for row in csv.DictReader(fh):
                al = ast.literal_eval(row["acceptance_list"])
                tot += sum(al)
                cyc += len(al)
        micro = tot / cyc
        rep = REP.get(rtag, {}).get(ds)
        diff = abs(micro - rep) if rep is not None else float("nan")
        ok = rep is not None and diff < 5e-5
        fail |= not ok
        rows.append(dict(method=name, dataset=ds, sum_tau=tot,
                         cycles=cyc, micro_tau=round(micro, 6),
                         reported_tau=rep,
                         absolute_difference=round(diff, 6),
                         **{"pass": ok}))
        m4.append(micro)
    mean4 = sum(m4) / 4
    rep4 = REP.get(rtag, {}).get("mean4")
    d4 = abs(mean4 - rep4)
    ok4 = d4 < 5e-5
    fail |= not ok4
    rows.append(dict(method=name, dataset="mean4", sum_tau="",
                     cycles="", micro_tau=round(mean4, 6),
                     reported_tau=rep4, absolute_difference=round(d4, 6),
                     **{"pass": ok4}))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}")
for r in rows:
    if r["dataset"] == "mean4":
        print(f"{r['method']}: raw={r['micro_tau']} reported="
              f"{r['reported_tau']} diff={r['absolute_difference']} "
              f"pass={r['pass']}")
sys.exit(1 if fail else 0)
