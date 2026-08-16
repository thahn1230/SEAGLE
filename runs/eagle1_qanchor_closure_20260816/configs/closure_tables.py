#!/usr/bin/env python3
"""Closure section 32: Tables A-E as CSVs under results/qanchor_closure/."""
import csv
import json
import os

REPO = "/home/thahn1230/SEAGLE"
NR = os.path.join(REPO, "runs/eagle1_qanchor_closure_20260816")
QRUN = os.path.join(REPO, "runs/eagle1_qat_qanchor_causal_20260814")
OUT = os.path.join(REPO, "results/qanchor_closure")
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")

ct = json.load(open(os.path.join(NR, "tables", "closure_taus.json")))
qt = json.load(open(os.path.join(QRUN, "tables",
                                 "final_taus.json")))["taus"]
dr = json.load(open(os.path.join(NR, "tables", "closure_drift.json")))
try:
    dr.update(json.load(open(os.path.join(NR, "tables",
                                          "seed_drift.json"))))
except FileNotFoundError:
    pass
holm = json.load(open(os.path.join(NR, "stats",
                                   "final_stats_holm.json")))


def T(tag):
    return ct.get(tag) or qt.get(tag) or {}


# ---- Table A: corrected frontier = pareto_frontier_dq.csv (already
# written by pareto_dominance.py; referenced, not duplicated) ----

# ---- Table B: 3-seed finalists ----
FAMS = {
    "HB hybrid b0.5%": ["CLFIN_HB_B_hybrid_b0.005_s0_st3000",
                        "CLFIN_HB_B_hybrid_b0.005_s1_st3000",
                        "CLFIN_HB_B_hybrid_b0.005_s2_st3000"],
    "HB conv b2%": ["CLFIN_HB_B_conv_b0.02_s0_st3000",
                    "CLFIN_HB_B_conv_b0.02_s1_st3000",
                    "CLFIN_HB_B_conv_b0.02_s2_st3000"],
    "plain conv 1e-6 (T6)": ["CLFIN_CAN_T6_gsr5_conv_s0_st1500",
                             "CFIN_T6conv_gsr5_s1",
                             "CLFIN_CAN_T6_gsr5_conv_s2_st2000"],
    "plain hybrid 1e-6 (T6)": ["CLFIN_CAN_T6_gsr5_hybrid_s0_st2500",
                               "CFIN_T6hybrid_gsr5_s1",
                               "CLFIN_CAN_T6_gsr5_hybrid_s2_st2500"],
    "plain conv 1e-5 (P2)": ["CFIN_P2conv_gsr5_s0",
                             "CFIN_P2conv_gsr5_s1",
                             "CFIN_P2conv_gsr5_s2"],
}
with open(os.path.join(OUT, "tableB_seed_finalists.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "s0", "s1", "s2", "seed_mean", "sd", "min",
                "max"])
    for name, tags in FAMS.items():
        vs = [T(t).get("mean4") for t in tags]
        if any(v is None for v in vs):
            w.writerow([name] + [v or "NA" for v in vs] + ["NA"] * 4)
            continue
        mu = sum(vs) / 3
        sd = (sum((v - mu) ** 2 for v in vs) / 2) ** 0.5
        w.writerow([name] + [f"{v:.4f}" for v in vs]
                   + [f"{mu:.4f}", f"{sd:.4f}", f"{min(vs):.4f}",
                      f"{max(vs):.4f}"])

# ---- Table C: W_first counterfactuals ----
part = json.load(open(os.path.join(
    OUT, "wfirst", "flip_partition.json")))
CFROWS = [
    ("PTQ anchor (C0)", "CANON_B9", 0.0, 0.0, 0.0),
    ("HB_controlled_only (== HB_no_Wfirst)", "CF_HB_no_Wfirst",
     None, None, None),
    ("HB_no_WfirstH", "CF_HB_no_WfirstH", None, None, None),
    ("HB_WfirstH_only", "CF_HB_WfirstH_only", None, None, None),
    ("HB_Wfirst_only", "CF_HB_Wfirst_only", None, None, None),
    ("full HB (CHB)", "FIN_HB_B_hybrid_b0.005_s0_st3000",
     0.013014, 0.203263, 3.1e-05),
]
ptq4 = T("CANON_B9")["mean4"]
hb4 = T("FIN_HB_B_hybrid_b0.005_s0_st3000")["mean4"]
with open(os.path.join(OUT, "wfirst", "tableC_counterfactuals.csv"),
          "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["counterfactual", *DS, "mean4", "delta_vs_PTQ",
                "delta_vs_fullHB", "fraction_of_HB_gain_preserved",
                "H_Q_all", "H_Q_Wfirst_h", "H_Q_controlled"])
    for name, tag, hall, hwf, hctl in CFROWS:
        d = T(tag)
        m4 = d.get("mean4")
        frac = (m4 - ptq4) / (hb4 - ptq4) if m4 is not None else None
        w.writerow([name] + [d.get(k, "") for k in DS]
                   + [m4, round(m4 - ptq4, 4), round(m4 - hb4, 4),
                      round(frac, 3),
                      hall if hall is not None else "",
                      hwf if hwf is not None else "",
                      hctl if hctl is not None else ""])

# ---- Table D: selectivity ----
sel = json.load(open(os.path.join(
    OUT, "selectivity", "selectivity_closure.json")))["down_audit"]
with open(os.path.join(OUT, "selectivity", "tableD_selectivity.csv"),
          "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "total_flips", "wfirst_h_flips",
                "wfirst_h_frac_of_flips", "down_flips",
                "down_frac_of_flips", "down_rate_within_down",
                "H_Q_controlled", "H_Q_all"])
    for n, a in sel.items():
        w.writerow([n, a["total_changed_codes"], a["wfirst_h_flips"],
                    a["wfirst_h_fraction_of_all"],
                    a["down_changed_codes"],
                    a["down_fraction_of_all_flips"],
                    a["down_flip_rate_within_down"],
                    a["H_Q_controlled"], a["H_Q_all"]])

# ---- Table E: statistical closure ----
with open(os.path.join(OUT, "tableE_statistics.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["comparison", "dataset", "delta_tau", "ci_low",
                "ci_high", "p", "holm_threshold", "verdict"])
    for fam, v in sorted(holm.items()):
        for ds in DS:
            r = v["per_dataset"][ds]
            verdict = ("WIN" if r["significant"]
                       and r["direction"] == "+"
                       else ("LOSS" if r["significant"] else "n.s."))
            w.writerow([fam, ds, r["delta"], r["ci"][0], r["ci"][1],
                        r["p"], r["holm_threshold"], verdict])
print("tables written under", OUT)
