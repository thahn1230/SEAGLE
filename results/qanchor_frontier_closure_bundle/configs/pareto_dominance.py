#!/usr/bin/env python3
"""Closure sections 27-28: computed (not eyeballed) Pareto dominance.
Assembles every frontier point (tau mean4 + uniform drift), determines
non-dominated sets under (max tau, min D_Q) and (max tau, min H_Q_all),
and separately under H_Q_controlled (diagnostic). Writes
results/qanchor_closure/frontier/pareto_frontier_{dq,hq,hqc}.csv and
plain_lr_frontier.csv (section 5)."""
import csv
import json
import os

REPO = "/home/thahn1230/SEAGLE"
NR = os.path.join(REPO, "runs/eagle1_qanchor_closure_20260816")
QRUN = os.path.join(REPO, "runs/eagle1_qat_qanchor_causal_20260814")
OUT = os.path.join(REPO, "results/qanchor_closure/frontier")
os.makedirs(OUT, exist_ok=True)

ct = json.load(open(os.path.join(NR, "tables", "closure_taus.json")))
qt = json.load(open(os.path.join(QRUN, "tables",
                                 "final_taus.json")))["taus"]
dr = json.load(open(os.path.join(NR, "tables", "closure_drift.json")))
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")


def drift(ck):
    m = dr.get(ck)
    if not m:
        return None
    ps = m["per_site"]
    return dict(H_Q_all=m["H_Q_all"], H_Q_controlled=m["H_Q_controlled"],
                H_Q_Wfirst=ps["W_first"]["flip"],
                H_Q_Wfirst_e=ps["W_first_e"]["flip"],
                H_Q_Wfirst_h=ps["W_first_h"]["flip"],
                H_Q_down=ps["down"]["flip"], D_Q=m["D_Q"],
                D_FP=m["D_FP"])


def tau(src, tag):
    d = src.get(tag, {})
    return d if "mean4" in d else None


# (method, family, objective, lr, budget/beta, seed, tau-src, tau-tag,
#  drift ckpt key, reused?)
ROWS = [
    ("PTQ GS+R5", "ptq", "-", "-", "-", "-", qt, "CANON_B9", None, True),
    ("plain conv 1e-5 s0", "plain", "conv", "1e-5", "-", 0, qt,
     "CFIN_P2conv_gsr5_s0", "CAN_P2_B_conv_s0.pt.step3000.pt", True),
    ("plain conv 1e-5 s1", "plain", "conv", "1e-5", "-", 1, qt,
     "CFIN_P2conv_gsr5_s1", "CAN_P2_B_conv_s1.pt.step1000.pt", True),
    ("plain conv 1e-5 s2", "plain", "conv", "1e-5", "-", 2, qt,
     "CFIN_P2conv_gsr5_s2", "CAN_P2_B_conv_s2.pt.step2500.pt", True),
    ("plain hybrid 1e-5 s0", "plain", "hybrid", "1e-5", "-", 0, None,
     "CFIN_P2hybrid_gsr5_s0", "CAN_P2_B_hybrid_s0.pt.step1000.pt", True),
    ("plain conv 1e-6 s0", "plain", "conv", "1e-6", "-", 0, ct,
     "CLFIN_CAN_T6_gsr5_conv_s0_st1500",
     "CAN_T6_gsr5_conv_s0.pt.step1500.pt", False),
    ("plain conv 1e-6 s1", "plain", "conv", "1e-6", "-", 1, None,
     "CFIN_T6conv_gsr5_s1", "CAN_T6_gsr5_conv_s1.pt.step1500.pt", True),
    ("plain conv 1e-6 s2", "plain", "conv", "1e-6", "-", 2, ct,
     "CLFIN_CAN_T6_gsr5_conv_s2_st2000",
     "CAN_T6_gsr5_conv_s2.pt.step2000.pt", False),
    ("plain hybrid 1e-6 s0", "plain", "hybrid", "1e-6", "-", 0, ct,
     "CLFIN_CAN_T6_gsr5_hybrid_s0_st2500",
     "CAN_T6_gsr5_hybrid_s0.pt.step2500.pt", False),
    ("plain hybrid 1e-6 s1", "plain", "hybrid", "1e-6", "-", 1, None,
     "CFIN_T6hybrid_gsr5_s1", "CAN_T6_gsr5_hybrid_s1.pt.step1000.pt",
     True),
    ("plain hybrid 1e-6 s2", "plain", "hybrid", "1e-6", "-", 2, ct,
     "CLFIN_CAN_T6_gsr5_hybrid_s2_st2500",
     "CAN_T6_gsr5_hybrid_s2.pt.step2500.pt", False),
    ("plain conv 3e-6 s1", "plain", "conv", "3e-6", "-", 1, ct,
     "CLFIN_QA_LRF_B_conv_3e-6_s1_st1000",
     "QA_LRF_B_conv_3e-6_s1.pt.step1000.pt", False),
    ("plain conv 3e-7 s0", "plain", "conv", "3e-7", "-", 0, ct,
     "CLFIN_QA_LRF_B_conv_3e-7_s0_st1000",
     "QA_LRF_B_conv_3e-7_s0.pt.step1000.pt", False),
    ("plain hybrid 3e-6 s0", "plain", "hybrid", "3e-6", "-", 0, ct,
     "CLFIN_QA_LRF_B_hybrid_3e-6_s0_st1500",
     "QA_LRF_B_hybrid_3e-6_s0.pt.step1500.pt", False),
    ("plain hybrid 3e-7 s2", "plain", "hybrid", "3e-7", "-", 2, ct,
     "CLFIN_QA_LRF_B_hybrid_3e-7_s2_st1500",
     "QA_LRF_B_hybrid_3e-7_s2.pt.step1500.pt", False),
    ("soft-cell conv b1 s2 (600)", "soft-cell", "conv", "1e-5",
     "beta1", 2, qt, "FIN_DP_B_conv_beta1_s2_st0600",
     "DP_B_conv_beta1_s2.pt.step0600.pt", True),
    ("soft-cell hybrid b0.1 s2 (600)", "soft-cell", "hybrid", "1e-5",
     "beta0.1", 2, qt, "FIN_DP_B_hybrid_beta0.1_s2_st0600",
     "DP_B_hybrid_beta0.1_s2.pt.step0600.pt", True),
    ("HB hybrid b0.5% s0", "hard-budget", "hybrid", "1e-5", "0.005", 0,
     qt, "FIN_HB_B_hybrid_b0.005_s0_st3000",
     "HB_B_hybrid_b0.005_s0.pt.step3000.pt", True),
    ("HB conv b2% s0", "hard-budget", "conv", "1e-5", "0.02", 0, qt,
     "FIN_HB_B_conv_b0.02_s0_st3000",
     "HB_B_conv_b0.02_s0.pt.step3000.pt", True),
    ("HB hybrid b0.5% s1", "hard-budget", "hybrid", "1e-5", "0.005", 1,
     ct, "CLFIN_HB_B_hybrid_b0.005_s1_st3000",
     "HB_B_hybrid_b0.005_s1.pt.step3000.pt", False),
    ("HB hybrid b0.5% s2", "hard-budget", "hybrid", "1e-5", "0.005", 2,
     ct, "CLFIN_HB_B_hybrid_b0.005_s2_st3000",
     "HB_B_hybrid_b0.005_s2.pt.step3000.pt", False),
    ("HB conv b2% s1", "hard-budget", "conv", "1e-5", "0.02", 1, ct,
     "CLFIN_HB_B_conv_b0.02_s1_st3000",
     "HB_B_conv_b0.02_s1.pt.step3000.pt", False),
    ("HB conv b2% s2", "hard-budget", "conv", "1e-5", "0.02", 2, ct,
     "CLFIN_HB_B_conv_b0.02_s2_st3000",
     "HB_B_conv_b0.02_s2.pt.step3000.pt", False),
    ("code-frozen LoRA r16", "lora", "-", "1e-4", "r16", 0, qt,
     "FIN_PC_B_r16_lr1e-4_s0_st1500", None, True),
    ("JB hybrid b0.5%+Wfirst-cell s0", "joint-basis", "hybrid", "1e-5",
     "0.005/lf1", 0, ct, "CLFIN_JB_B_hybrid_b0.005_lf1_s0_st3000",
     "JB_B_hybrid_b0.005_lf1_s0.pt.step3000.pt", False),
]

pts = []
for (name, fam, obj, lr, bud, seed, src, tag, dk, reused) in ROWS:
    d = tau(src if src is not None else ct, tag) or tau(qt, tag) \
        or tau(ct, tag)
    if d is None:
        print(f"  [skip] {name}: no tau ({tag})")
        continue
    dd = drift(dk) if dk else (dict(H_Q_all=0.0, H_Q_controlled=0.0,
                                    H_Q_Wfirst=0.0, H_Q_Wfirst_e=0.0,
                                    H_Q_Wfirst_h=0.0, H_Q_down=0.0,
                                    D_Q=0.0, D_FP=0.0)
                               if fam in ("ptq", "lora") else None)
    if dd is None:
        print(f"  [skip-drift] {name}: no drift ({dk})")
        continue
    pts.append(dict(method=name, family=fam, objective=obj, lr=lr,
                    budget=bud, seed=seed, reused=reused,
                    **{k: d[k] for k in DS}, mean4=d["mean4"], **dd))


def mark(points, xkey):
    for p in points:
        p[f"pareto_optimal_{xkey}"] = not any(
            (q["mean4"] >= p["mean4"] and q[xkey] <= p[xkey]
             and (q["mean4"] > p["mean4"] or q[xkey] < p[xkey]))
            for q in points)


for xk in ("D_Q", "H_Q_all", "H_Q_controlled"):
    mark(pts, xk)

FN = ["method", "family", "objective", "lr", "budget", "seed",
      "reused", *DS, "mean4", "H_Q_all", "H_Q_controlled",
      "H_Q_Wfirst", "H_Q_Wfirst_e", "H_Q_Wfirst_h", "H_Q_down",
      "D_Q", "D_FP", "pareto_optimal_D_Q", "pareto_optimal_H_Q_all",
      "pareto_optimal_H_Q_controlled"]
for fname, xk in (("pareto_frontier_dq.csv", "pareto_optimal_D_Q"),
                  ("pareto_frontier_hq.csv", "pareto_optimal_H_Q_all"),
                  ("pareto_frontier_hqc.csv",
                   "pareto_optimal_H_Q_controlled")):
    with open(os.path.join(OUT, fname), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FN)
        w.writeheader()
        for p in sorted(pts, key=lambda r: -r["mean4"]):
            w.writerow({k: p.get(k, "") for k in FN})
# section 5 CSV (plain points only)
with open(os.path.join(OUT, "plain_lr_frontier.csv"), "w",
          newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FN)
    w.writeheader()
    for p in sorted([p for p in pts if p["family"] == "plain"],
                    key=lambda r: (r["objective"], r["lr"], r["seed"])):
        w.writerow({k: p.get(k, "") for k in FN})
print(f"{len(pts)} points -> {OUT}")
for p in sorted(pts, key=lambda r: -r["mean4"]):
    fl = "".join(("D" if p["pareto_optimal_D_Q"] else "-",
                  "H" if p["pareto_optimal_H_Q_all"] else "-",
                  "C" if p["pareto_optimal_H_Q_controlled"] else "-"))
    print(f"{p['method']:34s} m4={p['mean4']:.4f} DQ={p['D_Q']:.6f} "
          f"Hall={p['H_Q_all']:.5f} [{fl}]")
