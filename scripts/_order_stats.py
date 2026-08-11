#!/usr/bin/env python
"""Ordering-experiment stats battery (paired prompt-cluster bootstrap
3000 + Holm within dataset families) + aggregate mean4 bootstrap for
the primary ordering contrast.

Pairs:
  order_primary : R5->QAT (QF_rdg_T4_s2)  vs QAT->R5 (ARMD_T4)
  vs_r5only     : R5 (B9_T4)              vs QAT->R5
  vs_qatonly    : QAT (QFR_ep3g_T4)       vs QAT->R5
  reopt_vs_armD : QAT->R5 (ARMD_T4)       vs R5->QAT->R5reopt (P3D_T4)
  reopt_vs_b    : R5->QAT (QF_rdg_T4_s2)  vs P3D_T4
  fresh_vs_warm : P3C_T4                  vs P3D_T4 (same weights,
                  fresh vs warm-start R5 — path dependence)
"""
import json, os, subprocess, sys

rd = sys.argv[1]
PAIRS = [
    "order_primary=QF_rdg_T4_s2@int4:ARMD_T4@int4",
    "vs_r5only=B9_T4@int4:ARMD_T4@int4",
    "vs_qatonly=QFR_ep3g_T4@int4:ARMD_T4@int4",
    "reopt_vs_armD=ARMD_T4@int4:P3D_T4@int4",
    "reopt_vs_b=QF_rdg_T4_s2@int4:P3D_T4@int4",
    "fresh_vs_warm=P3C_T4@int4:P3D_T4@int4",
]
for ds in ("mtbench", "gsm8k", "sharegpt", "humaneval"):
    cmd = ["python", "scripts/bootstrap_eagle_tau.py", "--run-dir", rd,
           "--dataset", ds, "--reps", "3000"]
    for p in PAIRS:
        cmd += ["--pair", p]
    subprocess.run(cmd, check=True)
subprocess.run(["python", "scripts/holm_adjust_bootstrap.py",
                "--run-dir", rd, "--reps", "3000"], check=True)
for a, b in (("QF_rdg_T4_s2", "ARMD_T4"), ("B9_T4", "ARMD_T4"),
             ("ARMD_T4", "P3D_T4")):
    subprocess.run(["python", "scripts/_causal_aggregate_boot.py",
                    rd, a, b], check=True)
open(os.path.join(rd, "stats", "holm_adjusted.json.order"),
     "w").write("done\n")
print("[order-stats] done")
