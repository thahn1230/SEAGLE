#!/usr/bin/env python
"""GS+R5 stats battery: 5 pre-registered paired-bootstrap comparisons
per dataset (reps 3000, paired prompt-cluster) + Holm per dataset.
Baseline tags (B3_T4, B7_T4, QF_ep3g_T4_s2, QF_rd_T4_s0) are canonical
PMG shards copied into this run dir. delta = tau_B - tau_A."""
import json, os, subprocess, sys

rd = sys.argv[1]
med = json.load(open(os.path.join(rd, "tables",
                                  "median_seeds.json")))["rdg_T4"]["seed"]
QN = f"QF_rdg_T4_s{med}@int4"
PAIRS = [
    # canonical-PMG-baseline comparisons (historical shards)
    f"r5_gain_ptq=B3_T4@int4:B9_T4@int4",
    f"gs_v_ls_r5ptq=B7_T4@int4:B9_T4@int4",
    f"r5_gain_qat=QF_ep3g_T4_s2@int4:{QN}",
    f"qat_gain_gsr5=B9_T4@int4:{QN}",
    f"gs_v_ls_r5qat=QF_rd_T4_s0@int4:{QN}",
    # §19 fresh-paired rotation comparison (same-session raw records)
    f"rt_vs_r5_gs=B3R_T4@int4:B9_T4@int4",
    f"rt_vs_r5_gs_qat=QFR_ep3g_T4@int4:{QN}",
    # §14 diagnostic: does moving LS->GS change the optimal R5?
    f"reused_vs_gsspecific_r5=B9_T4@int4:B9G_T4@int4",
    f"rt_vs_gsspecific_r5=B3R_T4@int4:B9G_T4@int4",
    f"gsspecific_vs_lsr5=B7R_T4@int4:B9G_T4@int4",
]
for ds in ("mtbench", "gsm8k", "sharegpt", "humaneval"):
    cmd = ["python", "scripts/bootstrap_eagle_tau.py", "--run-dir", rd,
           "--dataset", ds, "--reps", "3000"]
    for p in PAIRS:
        cmd += ["--pair", p]
    subprocess.run(cmd, check=True)
subprocess.run(["python", "scripts/holm_adjust_bootstrap.py",
                "--run-dir", rd, "--reps", "3000"], check=True)
print("[gsr5-stats] done")
