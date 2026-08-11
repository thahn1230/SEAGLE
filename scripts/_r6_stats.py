#!/usr/bin/env python
"""R6 study stats battery: paired prompt-cluster bootstrap (3000) +
Holm within pre-registered families.

F1 (PTQ):  R5-only (B9_T4) vs R5+R6 PTQ across 4 datasets
F2 (QAT):  R5+QAT (QF_rdg_T4_s2) vs R5+R6 QAT across 4 datasets
Exploratory (mtbench only, reported separately): random-R6 controls.
"""
import json, os, subprocess, sys

rd = sys.argv[1]
med = json.load(open(os.path.join(rd, "tables", "median_seeds.json")))
P = f"R6P_s{med['r6p']['seed']}@int4"
Q = f"R6Q_s{med['r6q']['seed']}@int4"
PAIRS = [
    f"r6_gain_ptq=B9_T4@int4:{P}",
    f"r6_gain_qat=QF_rdg_T4_s2@int4:{Q}",
    f"qat_gain_over_r6ptq={P}:{Q}",
]
RAND = [f"rand_r6_s{s}=B9_T4@int4:R6RAND_s{s}@int4"
        for s in (101, 102, 103)]
for ds in ("mtbench", "gsm8k", "sharegpt", "humaneval"):
    cmd = ["python", "scripts/bootstrap_eagle_tau.py", "--run-dir", rd,
           "--dataset", ds, "--reps", "3000"]
    for p in PAIRS:
        cmd += ["--pair", p]
    if ds == "mtbench":
        for p in RAND:
            cmd += ["--pair", p]
    subprocess.run(cmd, check=True)
subprocess.run(["python", "scripts/holm_adjust_bootstrap.py",
                "--run-dir", rd, "--reps", "3000"], check=True)
print("[r6-stats] done")
