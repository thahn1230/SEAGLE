#!/usr/bin/env python
"""SR6 auto-finalizer: runs when all eval shards exist. Median-seed table,
bootstrap vs canonical GS baseline, Holm, extended 2x2, summary.csv."""
import csv, glob, json, os, subprocess, sys, statistics

rd = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(rd, "..", ".."))
rd = os.path.relpath(rd)
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]

def tau(tag, ds):
    p = f"{rd}/shards/al__{tag}__int4__{ds}.csv"
    if not os.path.exists(p):
        return None
    tot = n = 0
    for r in csv.DictReader(open(p)):
        acc = [int(x) for x in r["acceptance_list"].split(";") if x != ""] \
            if ";" in r["acceptance_list"] else \
            [int(x) for x in r["acceptance_list"].strip("[] ").split(",")
             if x.strip()]
        tot += sum(a + 1 for a in acc)
        n += len(acc)
    return tot / max(n, 1)

# median seed by mtbench (pre-registered)
mt = {s: tau(f"SR6_s{s}", "mtbench") for s in (0, 1, 2)}
med_seed = sorted(mt, key=lambda s: mt[s])[1]
jm = {s: tau(f"JOINT_s{s}", "mtbench") for s in (0, 1, 2)}
jmed = sorted([s for s in jm if jm[s]], key=lambda s: jm[s])
jmed_seed = jmed[len(jmed) // 2] if len(jmed) == 3 else None

rows = {"GS_baseline": {"mtbench": 2.9955, "gsm8k": 3.4490,
                        "sharegpt": 3.1092, "humaneval": 3.7490}}
rows[f"GS_R6_standalone_s{med_seed}"] = {d: tau(f"SR6_s{med_seed}", d)
                                          for d in DS}
rows["TRANSPLANT_ctrl"] = {d: tau("TRANSPLANT", d) for d in DS}
if jmed_seed is not None:
    rows[f"GS_R5R6_joint_s{jmed_seed}"] = {d: tau(f"JOINT_s{jmed_seed}", d)
                                            for d in DS}
rows["GS_R5_ref"] = {"mtbench": 3.1645, "gsm8k": 3.6875,
                     "sharegpt": 3.2351, "humaneval": 3.8282}
rows["GS_R5R6_seq_ref"] = {"mtbench": 3.1698, "gsm8k": 3.6379,
                           "sharegpt": 3.2853, "humaneval": 3.8244}
with open(f"{rd}/tables/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method"] + DS + ["mean4"])
    for k, v in rows.items():
        vals = [v.get(d) for d in DS]
        mean = sum(vals) / 4 if all(x is not None for x in vals) else None
        w.writerow([k] + [round(x, 4) if x else "" for x in vals] +
                   [round(mean, 4) if mean else ""])
        print(k, [round(x, 4) if x else None for x in vals],
              round(mean, 4) if mean else "incomplete")

# paired bootstrap vs canonical baseline (B3_T4 shards already in rd)
pairs = [f"r6_vs_gs_{d}=B3_T4@int4:SR6_s{med_seed}@int4" for d in DS]
for d in DS:
    subprocess.run(["python", "scripts/bootstrap_eagle_tau.py",
                    "--run-dir", rd, "--dataset", d, "--pair",
                    f"r6_vs_gs_{d}=B3_T4@int4:SR6_s{med_seed}@int4"],
                   check=False)
subprocess.run(["python", "scripts/holm_adjust_bootstrap.py",
                "--run-dir", rd], check=False)
print("[finalize] DONE — tables/summary.csv + stats/")
