#!/usr/bin/env python
"""Pre-registered representative-seed rule (median mtbench tau, never
best) for the R6 PTQ and QAT-stage2 arms."""
import csv, json, os, sys

rd = sys.argv[1]
out = {}
for arm, pat in (("r6p", "R6P_s{s}"), ("r6q", "R6Q_s{s}")):
    taus = {}
    for s in (0, 1, 2):
        sh = os.path.join(rd, "shards",
                          f"al__{pat.format(s=s)}__int4__mtbench.csv")
        ts = [t for r in csv.DictReader(open(sh))
              for t in json.loads(r["acceptance_list"])]
        taus[s] = sum(ts) / max(len(ts), 1)
    med = sorted(taus, key=lambda s: taus[s])[1]
    out[arm] = dict(seed=med, taus={str(s): round(t, 4)
                                    for s, t in taus.items()},
                    rule="median mtbench tau (pre-registered, never "
                         "best)")
json.dump(out, open(os.path.join(rd, "tables", "median_seeds.json"),
                    "w"), indent=1)
print("[median]", json.dumps(out))
