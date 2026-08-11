#!/usr/bin/env python
"""Pre-registered representative-seed rule (PMG): the seed whose QF
mtbench-80 pooled micro tau is the MEDIAN of the three (never the best).
Writes tables/median_seeds.json for arm rdg_T4."""
import csv, json, os, sys

rd = sys.argv[1]
taus = {}
for s in (0, 1, 2):
    sh = os.path.join(rd, "shards",
                      f"al__QF_rdg_T4_s{s}__int4__mtbench.csv")
    ts = [t for r in csv.DictReader(open(sh))
          for t in json.loads(r["acceptance_list"])]
    taus[s] = sum(ts) / max(len(ts), 1)
med = sorted(taus, key=lambda s: taus[s])[1]
out = {"rdg_T4": dict(seed=med,
                      taus={str(s): round(t, 4)
                            for s, t in taus.items()},
                      rule="median QF mtbench tau (pre-registered, "
                           "never best)")}
json.dump(out, open(os.path.join(rd, "tables", "median_seeds.json"),
                    "w"), indent=1)
print(f"[median] rdg_T4 -> seed {med} ({taus})")
