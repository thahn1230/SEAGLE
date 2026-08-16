#!/usr/bin/env python3
"""Closure tau table: pooled micro-tau for every tag in NR/shards
(CLFIN_*, CF_*, QSEL_*) + mean4. Writes tables/closure_taus.json."""
import ast
import csv
import json
import os
import re

NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(NR, "shards")
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")

taus = {}
for f in sorted(os.listdir(SH)):
    m = re.match(r"al__([A-Za-z0-9._-]+)__int4__"
                 r"(mtbench|gsm8k|sharegpt|humaneval|c4)(__calib)?\.csv$",
                 f)
    if not m:
        continue
    tag, ds = m.group(1), m.group(2)
    tot = cyc = 0
    with open(os.path.join(SH, f)) as fh:
        for row in csv.DictReader(fh):
            al = ast.literal_eval(row["acceptance_list"])
            tot += sum(al)
            cyc += len(al)
    if cyc:
        taus.setdefault(tag, {})[ds] = round(tot / cyc, 4)

for tag, d in taus.items():
    if all(k in d for k in DS):
        d["mean4"] = round(sum(d[k] for k in DS) / 4, 4)

p = os.path.join(NR, "tables", "closure_taus.json")
json.dump(taus, open(p, "w"), indent=1, sort_keys=True)
print(f"wrote {p}")
for tag, d in sorted(taus.items()):
    if "mean4" in d:
        print(f"{tag}: mean4={d['mean4']} " +
              " ".join(f"{k}={d[k]}" for k in DS))
    elif "c4" in d:
        print(f"{tag}: calib={d['c4']}")
