#!/usr/bin/env python3
"""Pooled official micro-tau per FIN tag per dataset + mean4 + family
median seeds. Writes tables/final_taus.json."""
import ast
import csv
import json
import os
import re

QRUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(QRUN, "shards")
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")

taus = {}
for f in sorted(os.listdir(SH)):
    m = re.match(r"al__([A-Za-z0-9._-]+)__int4__"
                 r"(mtbench|gsm8k|sharegpt|humaneval)\.csv$", f)
    if not m:
        continue
    tag, ds = m.groups()
    if tag.endswith("__" + ds):        # FIN tags carry a ds suffix
        tag = tag[:-(len(ds) + 2)]
    tot = cyc = 0
    with open(os.path.join(SH, f)) as fh:
        for row in csv.DictReader(fh):
            al = ast.literal_eval(row["acceptance_list"])
            tot += sum(al)
            cyc += len(al)
    if cyc:
        taus.setdefault(tag, {})[ds] = round(tot / cyc, 4)

table = {}
for tag, d in sorted(taus.items()):
    if all(k in d for k in DS):
        d = dict(d)
        d["mean4"] = round(sum(d[k] for k in DS) / 4, 4)
    table[tag] = d

# family median seeds (by mean4) for 3-seed FIN families
fams = {}
for tag, d in table.items():
    m = re.match(r"(FIN_(?:FD|DP)_B_[a-z]+_beta[0-9.]+)_s(\d)_st\d+$"
                 r"|(CFIN_P2conv_gsr5)_s(\d)$", tag)
    if not m or "mean4" not in d:
        continue
    fam = m.group(1) or m.group(3)
    fams.setdefault(fam, []).append((d["mean4"], tag))
med = {}
for fam, xs in fams.items():
    xs.sort()
    med[fam] = dict(median_tag=xs[len(xs) // 2][1],
                    mean4_by_seed={t: v for v, t in xs})

out = dict(taus=table, median=med)
p = os.path.join(QRUN, "tables", "final_taus.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"wrote {p}")
for tag, d in sorted(table.items()):
    if "mean4" in d:
        print(f"{tag}: mean4={d['mean4']} " +
              " ".join(f"{k}={d[k]}" for k in DS))
for fam, v in sorted(med.items()):
    print(f"MEDIAN {fam}: {v['median_tag']}")
