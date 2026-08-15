#!/usr/bin/env python3
"""Cadence selection from QSEL calib shards (c4:20, offset-500 pool).

Pooled official micro-tau per tag = sum(accepted+1)/n_cycles over all
cycles of all prompts. Groups tags by base (strip _stNNNN), reports the
argmax cadence per base and per-family (mean over seeds) choices.
Writes tables/qsel_selection.json.
"""
import ast
import csv
import json
import os
import re
import sys

QRUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(QRUN, "shards")

taus = {}
for f in sorted(os.listdir(SH)):
    m = re.match(r"al__(QSEL_[A-Za-z0-9._-]+)__int4__c4__calib\.csv$", f)
    if not m:
        continue
    tag = m.group(1)
    tot = cyc = 0
    with open(os.path.join(SH, f)) as fh:
        for row in csv.DictReader(fh):
            al = ast.literal_eval(row["acceptance_list"])
            tot += sum(al)
            cyc += len(al)
    if cyc:
        taus[tag] = tot / cyc

by_base = {}
for tag, t in taus.items():
    m = re.match(r"(QSEL_)(.+)_st(\d+)$", tag)
    if not m:
        continue
    base, st = m.group(2), int(m.group(3))
    by_base.setdefault(base, {})[st] = round(t, 4)

sel = {}
for base, d in sorted(by_base.items()):
    best = max(d, key=lambda s: d[s])
    sel[base] = dict(steps=d, best_step=best, best_tau=d[best])

# family view: FD_B_{obj}_beta{b} mean over seeds per cadence
fam = {}
for base, d in by_base.items():
    m = re.match(r"(FD_B_[a-z]+_beta[0-9.]+)_s(\d)$", base)
    if not m:
        continue
    fam.setdefault(m.group(1), {}).setdefault("seeds", {})[m.group(2)] = d
for f, v in fam.items():
    steps = sorted({s for d in v["seeds"].values() for s in d})
    mean = {s: round(sum(d[s] for d in v["seeds"].values()
                         if s in d) / sum(1 for d in v["seeds"].values()
                                          if s in d), 4)
            for s in steps}
    v["mean_by_step"] = mean
    v["best_step"] = max(mean, key=lambda s: mean[s])
    v["best_mean_tau"] = mean[v["best_step"]]

out = dict(per_run=sel, family=fam,
           n_tags=len(taus),
           note="pooled micro-tau on c4:20 calib; PTQ calib ref 3.2794")
p = os.path.join(QRUN, "tables", "qsel_selection.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"wrote {p} ({len(taus)} tags)")
for f, v in sorted(fam.items()):
    print(f"{f}: best st{v['best_step']} mean {v['best_mean_tau']}")
for base, v in sorted(sel.items()):
    if base.startswith("PC_"):
        print(f"{base}: best st{v['best_step']} tau {v['best_tau']}")
