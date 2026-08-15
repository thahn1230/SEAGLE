#!/usr/bin/env python3
"""Holm correction across the 4 datasets within each planned family.
Reads stats/bootstrap_pairs_<ds>.json; p floored at 1/reps.
Writes stats/final_stats_holm.json."""
import json
import os

QRUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")
ALPHA = 0.05

pairs = {}
for ds in DS:
    p = os.path.join(QRUN, "stats", f"bootstrap_pairs_{ds}.json")
    for r in json.load(open(p)):
        if r["status"] != "ok":
            continue
        floor = 1.0 / r["reps"]
        r["p_floored"] = max(r["p_two_sided"], floor)
        pairs.setdefault(r["name"], {})[ds] = r

out = {}
for fam, by_ds in sorted(pairs.items()):
    items = sorted(by_ds.items(), key=lambda kv: kv[1]["p_floored"])
    m = len(items)
    holm = {}
    alive = True
    for i, (ds, r) in enumerate(items):
        thr = ALPHA / (m - i)
        sig = alive and (r["p_floored"] <= thr)
        if not sig:
            alive = False
        holm[ds] = dict(delta=r["delta_b_minus_a"],
                        ci=r["ci_delta"], p=r["p_floored"],
                        holm_threshold=round(thr, 5),
                        significant=sig,
                        direction=("+" if r["delta_b_minus_a"] > 0
                                   else "-"))
    wins = sum(1 for v in holm.values()
               if v["significant"] and v["direction"] == "+")
    losses = sum(1 for v in holm.values()
                 if v["significant"] and v["direction"] == "-")
    out[fam] = dict(per_dataset=holm, wins=wins, losses=losses,
                    a=items[0][1]["a"], b_example=items[0][1]["b"])
p = os.path.join(QRUN, "stats", "final_stats_holm.json")
json.dump(out, open(p, "w"), indent=1, sort_keys=True)
print(f"wrote {p}")
for fam, v in sorted(out.items()):
    marks = " ".join(f"{ds}:{'W' if v['per_dataset'][ds]['significant'] and v['per_dataset'][ds]['direction']=='+' else ('L' if v['per_dataset'][ds]['significant'] else 'ns')}"
                     for ds in DS)
    print(f"{fam}: {v['wins']}W/{v['losses']}L  {marks}")
