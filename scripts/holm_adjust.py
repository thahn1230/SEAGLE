#!/usr/bin/env python
"""Holm-Bonferroni adjustment over the rotation-panel paired deltas.

Groups tables/paired_deltas.csv rows into families by deployed target
(t8/t4/t4kv4, parsed from the shard names) and adjusts p_boot within
each family. Bootstrap p-values of 0 are floored at 1/REPS (2000).
Writes tables/paired_deltas_holm.csv.
"""
import csv, os, sys

RD = sys.argv[1]
src = os.path.join(RD, "tables", "paired_deltas.csv")
rows = list(csv.DictReader(open(src)))
FLOOR = 1.0 / 2000


def family(r):
    for tgt in ("t4kv4", "t8", "t4"):
        if f"__{tgt}__" in r["a"]:
            return tgt
    return "other"


fams = {}
for r in rows:
    fams.setdefault(family(r), []).append(r)
for fam, rs in fams.items():
    rs.sort(key=lambda r: max(float(r["p_boot"]), FLOOR))
    m = len(rs)
    prev = 0.0
    for i, r in enumerate(rs):
        p = max(float(r["p_boot"]), FLOOR)
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        r["family"] = fam
        r["p_holm"] = round(adj, 5)
        r["significant_0.05"] = str(adj < 0.05)
out = os.path.join(RD, "tables", "paired_deltas_holm.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
sig = sum(r["significant_0.05"] == "True" for r in rows)
print(f"[holm] {len(rows)} comparisons, {sig} significant at 0.05 "
      f"-> {out}")
