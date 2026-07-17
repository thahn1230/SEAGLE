#!/usr/bin/env python
"""Assemble every results table in the run dir into one markdown digest
(tables/RESULTS_DIGEST.md) for report writing and review."""
import csv, glob, json, os, sys

RD = sys.argv[1]
out = [f"# TLDR-KV4 results digest\nrun: {os.path.basename(RD)}\n"]
for p in sorted(glob.glob(os.path.join(RD, "tables", "*.csv"))):
    name = os.path.basename(p)
    with open(p) as f:
        rows = list(csv.reader(f))
    if not rows:
        continue
    out.append(f"\n## {name} ({len(rows)-1} rows)\n")
    out.append("| " + " | ".join(rows[0]) + " |")
    out.append("|" + "---|" * len(rows[0]))
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
for p in sorted(glob.glob(os.path.join(RD, "shards", "kv4mat__*.json"))):
    d = json.load(open(p))
    out.append(f"\n## {os.path.basename(p)}\n```json\n"
               f"{json.dumps(d, indent=1)}\n```")
dst = os.path.join(RD, "tables", "RESULTS_DIGEST.md")
with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"[digest] -> {dst} ({sum(len(l) for l in out)} chars)")
