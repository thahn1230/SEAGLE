#!/usr/bin/env python
"""Merge the 5 domain-parallel rd_t4 corpus manifests into one."""
import json, os, sys

rd = sys.argv[1]
doms = ["wiki", "c4", "sharegpt", "gsm8k", "code"]
shards, n = [], 0
for d in doms:
    mp = os.path.join(rd, "manifests", f"lkcorpus__rd_t4_greedy_{d}.json")
    m = json.load(open(mp))
    assert m["teacher"] == "t4" and m["mode"] == "greedy", (d, m)
    shards += m["shards"]
    n += m["n_windows"]
out = dict(teacher="t4", mode="greedy", n_windows=n, shards=shards,
           domains=",".join(doms), seed="1-5 per-domain",
           T=m["T"], K=m["K"], gen_len=m["gen_len"])
op = os.path.join(rd, "manifests", "lkcorpus__rd_t4_all.json")
json.dump(out, open(op, "w"), indent=1)
print(f"[merge] {n} windows, {len(shards)} shards -> {op}")
