#!/usr/bin/env python
"""Merge domain-parallel corpus manifests: _pmg_merge_corpus.py <run_dir> <tag_prefix> <teacher>"""
import json, os, sys

rd, prefix, teacher = sys.argv[1], sys.argv[2], sys.argv[3]
doms = ["wiki", "c4", "sharegpt", "gsm8k", "code"]
shards, n = [], 0
for d in doms:
    mp = os.path.join(rd, "manifests", f"lkcorpus__{prefix}_{d}.json")
    m = json.load(open(mp))
    assert m["teacher"] == teacher and m["mode"] == "greedy", (d, m)
    shards += m["shards"]
    n += m["n_windows"]
out = dict(teacher=teacher, mode="greedy", n_windows=n, shards=shards,
           domains=",".join(doms), T=m["T"], K=m["K"],
           gen_len=m["gen_len"])
op = os.path.join(rd, "manifests", f"lkcorpus__{prefix}_all.json")
json.dump(out, open(op, "w"), indent=1)
print(f"[merge] {n} windows, {len(shards)} shards -> {op}")
