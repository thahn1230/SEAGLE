#!/usr/bin/env python
"""AA-QAT study corpus merge: identical to _r1r2gs_merge_corpus.py
(deterministic seed-0 stratified shuffle before re-sharding, so the
trainer's last-10% validation tail is domain-balanced and — given
bit-identical windows — IDENTICAL in composition to the R1R2GS split),
parameterized for the a_chain-extended `aaq_t4_greedy_*` domain tags.

Usage: _aaq_merge_corpus.py <run_dir> [in_prefix] [out_tag]
"""
import hashlib, json, os, sys

import torch

rd = sys.argv[1]
in_prefix = sys.argv[2] if len(sys.argv) > 2 else "aaq_t4_greedy"
out_tag = sys.argv[3] if len(sys.argv) > 3 else "aaq_all"
doms = ["wiki", "c4", "sharegpt", "gsm8k", "code"]
windows, meta0 = [], None
for d in doms:
    mp = os.path.join(rd, "manifests", f"lkcorpus__{in_prefix}_{d}.json")
    m = json.load(open(mp))
    assert m["teacher"] == "t4" and m["mode"] == "greedy", (d, m)
    meta0 = m
    for sh in m["shards"]:
        p = os.path.join(rd, "rotations", sh["shard"])
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert sha == sh["sha256"], f"shard hash drift: {p}"
        windows += torch.load(p, map_location="cpu",
                              weights_only=False)["windows"]
n = len(windows)
g = torch.Generator().manual_seed(0)
perm = torch.randperm(n, generator=g).tolist()
windows = [windows[i] for i in perm]
dom_tail = {}
for w in windows[-max(n // 10, 16):]:
    dom_tail[w["domain"]] = dom_tail.get(w["domain"], 0) + 1

shards, shard_size = [], 500
for i0 in range(0, n, shard_size):
    sid = len(shards)
    p = os.path.join(rd, "rotations",
                     f"lkcorpus__{out_tag}__s{sid:03d}.pt")
    chunk = windows[i0:i0 + shard_size]
    torch.save(dict(windows=chunk, meta=dict(
        teacher="t4", mode="greedy", T=meta0["T"], K=meta0["K"],
        n=len(chunk), shuffled_seed=0, a_chain=True)), p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    shards.append(dict(shard=os.path.basename(p), n=len(chunk),
                       sha256=sha))
out = dict(teacher="t4", mode="greedy", n_windows=n, shards=shards,
           domains=",".join(doms), seed="1-5 per-domain",
           stratified_shuffle_seed=0, val_tail_domain_counts=dom_tail,
           a_chain=True,
           T=meta0["T"], K=meta0["K"], gen_len=meta0["gen_len"])
op = os.path.join(rd, "manifests", f"lkcorpus__{out_tag}.json")
json.dump(out, open(op, "w"), indent=1)
print(f"[merge-aaq] {n} windows, {len(shards)} shards, "
      f"val-tail domains {dom_tail} -> {op}")
