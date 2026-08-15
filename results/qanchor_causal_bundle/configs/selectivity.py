#!/usr/bin/env python3
"""Per-weight selectivity analysis (prereg): Jaccard overlap of
flip sets (vs anchor c0) across hard budgets, objectives, soft-cell
models, and the aggressive plain-QAT configuration; per-site shares.
Reads ckpts/codes__*.pt (+ bgrid aggressive codes). CPU-only.
Writes tables/selectivity.json."""
import json
import os
import sys

import torch

QRUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(QRUN), "..", "src"))
CKD = os.path.join(QRUN, "ckpts")
SITES = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")

anchor = torch.load(os.path.join(CKD, "anchor_gsr5_ptq.pt"),
                    map_location="cpu", weights_only=True)
c0 = {s: anchor[s]["c0"] for s in SITES}

MODELS = {}
for f in sorted(os.listdir(CKD)):
    if f.startswith("codes__") and f.endswith(".pt"):
        MODELS[f[len("codes__"):-3]] = os.path.join(CKD, f)
# aggressive plain-QAT codes = B-grid full-retention config
bg = os.path.join(CKD, "bgrid__b1_global_f0.pt")
if os.path.exists(bg):
    MODELS["aggressive_P2conv_s0"] = bg


def flip_mask(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(d, dict) and "codes" in d:
        d = d["codes"]
    return {s: (d[s] != c0[s]) for s in SITES}


names = sorted(MODELS)
masks = {n: flip_mask(MODELS[n]) for n in names}
n_total = sum(c0[s].numel() for s in SITES)

per_site = {}
flip_counts = {}
for n in names:
    m = masks[n]
    tot = int(sum(int(m[s].sum()) for s in SITES))
    flip_counts[n] = dict(
        n_flips=tot, H_Q=round(tot / n_total, 6),
        site_share={s: round(int(m[s].sum()) / max(tot, 1), 4)
                    for s in SITES},
        site_rate={s: round(float(m[s].float().mean()), 6)
                   for s in SITES})

jac = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        inter = un = 0
        for s in SITES:
            ma, mb = masks[a][s], masks[b][s]
            inter += int((ma & mb).sum())
            un += int((ma | mb).sum())
        jac[f"{a}__vs__{b}"] = round(inter / max(un, 1), 4)

out = dict(n_total=n_total, flip_counts=flip_counts, jaccard=jac)
p = os.path.join(QRUN, "tables", "selectivity.json")
json.dump(out, open(p, "w"), indent=1, sort_keys=True)
print(f"wrote {p}")
for n in names:
    fc = flip_counts[n]
    print(f"{n}: H_Q={fc['H_Q']} down_share={fc['site_share']['down']}")
top = sorted(jac.items(), key=lambda kv: -kv[1])[:12]
for k, v in top:
    print(f"J {k}: {v}")
