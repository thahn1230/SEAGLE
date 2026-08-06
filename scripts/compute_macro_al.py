#!/usr/bin/env python
"""Prompt-macro AL from official-evaluator shards.

macro-AL = mean over prompts of (mean of that prompt's per-cycle
sequence-growth deltas). Same underlying records as official micro
tau (cycle-pooled), NEVER conflated with it. Bootstrap CI resamples
prompts (3000 reps, seed 20260723).
"""
import argparse, csv, glob, json, os

import numpy as np


def macro_of(path, reps=3000, seed=20260723):
    per_prompt = []
    for r in csv.DictReader(open(path)):
        ts = json.loads(r["acceptance_list"])
        if ts:
            per_prompt.append(sum(ts) / len(ts))
    a = np.asarray(per_prompt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    boots = a[idx].mean(axis=1)
    return dict(macro_al=round(float(a.mean()), 4),
                ci=[round(float(np.percentile(boots, 2.5)), 4),
                    round(float(np.percentile(boots, 97.5)), 4)],
                n_prompts=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    out = {}
    for p in sorted(glob.glob(os.path.join(args.run_dir, "shards",
                                           "al__*.csv"))):
        key = os.path.basename(p)[4:-4]
        out[key] = macro_of(p, args.reps)
    path = os.path.join(args.run_dir, "tables", "macro_al_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[macro] {len(out)} shards -> {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
