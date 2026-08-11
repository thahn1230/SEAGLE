#!/usr/bin/env python
"""Aggregate 4-dataset-mean bootstrap for the R5-PTQ vs R5-QAT contrast.

Resampling procedure (documented, consistent with the reported metric):
for each of 3000 resamples, WITHIN each dataset independently resample
prompts (clusters) with replacement, recompute that dataset's
cycle-pooled micro-tau for both arms from the resampled clusters, then
take the arithmetic mean of the four dataset-level taus per arm; the
statistic is mean4(B) - mean4(A). Prompt pairing is preserved (same
resampled prompt set applied to both arms).

Usage: _causal_aggregate_boot.py <run_dir> <tagA> <tagB> [reps] [seed]
Writes stats/aggregate_boot_<tagA>_vs_<tagB>.json
"""
import csv, json, os, sys
import numpy as np

rd, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260811
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]


def load(tag, ds):
    p = os.path.join(rd, "shards", f"al__{tag}__int4__{ds}.csv")
    out = {}
    for r in csv.DictReader(open(p)):
        ts = json.loads(r["acceptance_list"])
        out[r["prompt_id"]] = (sum(ts), len(ts))
    return out


data = {}
for ds in DS:
    a, b = load(A, ds), load(B, ds)
    common = sorted(set(a) & set(b))
    data[ds] = (np.array([a[k] for k in common], dtype=float),
                np.array([b[k] for k in common], dtype=float))

rng = np.random.default_rng(seed)


def mean4(idx_by_ds, which):
    vals = []
    for ds in DS:
        arr = data[ds][which][idx_by_ds[ds]]
        vals.append(arr[:, 0].sum() / arr[:, 1].sum())
    return float(np.mean(vals))


point_a = mean4({ds: np.arange(len(data[ds][0])) for ds in DS}, 0)
point_b = mean4({ds: np.arange(len(data[ds][0])) for ds in DS}, 1)
deltas = np.empty(reps)
for i in range(reps):
    idx = {ds: rng.integers(0, len(data[ds][0]), len(data[ds][0]))
           for ds in DS}
    deltas[i] = mean4(idx, 1) - mean4(idx, 0)
lo, hi = np.percentile(deltas, [2.5, 97.5])
d0 = point_b - point_a
p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
p = max(p, 1.0 / reps)
out = dict(tagA=A, tagB=B, reps=reps, seed=seed,
           mean4_A=round(point_a, 4), mean4_B=round(point_b, 4),
           delta_mean4=round(d0, 4),
           ci=[round(float(lo), 4), round(float(hi), 4)],
           p_two_sided=("< 1/%d" % reps if p <= 1.0 / reps
                        else round(float(p), 4)),
           procedure="within-dataset paired prompt-cluster resampling; "
                     "arithmetic mean of 4 dataset-level pooled "
                     "micro-taus")
os.makedirs(os.path.join(rd, "stats"), exist_ok=True)
json.dump(out, open(os.path.join(
    rd, "stats", f"aggregate_boot_{A}_vs_{B}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
