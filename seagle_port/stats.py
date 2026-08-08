"""Paired prompt-cluster bootstrap + Holm step-down for DFST shards (§33).

Shard schema: shards/al__<tag>__<target>__<ds>.csv with per-(prompt,turn)
cycle tau lists. Pairing key = (prompt_id, turn); statistic = cycle-pooled
tau difference (sum taus / sum cycles, recomputed per resample — matches
SEAGLE bootstrap contract). 3000 resamples, percentile CI, two-sided p with
add-one smoothing (never report p=0).

Usage:
  python -m seagle_port.stats --run-dir RD --dataset mtbench \
      --pair name=A@fp16:B@w4a4 ... [--n 3000] [--subset-prompts 40]
Then: python -m seagle_port.stats --run-dir RD --holm
"""
import argparse
import glob
import json
import os

import numpy as np


def load_shard(rd, spec, ds, subset=None):
    tag, tgt = spec.split("@")
    p = os.path.join(rd, "shards", f"al__{tag}__{tgt}__{ds}.csv")
    rows = {}
    import csv
    with open(p) as f:
        for r in csv.DictReader(f):
            pid = int(r["prompt_id"])
            if subset is not None and pid >= subset:
                continue
            taus = [int(x) for x in r["taus"].split(";") if x]
            rows[(pid, int(r["turn"]))] = taus
    return rows


def pooled(rows, keys):
    s = c = 0
    for k in keys:
        t = rows[k]
        s += sum(t)
        c += len(t)
    return s / max(c, 1)


def bootstrap_pair(a, b, n=3000, seed=0):
    keys = sorted(set(a) & set(b))
    assert keys, "no paired keys"
    rng = np.random.default_rng(seed)
    base = pooled(b, keys) - pooled(a, keys)
    deltas = np.empty(n)
    karr = np.array(keys, dtype=object)
    for i in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        ks = [keys[j] for j in idx]
        deltas[i] = pooled(b, ks) - pooled(a, ks)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = max(p, 1.0 / n)          # add-one smoothing, never 0
    return dict(delta=base, ci=[float(lo), float(hi)], p=float(p),
                n_pairs=len(keys), n_resamples=n,
                a_tau=pooled(a, keys), b_tau=pooled(b, keys))


def holm(rd):
    out = {}
    # dedupe by comparison name: the same pair may have been written into
    # more than one bootstrap_*.json across re-runs; counting it twice would
    # inflate m and make the Holm thresholds wrong.
    uniq = {}
    for f in sorted(glob.glob(os.path.join(rd, "stats", "bootstrap_*.json"))):
        d = json.load(open(f))
        for name, r in d.items():
            uniq[name] = (r["p"], f)
    dups = sum(len(json.load(open(f)))
               for f in glob.glob(os.path.join(rd, "stats",
                                               "bootstrap_*.json"))) - len(uniq)
    if dups:
        print(f"[holm] deduped {dups} repeated comparison name(s)")
    allp = [(n, p, f) for n, (p, f) in uniq.items()]
    allp.sort(key=lambda x: x[1])
    m = len(allp)
    rejected = 0
    prev_ok = True
    for i, (name, p, f) in enumerate(allp):
        ok = prev_ok and p < 0.05 / (m - i)
        prev_ok = ok
        out[name] = dict(p=p, holm_alpha=0.05 / (m - i), significant=ok,
                         source=os.path.basename(f))
        rejected += ok
    json.dump(out, open(os.path.join(rd, "stats", "holm_adjusted.json"),
                        "w"), indent=1)
    print(f"[holm] {m} comparisons, {rejected} rejected at 0.05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--subset-prompts", type=int, default=None)
    ap.add_argument("--holm", action="store_true")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    rd = args.run_dir
    os.makedirs(os.path.join(rd, "stats"), exist_ok=True)
    if args.holm:
        holm(rd)
        return
    res = {}
    for pr in args.pair:
        name, spec = pr.split("=", 1)
        sa, sb = spec.split(":")
        a = load_shard(rd, sa, args.dataset, args.subset_prompts)
        b = load_shard(rd, sb, args.dataset, args.subset_prompts)
        r = bootstrap_pair(a, b, n=args.n)
        res[f"{args.dataset}:{name}"] = r
        print(f"[boot] {name}: {r['a_tau']:.4f} -> {r['b_tau']:.4f} "
              f"delta={r['delta']:.4f} CI={r['ci']} p={r['p']:.5f}")
    out = os.path.join(rd, "stats",
                       f"bootstrap_{args.out_name or args.dataset}.json")
    if os.path.exists(out):
        old = json.load(open(out))
        old.update(res)
        res = old
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
