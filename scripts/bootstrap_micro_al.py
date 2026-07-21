#!/usr/bin/env python
"""Paired prompt-cluster bootstrap for LK shards (study spec section 3).

Reuses the validated machinery from aggregate_micro_al.py (>=2000
replicates, prompts resampled as clusters, paired mean difference + 95%
CI). Applies it to lktree__/lkchain__ shards: every candidate is paired
against the shared baseline with the same mode/target/dataset suffix.

Writes tables/lk_micro_al_summary.csv and tables/lk_paired_deltas.csv.
"""
import argparse, csv, glob, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_micro_al import read_shard, pooled_mal, boot_ci, paired_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline-tag", default="SHARED_RT")
    args = ap.parse_args()
    rng = np.random.default_rng(20260721)
    sh = os.path.join(args.run_dir, "shards")
    clusters, rows = {}, []
    for path in sorted(glob.glob(os.path.join(sh, "lk*.csv"))):
        base = os.path.basename(path)[:-4]
        cl = [(pid, taus) for pid, taus, _ in read_shard(path)]
        if not cl:
            continue
        clusters[base] = cl
        mal = pooled_mal(cl)
        lo, hi = boot_ci(cl, rng)
        rows.append(dict(shard=base, micro_al=round(mal, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         n_prompts=len(cl),
                         n_cycles=sum(len(t) for _, t in cl)))
    tdir = os.path.join(args.run_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "lk_micro_al_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[boot] {len(rows)} shards summarized")

    # pair candidates vs the shared baseline with the same suffix
    prow = []
    for b in clusters:
        if f"__{args.baseline_tag}__" not in b:
            continue
        prefix, suffix = b.split(f"__{args.baseline_tag}__")
        for c in clusters:
            if c == b or not (c.startswith(prefix)
                              and c.endswith(suffix)):
                continue
            d, lo, hi, p, n = paired_boot(clusters[b], clusters[c], rng)
            prow.append(dict(baseline=b, candidate=c,
                             delta=round(d, 4), ci_lo=round(lo, 4),
                             ci_hi=round(hi, 4), p_boot=round(p, 5),
                             n_common=n))
    if prow:
        with open(os.path.join(tdir, "lk_paired_deltas.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prow[0]))
            w.writeheader(); w.writerows(prow)
    print(f"[boot] {len(prow)} paired deltas -> lk_paired_deltas.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
