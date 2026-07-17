#!/usr/bin/env python
"""Aggregate all TLDR shard CSVs into per-(family,config,dataset) micro-AL
tables with paired prompt-cluster bootstrap CIs.

Families are inferred from shard filename prefixes:
  panel__<ds>__<cfg>.csv          -> family=panel
  kv4mat__<t>__<d>.csv            -> family=kv4mat (dataset=mtbench)
  ctx__<cfg>.csv                  -> family=ctx (length column inside)
  drot__<tag>__<target>__<ds>.csv -> family=drot

micro-AL = sum(tau) / n_cycles pooled over all cycles of all prompts.
Bootstrap resamples PROMPTS (clusters) with replacement, >=2000 reps.
Paired deltas: same resample indices applied to both configs (shared
prompt manifests guarantee row alignment by prompt_id).

Writes tables/micro_al_summary.csv and tables/paired_deltas.csv.
"""
import argparse, csv, glob, json, os, sys

import numpy as np

REPS = 2000


def read_shard(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            taus = json.loads(r["acceptance_list"])
            rows.append((r.get("prompt_id") or r.get("row_id") or "?",
                         taus, r))
    return rows


def pooled_mal(clusters):
    taus = [t for _, ts in clusters for t in ts]
    return sum(taus) / max(len(taus), 1)


def boot_ci(clusters, rng, reps=REPS):
    n = len(clusters)
    stats = np.empty(reps)
    idx = rng.integers(0, n, size=(reps, n))
    for i in range(reps):
        sel = [clusters[j] for j in idx[i]]
        stats[i] = pooled_mal(sel)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_boot(ca, cb, rng, reps=REPS):
    """clusters aligned by prompt_id; returns (delta, lo, hi, p_two)."""
    ka = {c[0]: c[1] for c in ca}
    kb = {c[0]: c[1] for c in cb}
    common = sorted(set(ka) & set(kb))
    A = [(k, ka[k]) for k in common]
    B = [(k, kb[k]) for k in common]
    d0 = pooled_mal(B) - pooled_mal(A)
    n = len(common)
    idx = rng.integers(0, n, size=(reps, n))
    ds = np.empty(reps)
    for i in range(reps):
        sa = [A[j] for j in idx[i]]
        sb = [B[j] for j in idx[i]]
        ds[i] = pooled_mal(sb) - pooled_mal(sa)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, float(lo), float(hi), float(min(1.0, p)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pairs", default="",
                    help="semicolon list a_glob|b_glob paired comparisons")
    args = ap.parse_args()
    rng = np.random.default_rng(20260717)
    sh = os.path.join(args.run_dir, "shards")
    out_rows = []
    shard_clusters = {}
    for path in sorted(glob.glob(os.path.join(sh, "*.csv"))):
        base = os.path.basename(path)[:-4]
        if base.startswith("ctx__"):        # per-length rows, no taus
            continue
        clusters = [(pid, taus) for pid, taus, _ in read_shard(path)]
        if not clusters:
            continue
        shard_clusters[base] = clusters
        mal = pooled_mal(clusters)
        lo, hi = boot_ci(clusters, rng)
        ncy = sum(len(t) for _, t in clusters)
        out_rows.append(dict(shard=base, micro_al=round(mal, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             n_prompts=len(clusters), n_cycles=ncy))
    tdir = os.path.join(args.run_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "micro_al_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[agg] {len(out_rows)} shards -> tables/micro_al_summary.csv")

    if args.pairs:
        prow = []
        for pair in args.pairs.split(";"):
            a_g, b_g = pair.split("|")
            a_paths = sorted(glob.glob(os.path.join(sh, a_g)))
            b_paths = sorted(glob.glob(os.path.join(sh, b_g)))
            one_to_one = len(a_paths) == 1 and len(b_paths) == 1
            for a_path in a_paths:
                a = os.path.basename(a_path)[:-4]
                suffix = a.split("__")[-1]
                for b_path in b_paths:
                    b = os.path.basename(b_path)[:-4]
                    if not one_to_one and not b.endswith("__" + suffix):
                        continue
                    d, lo, hi, p, n = paired_boot(
                        shard_clusters[a], shard_clusters[b], rng)
                    prow.append(dict(a=a, b=b, delta=round(d, 4),
                                     ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                     p_boot=round(p, 5), n_common=n))
        if prow:
            with open(os.path.join(tdir, "paired_deltas.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(prow[0]))
                w.writeheader()
                w.writerows(prow)
            print(f"[agg] {len(prow)} paired deltas -> "
                  "tables/paired_deltas.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
