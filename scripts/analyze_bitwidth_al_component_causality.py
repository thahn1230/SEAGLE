#!/usr/bin/env python
"""Statistics for the bitwidth-AL component causality study.

Inputs: a matrix run dir containing shards/al__{stock,t8,t4}.csv (and optionally
component-ablation shards al__comp_*.csv with the same row schema).

Outputs (into <run_dir>/analysis/):
  matrix_al.csv            9-cell AL means + per-cell prompt-level SD
  paired_contrasts.csv     paired cluster-bootstrap contrasts (10k, 95% CI)
                           - target effect at fixed draft column (vs T16 row)
                           - draft effect at fixed target row (vs D16 column)
                           - every cell vs T16_D16
                           with practical-equivalence verdicts (eps_abs=0.05,
                           eps_rel=1% of the baseline mean)
  interaction.csv          non-separability: additive prediction
                           AL(Ti,Dj) ~ AL(T16,D16) + rowEff + colEff vs observed
  depth_hist.csv           accepted-depth histograms per cell
  summary.json             headline numbers

CPU-only; no GPU or CVD assertions.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

EPS_ABS = 0.05
EPS_REL = 0.01
N_BOOT = 10_000
SEED = 0

ROWS_T = ["T16", "T8", "T4"]
COLS_D = ["D16", "D8", "D4"]


def load_shards(run_dir):
    per = defaultdict(dict)          # cell -> prompt_id -> per-prompt record
    sh = os.path.join(run_dir, "shards")
    for fn in sorted(os.listdir(sh)):
        if not (fn.startswith("al__") and fn.endswith(".csv")):
            continue
        with open(os.path.join(sh, fn)) as f:
            for r in csv.DictReader(f):
                per[r["cell"]][r["prompt_id"]] = dict(
                    al=float(r["mean_acceptance"]),
                    depths=json.loads(r["accepted_depths"]),
                    n_cycles=int(r["n_cycles"]),
                    exact_match=r["exact_match"] == "True")
    return per


def paired_boot(a, b, rng):
    """a, b: per-prompt AL arrays (same prompt order). Returns mean diff,
    95% CI over prompt-cluster resampling."""
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), \
        float(np.quantile(boots, 0.975))


def verdict(diff, lo, hi, base_mean):
    eps = max(EPS_ABS, EPS_REL * abs(base_mean))
    if lo > eps or hi < -eps:
        return "different"
    if -eps <= lo and hi <= eps:
        return "practically_equivalent"
    return "inconclusive"


def contrast(per, cell_a, cell_b, label, rng, out):
    if cell_a not in per or cell_b not in per:
        return
    common = sorted(set(per[cell_a]) & set(per[cell_b]))
    if not common:
        return
    a = np.array([per[cell_a][p]["al"] for p in common])
    b = np.array([per[cell_b][p]["al"] for p in common])
    diff, lo, hi = paired_boot(a, b, rng)
    out.append(dict(contrast=label, cell=cell_a, baseline=cell_b,
                    n_prompts=len(common),
                    al_cell=round(float(a.mean()), 4),
                    al_base=round(float(b.mean()), 4),
                    diff=round(diff, 4), ci_lo=round(lo, 4),
                    ci_hi=round(hi, 4),
                    verdict=verdict(diff, lo, hi, float(b.mean()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    per = load_shards(rd)
    an = os.path.join(rd, "analysis")
    os.makedirs(an, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- 9-cell matrix ---
    mat_rows, mat = [], {}
    for cell in sorted(per):
        als = np.array([v["al"] for v in per[cell].values()])
        mat[cell] = float(als.mean())
        mat_rows.append(dict(cell=cell, n_prompts=len(als),
                             al_mean=round(float(als.mean()), 4),
                             al_sd=round(float(als.std(ddof=1)), 4)
                             if len(als) > 1 else 0.0,
                             al_min=round(float(als.min()), 4),
                             al_max=round(float(als.max()), 4)))
    _write(os.path.join(an, "matrix_al.csv"), mat_rows)

    # --- paired contrasts ---
    cons = []
    for d in COLS_D:                                  # target effect per column
        for t in ("T8", "T4"):
            contrast(per, f"{t}_{d}", f"T16_{d}",
                     f"target_effect@{d}", rng, cons)
    for t in ROWS_T:                                  # draft effect per row
        for d in ("D8", "D4"):
            contrast(per, f"{t}_{d}", f"{t}_D16",
                     f"draft_effect@{t}", rng, cons)
    for cell in sorted(per):                          # vs stock corner
        if cell != "T16_D16":
            contrast(per, cell, "T16_D16", "vs_stock", rng, cons)
    _write(os.path.join(an, "paired_contrasts.csv"), cons)

    # --- interaction / non-separability ---
    inter = []
    if all(f"{t}_{d}" in mat for t in ROWS_T for d in COLS_D):
        base = mat["T16_D16"]
        for t in ROWS_T:
            for d in COLS_D:
                pred = base + (mat[f"{t}_D16"] - base) + (mat[f"T16_{d}"] - base)
                obs = mat[f"{t}_{d}"]
                inter.append(dict(cell=f"{t}_{d}",
                                  observed=round(obs, 4),
                                  additive_pred=round(pred, 4),
                                  interaction=round(obs - pred, 4)))
    _write(os.path.join(an, "interaction.csv"), inter)

    # --- depth histograms ---
    dh = []
    for cell in sorted(per):
        cnt = defaultdict(int)
        for v in per[cell].values():
            for d in v["depths"]:
                cnt[d] += 1
        tot = sum(cnt.values()) or 1
        for d in sorted(cnt):
            dh.append(dict(cell=cell, accepted_depth=d, count=cnt[d],
                           frac=round(cnt[d] / tot, 4)))
    _write(os.path.join(an, "depth_hist.csv"), dh)

    summ = dict(run_dir=rd, cells={k: round(v, 4) for k, v in mat.items()},
                n_boot=N_BOOT, eps_abs=EPS_ABS, eps_rel=EPS_REL,
                n_contrasts=len(cons),
                n_different=sum(c["verdict"] == "different" for c in cons),
                max_abs_interaction=max((abs(i["interaction"]) for i in inter),
                                        default=None))
    with open(os.path.join(an, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return 0


def _write(path, rows):
    if not rows:
        open(path, "w").close()
        return
    fields = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
