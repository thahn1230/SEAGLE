#!/usr/bin/env python
"""Statistics for the component precision ablations (Phase 9).

Input: run dir with shards/comp__{draft,branch,thead,tembed,tbody}.csv
(columns: config, group, prompt_id, mean_acceptance, acceptance_list,
exact_match).

Baselines for paired contrasts (same prompts):
  draft/branch groups  -> stock identity-mode fp16 draft ("draft_full__" with
                          no quant is not run; the fp16 anchor is the T16_D16
                          matrix value; per-config CIs are computed against
                          the group-internal fp16 config when present, else
                          reported as absolute AL only)
  thead/tembed         -> target_<kind>__fp16-like config when present
  tbody                -> target_body__w16a16 absent; anchor = matrix T16_D16

Because every ablation shard carries per-prompt ALs, any two configs in the
same group are paired; we emit (a) per-config AL means, (b) paired deltas vs
the LEAST-quantized config of the same component family, with 10k
cluster-bootstrap CIs and the study's equivalence verdicts.

Outputs into <run_dir>/analysis/: component_al.csv, component_contrasts.csv,
summary.json.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

EPS_ABS = 0.05
EPS_REL = 0.01
N_BOOT = 10_000
SEED = 0
MODE_ORDER = ["w8a16", "w16a8", "w8a8", "w4a16", "w16a4", "w4a4"]


def family(config):
    """component family = config name minus the trailing __<mode>."""
    return config.rsplit("__", 1)[0] if "__" in config else config


def mode(config):
    return config.rsplit("__", 1)[1] if "__" in config else ""


def paired_boot(a, b, rng):
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), \
        float(np.quantile(boots, 0.975))


def verdict(lo, hi, base_mean):
    eps = max(EPS_ABS, EPS_REL * abs(base_mean))
    if lo > eps or hi < -eps:
        return "different"
    if -eps <= lo and hi <= eps:
        return "practically_equivalent"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    sh = os.path.join(rd, "shards")
    per = defaultdict(dict)     # config -> prompt_id -> al
    grp = {}
    for fn in sorted(os.listdir(sh)):
        if not fn.startswith("comp__"):
            continue
        with open(os.path.join(sh, fn)) as f:
            for r in csv.DictReader(f):
                per[r["config"]][r["prompt_id"]] = float(r["mean_acceptance"])
                grp[r["config"]] = r["group"]
    an = os.path.join(rd, "analysis")
    os.makedirs(an, exist_ok=True)
    rng = np.random.default_rng(SEED)

    al_rows = []
    for cfgn in sorted(per):
        als = np.array(list(per[cfgn].values()))
        al_rows.append(dict(config=cfgn, group=grp[cfgn],
                            family=family(cfgn), mode=mode(cfgn),
                            n_prompts=len(als),
                            al_mean=round(float(als.mean()), 4),
                            al_sd=round(float(als.std(ddof=1)), 4)
                            if len(als) > 1 else 0.0))
    _write(os.path.join(an, "component_al.csv"), al_rows)

    # paired contrasts: within each family, every mode vs the family's
    # least-destructive mode (highest AL mean, typically w8a16)
    cons = []
    fams = defaultdict(list)
    for r in al_rows:
        fams[r["family"]].append(r)
    for fam, rows in sorted(fams.items()):
        if len(rows) < 2:
            continue
        base = max(rows, key=lambda r: r["al_mean"])
        common_all = None
        for r in rows:
            if r is base:
                continue
            common = sorted(set(per[r["config"]]) & set(per[base["config"]]))
            if not common:
                continue
            a = np.array([per[r["config"]][p] for p in common])
            b = np.array([per[base["config"]][p] for p in common])
            diff, lo, hi = paired_boot(a, b, rng)
            cons.append(dict(family=fam, config=r["config"],
                             baseline=base["config"], n_prompts=len(common),
                             al=round(float(a.mean()), 4),
                             al_base=round(float(b.mean()), 4),
                             diff=round(diff, 4), ci_lo=round(lo, 4),
                             ci_hi=round(hi, 4),
                             verdict=verdict(lo, hi, float(b.mean()))))
    # explicit branch-group cross-config pairs (A-vs-B recovery, e-vs-h)
    BRANCH_PAIRS = [
        ("branch_B_branchwise_W16A4", "branch_A_fullconcat_W16A4"),
        ("branch_B_branchwise_W4A4", "branch_A_fullconcat_W4A4"),
        ("branch_C_eA4_hFP16", "branch_C_eFP16_hA4"),
        ("branch_C_eA4_hA8", "branch_C_eA8_hA4"),
    ]
    for ca, cb in BRANCH_PAIRS:
        if ca not in per or cb not in per:
            continue
        common = sorted(set(per[ca]) & set(per[cb]))
        if not common:
            continue
        a = np.array([per[ca][p] for p in common])
        b = np.array([per[cb][p] for p in common])
        diff, lo, hi = paired_boot(a, b, rng)
        cons.append(dict(family="branch_pair", config=ca, baseline=cb,
                         n_prompts=len(common),
                         al=round(float(a.mean()), 4),
                         al_base=round(float(b.mean()), 4),
                         diff=round(diff, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4),
                         verdict=verdict(lo, hi, float(b.mean()))))
    _write(os.path.join(an, "component_contrasts.csv"), cons)

    summ = dict(run_dir=rd, n_configs=len(al_rows), n_contrasts=len(cons),
                n_boot=N_BOOT,
                families={f: {r["mode"]: r["al_mean"] for r in rows}
                          for f, rows in sorted(fams.items())})
    with open(os.path.join(an, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return 0


def _write(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
