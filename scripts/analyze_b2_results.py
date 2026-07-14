#!/usr/bin/env python
"""B2 matrix analysis: paired prompt-level bootstrap (10k) + practical-
equivalence (TOST-style) decisions + acceptance-by-depth curves.

Pre-registered equivalence margin (docs/B2_EXPERIMENT_PROTOCOL.md):
    epsilon = max(0.05 accepted tokens/round, 1% of the baseline mean)
Sensitivity margins: 0.02 / 0.05 / 0.10.

Decisions: CI>+eps meaningful increase; CI<-eps meaningful decrease;
CI within [-eps,+eps] practically equivalent; else inconclusive.

Usage: python scripts/analyze_b2_results.py --run-dir runs/b2_matrix_<ts>
(CPU only.)
"""

import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")
N_BOOT = 10_000
PPL = {"FP16": 6.945, "fake_W4A16": 8.938, "fake_W4A4": 10.627,
       "fake_W4A4KV4": 10.939, "FP16_rotated": 6.945}

# (treatment, baseline, family)
PAIRINGS = [
    ("Q10_targetW4A4_draftFP16_archA", "Q00_targetFP16_draftFP16_stock", "target_only"),
    ("Q10s_targetW4A4_draftFP16_B2split", "Q00_targetFP16_draftFP16_stock", "target_only"),
    ("TQ1_targetW4A16_draftFP16_archA", "Q00_targetFP16_draftFP16_stock", "target_only"),
    ("TQ4_targetW4A4KV4_draftFP16_archA", "Q00_targetFP16_draftFP16_stock", "target_only"),
    ("FP02_B2split_draftFP16", "Q00_targetFP16_draftFP16_stock", "rotation_control"),
    ("DQ_first_only_W4A4", "FP02_B2split_draftFP16", "draft_only"),
    ("DQ_recurrent_only_W4A4", "FP02_B2split_draftFP16", "draft_only"),
    ("DQ_both_proj_W4A4", "FP02_B2split_draftFP16", "draft_only"),
    ("DQ_ar_only_W4A4", "FP02_B2split_draftFP16", "draft_only"),
    ("Q01_targetFP16rot_draftW4A4_full", "FP02_B2split_draftFP16", "draft_only"),
    ("DQ_first_only_W4A16", "FP02_B2split_draftFP16", "draft_only"),
    ("DQ_recurrent_only_W4A16", "FP02_B2split_draftFP16", "draft_only"),
    ("Q11_targetW4A4_draftW4A4_B2split", "Q00_targetFP16_draftFP16_stock", "both"),
    ("Q11_targetW4A4_draftW4A4_B2split", "Q10_targetW4A4_draftFP16_archA", "both_vs_targetonly"),
    ("Q11_targetW4A4_draftW4A4_B2split", "Q01_targetFP16rot_draftW4A4_full", "both_vs_draftonly"),
]


def paired_boot(delta, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide(lo, hi, eps):
    if lo > eps:
        return "meaningful_increase"
    if hi < -eps:
        return "meaningful_decrease"
    if -eps <= lo and hi <= eps:
        return "practically_equivalent"
    return "inconclusive"


def depth_alphas(df):
    """Chain-style alpha_d = P(accepted>=d | accepted>=d-1); per cycle the
    accepted DRAFT tokens = delta-1 (each verification adds 1 bonus token)."""
    rows = []
    for cfg, g in df.groupby("config"):
        acc = []
        for s in g["acceptance_list"]:
            acc += [max(0, d - 1) for d in json.loads(s)]
        acc = np.array(acc)
        for d in range(1, 6):
            reach = (acc >= d - 1).sum()
            rows.append(dict(config=cfg, depth=d,
                             alpha=float((acc >= d).sum() / reach) if reach else np.nan,
                             n_cycles_reaching=int(reach)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    df = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(rd, "shards", "accept__*.csv")))],
                   ignore_index=True)
    os.makedirs(os.path.join(ART, "summary_tables"), exist_ok=True)
    df.to_csv(os.path.join(ART, "raw_acceptance_traces",
                           "acceptance_matrix_prompt_level.csv"), index=False)

    piv = df.pivot_table(index="prompt_id", columns="config",
                         values="mean_acceptance")
    meta = df.drop_duplicates("config").set_index("config")[
        ["target_precision", "draft_precision"]]

    out = []
    for treat, base, family in PAIRINGS:
        if treat not in piv.columns or base not in piv.columns:
            continue
        d = (piv[treat] - piv[base]).dropna().values
        lo, hi = paired_boot(d)
        bmean = float(piv[base].mean())
        eps = max(0.05, 0.01 * bmean)
        row = dict(
            family=family, treatment=treat, baseline=base,
            target_precision=meta.loc[treat, "target_precision"],
            draft_precision=meta.loc[treat, "draft_precision"],
            baseline_mean=round(bmean, 4),
            treatment_mean=round(float(piv[treat].mean()), 4),
            paired_delta_mean=round(float(d.mean()), 4),
            paired_delta_median=round(float(np.median(d)), 4),
            delta_std=round(float(d.std(ddof=1)), 4),
            effect_size_dz=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3),
            ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
            n_prompts=len(d), epsilon=round(eps, 4),
            decision=decide(lo, hi, eps),
            **{"decision_eps_0.02": decide(lo, hi, 0.02),
               "decision_eps_0.05": decide(lo, hi, 0.05),
               "decision_eps_0.10": decide(lo, hi, 0.10)},
            target_ppl=PPL.get(meta.loc[treat, "target_precision"], np.nan),
            baseline_target_ppl=PPL.get(meta.loc[base, "target_precision"], np.nan))
        out.append(row)
        print(f"[b2an] {family:18s} {treat:38s} Δ={row['paired_delta_mean']:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}] -> {row['decision']}")

    eff = pd.DataFrame(out)
    eff.to_csv(os.path.join(ART, "summary_tables", "b2_effects.csv"), index=False)

    da = depth_alphas(df)
    da.to_csv(os.path.join(ART, "summary_tables", "b2_depth_alphas.csv"), index=False)

    means = df.groupby("config").agg(
        mean_acceptance=("mean_acceptance", "mean"),
        n_prompts=("prompt_id", "nunique"),
        exact_match_rate=("exact_match", "mean")).round(4)
    means.to_csv(os.path.join(ART, "summary_tables", "b2_config_means.csv"))
    print(means.to_string())

    with open(os.path.join(ART, "summary_tables", "b2_decisions.json"), "w") as f:
        json.dump(dict(n_bootstrap=N_BOOT,
                       epsilon_rule="max(0.05, 1% of baseline mean)",
                       ppl_wikitext2=PPL, effects=out), f, indent=2)
    print(f"[b2an] DONE -> {ART}/summary_tables")
    return 0


if __name__ == "__main__":
    os.makedirs(os.path.join(ART, "raw_acceptance_traces"), exist_ok=True)
    sys.exit(main())
