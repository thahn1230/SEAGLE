#!/usr/bin/env python
"""Concat-selective matrix analysis: paired bootstrap + equivalence decisions +
depth alphas + the architecture-robustness contrasts (new vs prev-B2 under
identical quant policy). CPU only.

Usage: python scripts/analyze_concat_selective_results.py --run-dir runs/cs_matrix_<ts>
"""

import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")
N_BOOT = 10_000
PPL = {"FP16": 6.945, "fake_W4A4": 10.627, "FP16_rotated": 6.945}

PAIRINGS = [
    ("Q10_targetW4A4_draftFP16_archA", "Q00_targetFP16_draftFP16_stock", "target_only"),
    ("CS_fp16_baseline", "Q00_targetFP16_draftFP16_stock", "rotation_control"),
    ("DQ_first_only_W4A4", "CS_fp16_baseline", "draft_only"),
    ("DQ_recurrent_only_W4A4", "CS_fp16_baseline", "draft_only"),
    ("DQ_both_proj_W4A4", "CS_fp16_baseline", "draft_only"),
    ("DQ_ar_only_W4A4", "CS_fp16_baseline", "draft_only"),
    ("Q01_new_draft_full_W4A4", "CS_fp16_baseline", "draft_only"),
    ("Q01_prevB2_draft_full_W4A4", "CS_fp16_baseline", "draft_only"),
    ("DQ_first_only_W4A16", "CS_fp16_baseline", "draft_only"),
    ("DQ_recurrent_only_W4A16", "CS_fp16_baseline", "draft_only"),
    ("DQ_embed_only_W4A16", "CS_fp16_baseline", "draft_only"),
    ("Q11_new_targetW4A4_draftW4A4_concat", "Q00_targetFP16_draftFP16_stock", "both"),
    ("Q11_prevB2_targetW4A4_draftW4A4", "Q00_targetFP16_draftFP16_stock", "both"),
    # architecture-robustness direct contrasts (same target, same quant policy)
    ("Q01_new_draft_full_W4A4", "Q01_prevB2_draft_full_W4A4", "arch_contrast"),
    ("Q11_new_targetW4A4_draftW4A4_concat", "Q11_prevB2_targetW4A4_draftW4A4",
     "arch_contrast"),
]


def paired_boot(d, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decide(lo, hi, eps):
    if lo > eps:
        return "meaningful_increase"
    if hi < -eps:
        return "meaningful_decrease"
    if -eps <= lo and hi <= eps:
        return "practically_equivalent"
    return "inconclusive"


def depth_alphas(df):
    rows = []
    for cfg, g in df.groupby("config"):
        acc = []
        for s in g["acceptance_list"]:
            acc += [max(0, d - 1) for d in json.loads(s)]
        acc = np.array(acc)
        for d in range(1, 6):
            reach = int((acc >= d - 1).sum())
            rows.append(dict(config=cfg, depth=d,
                             alpha=float((acc >= d).sum() / reach)
                             if reach else np.nan,
                             n_cycles_reaching=reach))
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
    os.makedirs(os.path.join(ART, "raw_acceptance"), exist_ok=True)
    df.to_csv(os.path.join(ART, "raw_acceptance",
                           "cs_matrix_prompt_level.csv"), index=False)

    piv = df.pivot_table(index="prompt_id", columns="config",
                         values="mean_acceptance")
    out = []
    for treat, base, family in PAIRINGS:
        if treat not in piv.columns or base not in piv.columns:
            continue
        d = (piv[treat] - piv[base]).dropna().values
        lo, hi = paired_boot(d)
        bmean = float(piv[base].mean())
        eps = max(0.05, 0.01 * bmean)
        out.append(dict(
            family=family, treatment=treat, baseline=base,
            baseline_mean=round(bmean, 4),
            treatment_mean=round(float(piv[treat].mean()), 4),
            paired_delta_mean=round(float(d.mean()), 4),
            delta_std=round(float(d.std(ddof=1)), 4),
            effect_size_dz=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3),
            ci95_lo=round(lo, 4), ci95_hi=round(hi, 4), n_prompts=len(d),
            epsilon=round(eps, 4), decision=decide(lo, hi, eps)))
        print(f"[csan] {family:16s} {treat:42s} Δ={out[-1]['paired_delta_mean']:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}] -> {out[-1]['decision']}")
    pd.DataFrame(out).to_csv(os.path.join(ART, "summary_tables",
                                          "cs_effects.csv"), index=False)
    depth_alphas(df).to_csv(os.path.join(ART, "summary_tables",
                                         "cs_depth_alphas.csv"), index=False)
    means = df.groupby("config").agg(
        mean_acceptance=("mean_acceptance", "mean"),
        n_prompts=("prompt_id", "nunique"),
        exact_match_rate=("exact_match", "mean")).round(4)
    means.to_csv(os.path.join(ART, "summary_tables", "cs_config_means.csv"))
    print(means.to_string())
    with open(os.path.join(ART, "summary_tables", "cs_decisions.json"), "w") as f:
        json.dump(dict(n_bootstrap=N_BOOT, ppl_wikitext2=PPL, effects=out),
                  f, indent=2)
    print(f"[csan] DONE -> {ART}/summary_tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
