#!/usr/bin/env python
"""Merge rotation-aware study shards -> canonical CSVs + summary.md with
verdicts on the 5 falsifiable predictions of
docs/rotation_aware_eagle_overview.md §6.

Usage: python scripts/summarize_rotation_aware.py --run-id <id> [--audit-dir <dir>]
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSVS = ["rotated_loop_oracle", "embedding_branch_ablation",
        "algebraic_rotated_draft", "rotation_aware_training"]


def bootstrap_ci(x, iters=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(df, run_dir, drop_smoke=True):
    if drop_smoke:
        df = df[~df["notes"].astype(str).str.contains('"tag": "smoke') &
                ~df["notes"].astype(str).str.contains('"tag": "f_smoke')]
    rows = []
    for v, sub in df.groupby("variant"):
        acc = pd.to_numeric(sub["acceptance_length"], errors="coerce")
        lo, hi = bootstrap_ci(acc)
        rows.append({
            "variant": v, "n": len(sub), "accept_mean": acc.mean(),
            "ci": (lo, hi),
            "conv_ms": pd.to_numeric(sub["basis_conversion_time_ms"],
                                     errors="coerce").mean(),
            "n_conv": pd.to_numeric(sub["number_of_basis_conversions"],
                                    errors="coerce").mean(),
            "head": ("W@R1 (R1-folded)" if v == "F_R_only" else
                     "rotated" if sub["rotated_lm_head_used"].iloc[0] in (True, "True")
                     else "original"),
            "emb": sub["embedding_basis"].iloc[0],
            "recyc": sub["recycled_feature_basis"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values("accept_mean", ascending=False)


def md_table(t):
    L = ["| variant | n | accept mean | 95% CI | head | embedding basis | "
         "recycled basis | conv/prompt (ms) | #conv |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['variant']} | {r['n']} | {r['accept_mean']:.3f} | "
                 f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['head']} | "
                 f"{r['emb']} | {r['recyc']} | "
                 f"{'' if not np.isfinite(r['conv_ms']) else f'{r.conv_ms:.2f}'} | "
                 f"{'' if not np.isfinite(r['n_conv']) else f'{r.n_conv:.0f}'} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--audit-dir", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)

    frames = {}
    all_rows = []
    for name in CSVS:
        shards = sorted(glob.glob(os.path.join(run_dir, "shards", f"{name}__*.csv")))
        if not shards:
            continue
        df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        df.to_csv(os.path.join(run_dir, f"{name}.csv"), index=False)
        frames[name] = df
        all_rows.append(df)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            os.path.join(run_dir, "acceptance_main.csv"), index=False)

    diag = {}
    dpath = os.path.join(run_dir, "algebraic_rotated_draft_diagnostics.json")
    if os.path.isfile(dpath):
        diag = json.load(open(dpath))
    ledger = {}
    if args.audit_dir:
        lp = os.path.join(PROJECT_ROOT, args.audit_dir, "basis_ledger_verdicts.json")
        if os.path.isfile(lp):
            ledger = json.load(open(lp))

    L = ["# Rotation-aware EAGLE study — summary", "",
         "All numbers quant OFF, MT-bench first-n prompts, seed 0, depth-5 "
         "tree, fp16 pipeline. Oracles (D) are diagnostics, NOT deployment "
         "designs.", ""]

    ver = {}
    if "rotated_loop_oracle" in frames:
        t = table(frames["rotated_loop_oracle"], run_dir)
        L += ["## Phase 2 — rotated-loop oracle (n=20)", ""] + md_table(t) + [""]
        g = {r["variant"]: r["accept_mean"] for _, r in t.iterrows()}
        ver["P1_D_recovers_A_B2"] = bool(
            abs(g.get("D1", 0) - g.get("A", 1)) < 0.05 * g.get("A", 1)
            and abs(g.get("D2", 0) - g.get("A", 1)) < 0.05 * g.get("A", 1))
    if "embedding_branch_ablation" in frames:
        t = table(frames["embedding_branch_ablation"], run_dir)
        L += ["## Phase 3 — embedding-branch ablation (n=20)", ""] + md_table(t) + [""]
        g = {r["variant"]: r["accept_mean"] for _, r in t.iterrows()}
        ver["P2_unfolded_e_rotation_collapses"] = bool(
            g.get("E_r1", 9) < 0.6 * g.get("D1", 1)
            and g.get("E_r1_gamma", 9) < 0.6 * g.get("D1", 1))
        ver["P2b_cofolded_e_rotation_exact"] = bool(
            abs(g.get("E_r1_cofold", 0) - g.get("D1", 1)) < 0.05 * g.get("D1", 1))
    if "algebraic_rotated_draft" in frames:
        t = table(frames["algebraic_rotated_draft"], run_dir)
        L += ["## Phase 4 — algebraic fully rotated draft (n=20)", ""] + md_table(t) + [""]
        g = {r["variant"]: r["accept_mean"] for _, r in t.iterrows()}
        ver["P3_F_R_only_exact"] = bool(
            abs(g.get("F_R_only", 0) - g.get("A", 1)) < 0.05 * g.get("A", 1))
        ver["P4_F_R_gamma_degrades"] = bool(
            g.get("F_R_gamma", 9) < 0.9 * g.get("A", 1))
        if diag:
            v = diag.get("verdicts", {})
            L += [f"- single-forward rel-L2: F_R_only "
                  f"{v.get('F_R_only_single_forward_rel_l2', float('nan')):.4f}, "
                  f"F_R_gamma {v.get('F_R_gamma_single_forward_rel_l2', float('nan')):.4f}",
                  f"- breaking layer (F_R_gamma): {v.get('F_R_gamma_breaking_layer')}",
                  f"- rms(x@S)/rms(x) on real residuals: "
                  f"{json.dumps({k: round(x, 3) for k, x in diag.get('rms_ratio', {}).get('gamma_p0', {}).items() if isinstance(x, float)})}",
                  ""]
    if "rotation_aware_training" in frames:
        t = table(frames["rotation_aware_training"], run_dir, drop_smoke=False)
        L += ["## Phase 5 — rotation-aware trained draft (G)", ""] + md_table(t) + [""]
        g = {r["variant"]: r["accept_mean"] for _, r in t.iterrows()}
        if "G_native" in g:
            ver["P5_G_matches_B2"] = bool(g["G_native"] > 0.95 * g.get("B2", g.get("A", 9)))
            ver["G_native_accept"] = g["G_native"]

    if ledger:
        L += ["## Basis-ledger verdicts (Phase 1)", "",
              "```json", json.dumps(ledger, indent=2), "```", ""]
    L += ["## Prediction verdicts (overview doc §6)", "",
          "```json", json.dumps(ver, indent=2), "```", "",
          "## Claims discipline", "",
          "Supported: B failure is caused by recycled features entering the "
          "folded fc in the wrong basis (ledger cos 0.9999 to original-basis "
          "reference; D oracles recover A exactly). Draft recycled features "
          "are empirically original-basis. A fully rotation-aware draft is "
          "NOT feasible by algebraic conversion in a single path: the "
          "obstruction is the draft's post_attention_layernorm interacting "
          "with non-uniform gamma_f (F_R_only exact, F_R_gamma breaks there); "
          "recovering the correct rms statistic from S-basis features costs "
          "one dense GEMV per token — the same cost class B2 avoids.",
          "",
          "NOT claimed: B2 optimality; that the draft should always stay "
          "original-basis (E_r1_cofold and F_R_only show bases are movable); "
          "any deployment speed from oracles; G conclusions beyond the "
          "training budget actually run."]
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(ver, indent=2))
    print(f"-> {run_dir}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
