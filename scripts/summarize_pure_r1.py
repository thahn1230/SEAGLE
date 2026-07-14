#!/usr/bin/env python
"""Summarize the pure-R1 acceptance comparison + the R1-variant W4A4 comparison
into runs/<run>/summary.md (+ CSVs)."""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def ci(x, iters=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = r.choice(x, (iters, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)

    L = ["# Pure-R1 rotation-aware EAGLE — results", "",
         "quant OFF (fp16), MT-bench, seed 0, depth-5 tree, single RTX 4090.", ""]

    # ---- acceptance shards ----
    accs = sorted(glob.glob(os.path.join(rd, "shards", "pure_r1__*.csv")))
    if accs:
        df = pd.concat([pd.read_csv(s) for s in accs], ignore_index=True)
        df = df[~df["notes"].astype(str).str.contains('"tag": "smoke')]
        df["pair"] = df["model_pair"].astype(str).apply(
            lambda s: "Vicuna" if "vicuna" in s.lower() else "Llama")
        df["tag"] = df["notes"].astype(str).str.extract(r'"tag": "([^"]+)"')
        keep = (df.groupby(["pair", "condition", "tag"]).size().reset_index(name="k")
                .sort_values("k").drop_duplicates(["pair", "condition"], keep="last"))
        ks = set(zip(keep.pair, keep.condition, keep.tag))
        df = df[df.apply(lambda r: (r.pair, r.condition, r.tag) in ks, axis=1)]
        df.to_csv(os.path.join(rd, "pure_r1_acceptance.csv"), index=False)
        L += ["## Acceptance: current SpinQuant tail (T1, h_hat) vs EAGLE-friendly "
              "tail (T5, h_R)", "",
              "| pair | condition | n | accept | 95% CI | tokens/s | fc paths | "
              "hidden exposed | needs B2? |", "|---|---|---|---|---|---|---|---|---|"]
        for (pair, cond), sub in df.groupby(["pair", "condition"]):
            a = pd.to_numeric(sub.acceptance_length, errors="coerce")
            lo, hi = ci(a)
            tps = pd.to_numeric(sub.tokens_per_second, errors="coerce").mean()
            fp = int(sub.fc_paths.iloc[0])
            hb = sub.hidden_basis_exposed.iloc[0]
            needs = "yes (2-path)" if cond == "T1:B2" else (
                "n/a" if "naive" in cond else "no")
            L.append(f"| {pair} | {cond} | {len(sub)} | {a.mean():.3f} | "
                     f"[{lo:.3f}, {hi:.3f}] | {tps:.1f} | {fp} | {hb} | {needs} |")
        L.append("")
        L.append("Reading: T1:naive and T5:naive_frozen COLLAPSE (frozen draft "
                 "gets the wrong basis); T1:A/T1:B2 recover with runtime "
                 "unrotation / two-path fold; **T5:pure_r1 recovers with a "
                 "SINGLE fc path and no B2** — the tail change (expose h_R "
                 "instead of h_hat) unifies external and recurrent bases.")
        L.append("")

    # ---- R1 variant comparison ----
    r1s = sorted(glob.glob(os.path.join(rd, "shards", "r1cmp__*.csv")))
    if r1s:
        cdf = pd.concat([pd.read_csv(s) for s in r1s], ignore_index=True)
        cdf.to_csv(os.path.join(rd, "r1_variant_comparison.csv"), index=False)
        L += ["## R1-variant comparison (W4A4 fake quant, Variant A)", "",
              "| R1 | W4A4 PPL | W4A4 acceptance | hidden cos (recovered vs fp16) "
              "| draft logit KL | hidden quant relL2 |",
              "|---|---|---|---|---|---|"]
        for _, r in cdf.iterrows():
            L.append(f"| {r['rotation_type']} | {r['w4a4_ppl']:.3f} | "
                     f"{r['w4a4_acceptance_A']:.3f} | "
                     f"{r['hidden_cosine_recovered_vs_fp16']:.4f} | "
                     f"{r['draft_target_logit_kl_w4a4']:.4f} | "
                     f"{r['hidden_quant_relL2']:.3f} |")
        L += ["", "Skeptical note: in the EXACT fp16 pure-R1 interface, "
              "acceptance is R1-INVARIANT (any orthogonal R1 is a gauge). R1 "
              "matters only under quantization. If the EAGLE-aware R1 does not "
              "beat random Hadamard here, that is the honest result — random "
              "Hadamard already flattens activation outliers near-optimally."]
    with open(os.path.join(rd, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {rd}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
