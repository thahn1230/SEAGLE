#!/usr/bin/env python
"""Merge spinquant_draft_pure_r1 shards -> baseline_comparison.csv,
acceptance_summary.csv, ablation_results.csv."""
import argparse, glob, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True)
a = ap.parse_args()
rd = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(PROJECT_ROOT, a.run_dir)
shards = sorted(glob.glob(os.path.join(rd, "shards", "*.csv")))
df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
# dedup: prefer the most recent shard per (baseline_name, quant_mode) if dup
df = df.drop_duplicates(["baseline_name", "quant_mode", "prompt_id"], keep="last")
df.to_csv(os.path.join(rd, "baseline_comparison.csv"), index=False)

summ = df.groupby(["quant_mode", "baseline_name"]).agg(
    acceptance_mean=("acceptance_length_mean", "mean"),
    exact_match_rate=("exact_token_match", "mean"),
    verifier_rel_l2=("verifier_logit_rel_l2_mean", "mean"),
    n=("prompt_id", "count")).round(4).reset_index()
summ.to_csv(os.path.join(rd, "acceptance_summary.csv"), index=False)

DESC = {"B0":"original EAGLE (unrotated)","B1":"unfused target + original draft",
        "B2":"prev [R1.T,I]/[R1.T,R1.T]+fused head (neg ctrl)","B3":"NEW pure-R1 draft",
        "A1":"external h NOT rotated","A2":"@R1 once per prompt","A3":"recurrent f_R rotated again",
        "A4":"input rotated but fc NOT conjugated","A5":"PL-conjugated, no R2/R3/R4",
        "A6":"gamma_f-fused head on f_R"}
abl = summ[(summ.baseline_name.str.startswith("A")) & (summ.quant_mode=="none")].copy()
abl["expectation"] = abl.baseline_name.map(DESC)
b3 = summ[(summ.baseline_name=="B3")&(summ.quant_mode=="none")]["acceptance_mean"]
b3v = float(b3.iloc[0]) if len(b3) else float("nan")
abl["vs_B3_delta"] = (abl["acceptance_mean"] - b3v).round(3)
abl.to_csv(os.path.join(rd, "ablation_results.csv"), index=False)
print(summ.to_string(index=False))
print(f"\nB3 fp16 = {b3v}")
print(f"-> {rd}/baseline_comparison.csv, acceptance_summary.csv, ablation_results.csv")
