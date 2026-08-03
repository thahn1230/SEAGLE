#!/usr/bin/env python
"""Folding/fusion runtime benchmark (study §20). Rolls up:
  - kernel micro-bench (from build_eagle_fused_transform_quant_kernel,
    rerun here for fresh numbers): K0 concat+A4 baseline, K1 +scale,
    K2 pairwise Givens, K3 block-16 cross rotation — all fused Triton,
    code-parity-checked vs the official Python quantizer
  - explicit Python transform path (unfused) for the same shapes
  - e2e ms/token from evaluator shards where present
Writes tables/folding_benchmark.json with the spec's terminology
(zero-additional-kernel vs fully-folded vs fused-not-folded).
"""
import argparse, csv, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    # refresh kernel bench (writes kernel_bench.json/kernel_verify.json)
    subprocess.run([PY, "scripts/"
                    "build_eagle_fused_transform_quant_kernel.py",
                    "--run-dir", rd, "--mode", "bench"], cwd=ROOT)
    kb = json.load(open(os.path.join(rd, "tables",
                                     "kernel_bench.json")))
    kv = json.load(open(os.path.join(rd, "tables",
                                     "kernel_verify.json")))
    e2e = {}
    for p in glob.glob(os.path.join(rd, "shards",
                                    "al__*__int4__mtbench.csv")):
        tag = os.path.basename(p).split("__")[1]
        if tag not in ("CMP_FULL", "CMP_FPDRAFT", "CMP_B12_ar_fp"):
            continue
        secs = toks = 0.0
        for r in csv.DictReader(open(p)):
            secs += float(r["gen_seconds"])
            toks += sum(json.loads(r["acceptance_list"]))
        e2e[tag] = round(1000 * secs / max(toks, 1), 2)
    out = dict(kernel_bench=kb, kernel_verify=kv,
               e2e_ms_per_token=e2e,
               terminology=dict(
                   fully_folded="F0/F1 (EP3P table views, W-side "
                                "folds, R2)",
                   fused_not_folded="F2 (cross-branch Q, recurrent "
                                    "rescale) — inside K1-K3, zero "
                                    "additional kernel",
                   necessarily_online="F3 (R4 down_proj Hadamard)"))
    json.dump(out, open(os.path.join(
        rd, "tables", "folding_benchmark.json"), "w"), indent=1)
    print(json.dumps(e2e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
