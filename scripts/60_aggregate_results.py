#!/usr/bin/env python
"""Aggregate all run JSONLs into a master results table (target 11 / task X-4).

Collects results/*.jsonl (+ runs/*/ *.jsonl), attaches provenance (both
third_party commits, env), and writes results/results.jsonl + results/results.csv.

Usage: python scripts/60_aggregate_results.py
"""

import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eagle_spinquant import logging_utils  # noqa: E402


def main() -> int:
    env = logging_utils.env_summary()
    prov = {"eagle_commit": env.get("eagle_commit"),
            "eagle_branch": env.get("eagle_branch"),
            "spinquant_commit": env.get("spinquant_commit"),
            "torch": env.get("torch"), "transformers": env.get("transformers")}

    sources = sorted(set(
        glob.glob(os.path.join(PROJECT_ROOT, "results", "*.jsonl")) +
        glob.glob(os.path.join(PROJECT_ROOT, "runs", "*", "*.jsonl"))))

    rows = []
    for src in sources:
        base = os.path.basename(src)
        if base in ("results.jsonl",):
            continue
        for r in logging_utils.read_jsonl(src):
            r = dict(r)
            r["_source"] = os.path.relpath(src, PROJECT_ROOT)
            r.update(prov)
            rows.append(r)

    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "results.jsonl"), rows)
    # a compact CSV with the key comparison columns
    cols = ["_source", "method", "quant_setting", "target_quant", "runtime_mode",
            "batch_size", "context_len", "avg_accept_length", "tokens_per_s",
            "ms_per_token", "peak_mem_gib", "new_tokens", "forward_rounds",
            "eagle_commit", "spinquant_commit"]
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "results.csv"), rows, cols)
    print(f"aggregated {len(rows)} rows from {len(sources)} sources")
    print(f"  -> results/results.jsonl")
    print(f"  -> results/results.csv")
    # quick provenance echo
    print(f"provenance: EAGLE {prov['eagle_commit']} ({prov['eagle_branch']}), "
          f"SpinQuant {prov['spinquant_commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
