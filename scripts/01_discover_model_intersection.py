#!/usr/bin/env python
"""Model intersection discovery: EAGLE-1 draft support x SpinQuant rotation.

Inspects the LOCAL repos (not just READMEs) to re-verify each candidate in
configs/model_candidates.yaml, and writes the verification results back into the
YAML under each candidate's `checks:` field. Also reports the local-repo evidence
it used (eval scripts present, SpinQuant modeling arch, train configs).

Fetches only config.json (cached first). Never downloads weights.

Usage: python scripts/01_discover_model_intersection.py [--offline] [--write]
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import yaml  # noqa: E402
from eagle_spinquant import model_discovery as md  # noqa: E402

EAGLE_DIR = os.path.join(PROJECT_ROOT, "third_party", "EAGLE")
SPINQUANT_DIR = os.path.join(PROJECT_ROOT, "third_party", "SpinQuant")
CANDIDATES_YAML = os.path.join(PROJECT_ROOT, "configs", "model_candidates.yaml")


def inspect_local_repos() -> dict:
    """Gather non-README evidence from the checked-out repos."""
    ev = {}
    # EAGLE eval scripts (per model family) + train configs
    ev["eagle_eval_scripts"] = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(EAGLE_DIR, "eagle", "evaluation", "gen_*.py")))
    ev["eagle_ge_data_scripts"] = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(EAGLE_DIR, "eagle", "ge_data", "ge_data_*.py")))
    ev["eagle_train_configs"] = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(EAGLE_DIR, "eagle", "train", "*_config.json")))
    # SpinQuant supported modeling files (architecture wrappers)
    ev["spinquant_modeling_files"] = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(SPINQUANT_DIR, "eval_utils", "modeling_*.py")) +
        glob.glob(os.path.join(SPINQUANT_DIR, "train_utils", "modeling_*.py")))
    ev["spinquant_scripts"] = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(SPINQUANT_DIR, "scripts", "*.sh")))
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="write verification checks back into the YAML")
    args = ap.parse_args()

    evidence = inspect_local_repos()
    print("=== local repo evidence ===")
    print(json.dumps(evidence, indent=2))

    candidates = md.load_candidates()
    any_fail = False
    enriched = []
    for cand in candidates:
        md.check_candidate(cand, allow_network=not args.offline)
        rec = " [RECOMMENDED]" if cand.recommended_initial_candidate else ""
        print(f"\n=== {cand.candidate_name}{rec} ===")
        print(f"    target: {cand.hf_target_model}")
        print(f"    draft : {cand.hf_eagle_draft_or_training_source}")
        for k, v in cand.checks.items():
            flag = ""
            if k.endswith("_match") and v is False:
                flag = "  <-- MISMATCH"
                any_fail = True
            print(f"    {k}: {v}{flag}")
        enriched.append(cand)

    if args.write:
        # Preserve the authored file structure, add checks + evidence.
        with open(CANDIDATES_YAML) as f:
            data = yaml.safe_load(f)
        by_name = {c.candidate_name: c for c in enriched}
        for c in data["candidates"]:
            chk = by_name[c["candidate_name"]].checks
            c["checks"] = chk
        data["_local_repo_evidence"] = evidence
        data["_verified_at"] = "2026-07-02"
        with open(CANDIDATES_YAML, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, width=100)
        print(f"\nwrote verification back to {CANDIDATES_YAML}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
