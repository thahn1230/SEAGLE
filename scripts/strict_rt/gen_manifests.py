#!/usr/bin/env python
"""Materialize + verify the canonical eval-prompt manifests (Gate F)
for the strict SEAGLE-RT run dir, and cross-check against the AAQ
reference manifests' sha256 (the canonical pinning)."""
import json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eagle_spinquant.eval_datasets import (
    load_eval_prompts, write_or_verify_manifest, check_no_overlap)

RUN = os.path.join(PROJECT_ROOT, "runs",
                   "eagle1_strict_seagle_rt_native_w4a4_20260816_172552")
REF = os.path.join(PROJECT_ROOT, "runs",
                   "eagle1_acceptance_aware_qat_20260812_120000",
                   "manifests")
EXPECT = {"mtbench__eval": "82137e5372f5ff7a",
          "gsm8k__eval": "6a94d85419e8cd66",
          "sharegpt__eval": "96060988a4674227",
          "humaneval__eval": "fd806a46bdb75c16",
          "c4__calib": "ef5311ccd40e8a69"}
JOBS = [("mtbench", 80, "eval"), ("gsm8k", 200, "eval"),
        ("sharegpt", 80, "eval"), ("humaneval", 164, "eval"),
        ("c4", 20, "calib")]

ok = True
for name, n, pool in JOBS:
    _, man_new = load_eval_prompts(name, n, pool=pool)
    write_or_verify_manifest(RUN, name, pool, man_new)
    key = f"{name}__{pool}"
    man = json.load(open(os.path.join(RUN, "manifests", key + ".json")))
    sha = man["manifest_sha256"][:16]
    ref_path = os.path.join(REF, key + ".json")
    ref_sha = (json.load(open(ref_path))["manifest_sha256"][:16]
               if os.path.exists(ref_path) else None)
    match = sha == EXPECT[key] and (ref_sha in (None, sha))
    ok &= match
    print(f"{key}: n={man['n']} sha={sha} expect={EXPECT[key]} "
          f"ref={ref_sha} {'OK' if match else 'MISMATCH'}")
viol = check_no_overlap(RUN)
print("gateF_no_overlap:", "PASS" if not viol else viol)
ok &= not viol
print("MANIFESTS", "ALL OK" if ok else "FAILURE")
sys.exit(0 if ok else 1)
