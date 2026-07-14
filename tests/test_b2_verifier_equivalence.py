"""Test 6 (spec §13): speculative-decoding output preservation, artifact-backed.
With quantization disabled, for every correct config (FP00 stock, FP01 A-explicit,
FP01b A-folded, FP02 B2-split, FP02x explicit-Mgamma):
  vanilla target greedy tokens == target+EAGLE greedy tokens  (per prompt)
and the rotated target's greedy tokens == the stock target's greedy tokens.
Negative controls must NOT be output-preserving or must lose acceptance."""

import json
import os

from b2_common import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")


def test_verifier_and_negative_controls():
    p = os.path.join(ART, "fp_equivalence.json")
    if not os.path.isfile(p):
        import pytest
        pytest.skip("run scripts/validate_b2_fp_equivalence.py first")
    g = json.load(open(p))
    assert g["stop_gate_a_pass"] is True, g.get("gate_reasons")
    per = {r["config"]: r for r in g["configs"]}
    for cfg in ("FP00_stock", "FP01_A_explicit", "FP01b_A_folded",
                "FP02_B2_split", "FP02x_explicit_Mgamma"):
        assert per[cfg]["exact_match_rate"] == 1.0, (cfg, per[cfg])
    ncs = g["negative_control_paired_stats"]
    for cfg, st in ncs.items():
        assert st["paired_mean_drop"] > 0.1 and st["frac_prompts_degraded"] >= 0.75, \
            (cfg, "negative control did not degrade", st)


if __name__ == "__main__":
    test_verifier_and_negative_controls()
    print("OK")
