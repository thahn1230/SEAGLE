"""Tree-generation gate (artifact-backed): with quantization disabled the
concat-selective architecture must preserve the target's greedy output exactly,
match the previous rotated-embedding B2, and the negative controls must
degrade (paired criterion)."""

import json
import os

from b2_common import PROJECT_ROOT

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")


def test_gate_and_architecture_comparison():
    p = os.path.join(ART, "fp16_equivalence", "fp_equivalence.json")
    if not os.path.isfile(p):
        import pytest
        pytest.skip("run scripts/validate_concat_selective_fp.py first")
    g = json.load(open(p))
    assert g["stop_gate_a_pass"] is True, g.get("gate_reasons")
    per = {r["config"]: r for r in g["configs"]}
    for cfg in ("F0_stock", "F1_prev_B2_rotated_embedding",
                "F2_concat_selective_explicit", "F3_concat_selective_folded"):
        assert per[cfg]["exact_match_rate"] == 1.0, (cfg, per[cfg])
    # F2 == F3 == F1 acceptance (all correct architectures agree)
    accs = [per[c]["mean_acceptance"] for c in
            ("F1_prev_B2_rotated_embedding", "F2_concat_selective_explicit",
             "F3_concat_selective_folded")]
    assert max(accs) - min(accs) < 0.05, accs
    for cfg, st in g["negative_control_paired_stats"].items():
        assert st["paired_mean_drop"] > 0.1 and \
            st["frac_prompts_degraded"] >= 0.75, (cfg, st)


if __name__ == "__main__":
    test_gate_and_architecture_comparison()
    print("OK")
