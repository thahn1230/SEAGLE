"""State-reset correctness (spec §9): no first/recurrent state leakage across
verification rounds or prompts. Artifact-backed on the real 7B dispatch trace:
every verification round restarts at draft_forward_index 0 with
projection_first, and prompt boundaries reset the round counter."""

import json
import os

from b2_common import PROJECT_ROOT

TRACE = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study",
                     "projection_dispatch_trace.jsonl")


def test_round_and_prompt_resets():
    if not os.path.isfile(TRACE):
        import pytest
        pytest.skip("run scripts/validate_b2_fp_equivalence.py first")
    rows = [json.loads(l) for l in open(TRACE)]
    rows = [r for r in rows if r.get("nc_mode", "") == ""]
    by_cfg_prompt = {}
    for r in rows:
        by_cfg_prompt.setdefault((r["config"], r["prompt_id"]), []).append(r)
    for (cfg, pid), rs in by_cfg_prompt.items():
        # rounds start at 1 and every round's first row has idx 0 + first-proj
        rounds = {}
        for r in rs:
            rounds.setdefault(r["verification_round"], []).append(r)
        assert min(rounds) == 1, (cfg, pid, "round counter did not reset")
        for rnd, rr in rounds.items():
            idx0 = [x for x in rr if x["draft_forward_index"] == 0]
            assert idx0, (cfg, pid, rnd, "no idx-0 row: stale _fc_idx")
            assert all(x["projection_selected"] == "projection_first"
                       for x in idx0), (cfg, pid, rnd)


if __name__ == "__main__":
    test_round_and_prompt_resets()
    print("OK" if os.path.isfile(TRACE) else "trace missing (GPU gate not run)")
