"""Dispatch correctness. CPU part: B2SplitProjection mechanics (explicit select,
no shape guessing, hard assert when unarmed). Artifact part: the REAL 7B
dispatch trace from scripts/validate_b2_fp_equivalence.py — projection_first
exactly once per verification round, at draft_forward_index 0, never on
draft-originated rows."""

import json
import os

import torch
import torch.nn as nn

from b2_common import PROJECT_ROOT
from eagle_spinquant.b2_projection import B2SplitProjection

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")
TRACE = os.path.join(ART, "projection_dispatch_trace.jsonl")


def test_split_module_mechanics():
    f = nn.Linear(8, 4); r = nn.Linear(8, 4)
    with torch.no_grad():
        f.weight.fill_(1.0); r.weight.fill_(-1.0)
        f.bias.zero_(); r.bias.zero_()
    s = B2SplitProjection(f, r)
    z = torch.ones(2, 8)
    try:
        s(z); assert False, "forward without arming must fail"
    except AssertionError:
        pass
    s.select = "first"
    assert torch.allclose(s(z), torch.full((2, 4), 8.0))
    s.select = "recurrent"
    assert torch.allclose(s(z), torch.full((2, 4), -8.0))
    assert s.calls_first == 1 and s.calls_recurrent == 1
    assert s.projection_first is f and s.projection_recurrent is r


def test_real_trace_first_exactly_once_per_round():
    if not os.path.isfile(TRACE):
        import pytest
        pytest.skip("run scripts/validate_b2_fp_equivalence.py first")
    rows = [json.loads(l) for l in open(TRACE)]
    rows = [r for r in rows if r.get("nc_mode", "") == ""]      # correct configs only
    assert rows, "no non-nc dispatch rows"
    from collections import defaultdict
    per_round = defaultdict(list)
    for r in rows:
        per_round[(r["config"], r["prompt_id"], r["verification_round"])].append(r)
    for key, rs in per_round.items():
        firsts = [r for r in rs if r["projection_selected"] == "projection_first"]
        assert len(firsts) == 1, (key, len(firsts))
        assert firsts[0]["draft_forward_index"] == 0, key
        assert firsts[0]["feature_origin"] == "target_first", key
        assert firsts[0]["gamma_already_included"] is False, key
        for r in rs:
            if r["draft_forward_index"] > 0:
                assert r["projection_selected"] == "projection_recurrent", key
                assert r["feature_origin"] == "draft_recurrent", key
                assert r["gamma_already_included"] is True, key


if __name__ == "__main__":
    test_split_module_mechanics()
    print("mechanics OK; trace present:", os.path.isfile(TRACE))
