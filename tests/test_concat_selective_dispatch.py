"""Dispatch mechanics (CPU) + real-7B dispatch-trace assertions (artifact)."""

import json
import os

import torch
import torch.nn as nn

from b2_common import PROJECT_ROOT
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveProjection, PostProjectionR1)

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")
TRACE = os.path.join(ART, "dispatch_traces", "dispatch_trace.jsonl")


def test_module_mechanics():
    torch.manual_seed(0)
    f = nn.Linear(8, 4); r = nn.Linear(8, 4)
    R1 = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))[0].float()
    s = ConcatSelectiveProjection(f, r, PostProjectionR1(R1))
    z = torch.randn(2, 8)
    try:
        s(z); assert False, "unarmed forward must fail"
    except RuntimeError:
        pass
    s.select = "first"
    y1 = s(z)
    s.select = "recurrent"
    y2 = s(z)
    assert s.calls_first == 1 and s.calls_recurrent == 1
    assert s.post_projection_R1.n_calls == 2
    # output R applied: recompute manually
    assert torch.allclose(y1, (f(z).float() @ R1).to(y1.dtype), atol=1e-5)
    assert torch.allclose(y2, (r(z).float() @ R1).to(y2.dtype), atol=1e-5)


def test_real_trace_contract():
    if not os.path.isfile(TRACE):
        import pytest
        pytest.skip("run scripts/validate_concat_selective_fp.py first")
    rows = [json.loads(l) for l in open(TRACE)]
    rows = [r for r in rows if r.get("nc_mode", "") == ""]
    assert rows
    from collections import defaultdict
    per = defaultdict(list)
    for r in rows:
        per[(r["config"], r["prompt_id"], r["verification_round"])].append(r)
    for key, rs in per.items():
        firsts = [x for x in rs if x["selected_projection"]
                  == "projection_first_preR"]
        assert len(firsts) == 1 and firsts[0]["draft_forward_index"] == 0, key
        for x in rs:
            assert x["embedding_basis"] == "original", key
            assert x["output_basis"].startswith("rotated"), key
            assert x["post_projection_R1_executed"] is True, key
            if x["draft_forward_index"] > 0:
                assert x["selected_projection"] == "projection_recurrent_preR"
                assert x["gamma_applied"] is False, key
            else:
                assert x["gamma_applied"] is True, key


if __name__ == "__main__":
    test_module_mechanics()
    print("OK; trace present:", os.path.isfile(TRACE))
