"""Test 5 (spec §13): full chain equivalence at depths 1..max on the REAL
vendored cnets.Model (tiny fp64), logging rel-L2 / cosine / max-abs per depth,
plus a 'token-logit' check through a random head (W and W·R1). The 7B fp16
end-to-end version is scripts/validate_b2_fp_equivalence.py (STOP GATE A);
this file also asserts its artifact when present."""

import json
import os

import torch

from b2_common import PROJECT_ROOT, rel_l2, run_chain, tiny_setup
from eagle_spinquant.b2_projection import build_b2_weights_arch_b

DEPTHS = 5
ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")


def test_tiny_full_chain_all_depths():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=DEPTHS)
    conv, W_first, W_rec, b_rot = build_b2_weights_arch_b(sd, R1, gamma)

    m.load_state_dict(sd)
    ref = run_chain(m, h_t, ids, [sd["fc.weight"]] * DEPTHS, sd["fc.bias"])

    m.load_state_dict({k: v.to(m.fc.weight.dtype) for k, v in conv.items()})
    outs = run_chain(m, a_t, ids, [W_first] + [W_rec] * (DEPTHS - 1), b_rot)

    W_head = torch.randn(97, h_t.shape[-1], dtype=torch.float64)
    rows = []
    for d, (o, r) in enumerate(zip(outs, ref), start=1):
        un = o @ R1.t()
        cos = float(torch.nn.functional.cosine_similarity(
            un.flatten(), r.flatten(), 0))
        logit_ref = r @ W_head.t()
        logit_rot = o @ (W_head @ R1).t()          # rotated head on rotated feat
        rows.append(dict(depth=d, feature_rel_l2=rel_l2(un, r),
                         feature_cosine=cos,
                         feature_max_abs=float((un - r).abs().max()),
                         logit_rel_l2=rel_l2(logit_rot, logit_ref),
                         top1_agree=float((logit_rot.argmax(-1)
                                           == logit_ref.argmax(-1)).float().mean())))
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "tiny_chain_equivalence.json"), "w") as f:
        json.dump(rows, f, indent=2)
    assert all(r["feature_rel_l2"] < 1e-6 for r in rows), rows
    assert all(r["logit_rel_l2"] < 1e-6 for r in rows), rows
    assert all(r["top1_agree"] == 1.0 for r in rows), rows


def test_7b_gate_artifact_if_present():
    p = os.path.join(ART, "fp_equivalence.json")
    if not os.path.isfile(p):
        import pytest
        pytest.skip("run scripts/validate_b2_fp_equivalence.py first (GPU gate)")
    gate = json.load(open(p))
    assert gate["stop_gate_a_pass"], gate


if __name__ == "__main__":
    test_tiny_full_chain_all_depths()
    print("OK tiny chain; gate artifact:",
          os.path.isfile(os.path.join(ART, "fp_equivalence.json")))
