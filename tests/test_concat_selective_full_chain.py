"""§12.5 full chain: original unrotated EAGLE chain vs concat-selective chain at
depths 1..5 on the REAL tiny cnets decoder, logging feature/logit metrics.
Also asserts the 7B gate artifact when present."""

import json
import os

import torch
import torch.nn as nn

from b2_common import PROJECT_ROOT, rel_l2, tiny_setup
from eagle_spinquant import rotation_aware as ra
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveProjection, PostProjectionR1, build_concat_selective_weights)

DEPTHS = 5
ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")


def test_tiny_full_chain():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=DEPTHS)
    m.load_state_dict(sd)
    ref, h = [], h_t
    for k in ids:
        h = m(h, input_ids=k)
        ref.append(h)

    conv, _ = ra.convert_draft_state(sd, R1, gamma, mode="r1")
    new_sd = {k: v.clone() for k, v in sd.items()}
    for k, v in conv.items():
        if k.startswith("layers.0."):
            new_sd[k] = v
    m.load_state_dict({k: v.to(m.fc.weight.dtype) for k, v in new_sd.items()})
    W_first, W_rec, bias = build_concat_selective_weights(sd, R1, gamma)

    def lin(Wt):
        l = nn.Linear(Wt.shape[1], Wt.shape[0], bias=True)
        l.weight.data = Wt.double(); l.bias.data = bias.double()
        return l.double()
    split = ConcatSelectiveProjection(lin(W_first), lin(W_rec),
                                      PostProjectionR1(R1).double())
    orig_fc = m.fc
    m.fc = split
    outs, h = [], a_t
    for k, ids_k in enumerate(ids):
        split.select = "first" if k == 0 else "recurrent"
        h = m(h, input_ids=ids_k)
        outs.append(h)
    m.fc = orig_fc

    W_head = torch.randn(97, h_t.shape[-1], dtype=torch.float64)
    rows = []
    for d, (o, r) in enumerate(zip(outs, ref), start=1):
        un = o @ R1.t()
        lg_ref = r @ W_head.t()
        lg_rot = o @ (W_head @ R1).t()
        rows.append(dict(depth=d,
                         feature_rel_l2=rel_l2(un, r),
                         feature_cosine=float(torch.nn.functional.cosine_similarity(
                             un.flatten(), r.flatten(), 0)),
                         feature_max_abs=float((un - r).abs().max()),
                         logit_rel_l2=rel_l2(lg_rot, lg_ref),
                         top1_agree=float((lg_rot.argmax(-1)
                                           == lg_ref.argmax(-1)).float().mean())))
    os.makedirs(os.path.join(ART, "fp64_algebra"), exist_ok=True)
    with open(os.path.join(ART, "fp64_algebra", "tiny_chain.json"), "w") as f:
        json.dump(rows, f, indent=2)
    assert all(r["feature_rel_l2"] < 1e-6 for r in rows), rows
    assert all(r["top1_agree"] == 1.0 for r in rows), rows


def test_7b_gate_artifact_if_present():
    p = os.path.join(ART, "fp16_equivalence", "fp_equivalence.json")
    if not os.path.isfile(p):
        import pytest
        pytest.skip("run scripts/validate_concat_selective_fp.py first")
    g = json.load(open(p))
    assert g["stop_gate_a_pass"], g.get("gate_reasons")


if __name__ == "__main__":
    test_tiny_full_chain()
    print("OK")
