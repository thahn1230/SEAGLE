"""Gamma exactly once, through the FIRST pre-R projection only — multi-depth
chain through the REAL tiny cnets decoder with the actual split-module class.

Correct: fc = ConcatSelectiveProjection(first,[W_e|W_h D_γ R1]; rec,[W_e|W_h R1])
+ post-R1, decoder R1-conjugated, embedding ORIGINAL. Must match the unrotated
reference at every depth. Controls (first-for-recurrent, recurrent-for-first,
no_output_R) must diverge."""

import torch
import torch.nn as nn

from b2_common import rel_l2, tiny_setup
from eagle_spinquant import rotation_aware as ra
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveProjection, PostProjectionR1,
    build_concat_selective_weights)

TOL_OK = 1e-6
TOL_BAD = 1e-3
DEPTH = 4


def _mk_split(W_first, W_rec, bias, R1, nc=None):
    def lin(Wt):
        l = nn.Linear(Wt.shape[1], Wt.shape[0], bias=True)
        l.weight.data = Wt.double()
        l.bias.data = bias.double()
        return l.double()
    return ConcatSelectiveProjection(lin(W_first), lin(W_rec),
                                     PostProjectionR1(R1).double(), nc)


@torch.no_grad()
def _chain(model, split, first_hidden, ids, selects):
    outs, h = [], first_hidden
    for k, ids_k in enumerate(ids):
        split.select = selects[k]
        h = model(h, input_ids=ids_k)
        outs.append(h)
    return outs


def test_gamma_once_chain_and_controls():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=DEPTH)

    # reference: ORIGINAL model
    m.load_state_dict(sd)
    ref = _chain(m, type("S", (), {"select": None})(), h_t, ids,
                 ["x"] * DEPTH) if False else None
    outs_ref = []
    h = h_t
    for ids_k in ids:
        h = m(h, input_ids=ids_k)
        outs_ref.append(h)

    # concat-selective: decoder conjugated, embedding ORIGINAL, split fc
    conv, _ = ra.convert_draft_state(sd, R1, gamma, mode="r1")
    new_sd = {k: v.clone() for k, v in sd.items()}
    for k, v in conv.items():
        if k.startswith("layers.0."):
            new_sd[k] = v
    W_first, W_rec, bias = build_concat_selective_weights(sd, R1, gamma)

    def run(selects, nc=None):
        m.load_state_dict({k: v.to(m.fc.weight.dtype) for k, v in new_sd.items()})
        split = _mk_split(W_first, W_rec, bias, R1, nc)
        orig_fc = m.fc
        m.fc = split
        try:
            return _chain(m, split, a_t, ids, selects)
        finally:
            m.fc = orig_fc

    good = run(["first"] + ["recurrent"] * (DEPTH - 1))
    errs = [rel_l2(o @ R1.t(), r) for o, r in zip(good, outs_ref)]
    assert max(errs) < TOL_OK, f"correct chain diverges: {errs}"

    # control: first projection reused recurrently (gamma re-applied)
    bad = run(["first"] * DEPTH)
    errs_b = [rel_l2(o @ R1.t(), r) for o, r in zip(bad, outs_ref)]
    assert errs_b[0] < TOL_OK and max(errs_b[1:]) > TOL_BAD, errs_b

    # control: recurrent projection for the first forward (gamma omitted)
    bad = run(["recurrent"] * DEPTH)
    errs_b = [rel_l2(o @ R1.t(), r) for o, r in zip(bad, outs_ref)]
    assert errs_b[0] > TOL_BAD, errs_b

    # control: output R omitted -> decoder receives unrotated features
    bad = run(["first"] + ["recurrent"] * (DEPTH - 1), nc="no_output_R")
    errs_b = [rel_l2(o @ R1.t(), r) for o, r in zip(bad, outs_ref)]
    assert errs_b[0] > TOL_BAD, errs_b


if __name__ == "__main__":
    test_gamma_once_chain_and_controls()
    print("OK")
