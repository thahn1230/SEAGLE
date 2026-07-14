"""Test 3 (spec §13): gamma applied exactly once — full multi-depth chain
through the REAL vendored cnets decoder (tiny fp64 instance).

Correct: first step uses fc_ext (gamma inside), recurrent steps use the
conjugated fc. Must match the unrotated reference at EVERY depth.

Negative controls (must diverge):
  NC-single_folded   fc_ext reused for every step        (gamma applied k times)
  NC-gamma_omitted   conjugated fc for every step        (gamma never applied)
  NC-elementwise     a_t*γ in the rotated basis, then conjugated fc
  NC-first_for_rec / rec_for_first swaps
"""

import torch

from b2_common import rel_l2, run_chain, tiny_setup
from eagle_spinquant import rotation_aware as ra
from eagle_spinquant.b2_projection import build_b2_weights_arch_b

TOL_OK = 1e-6      # cnets rotary runs fp32 internally -> ~4e-8 noise; controls are >1e-3
TOL_BAD = 1e-3      # controls must exceed this at some depth
DEPTH = 4


def _chains():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=DEPTH)
    conv, W_first, W_rec, b_rot = build_b2_weights_arch_b(sd, R1, gamma)

    # reference: ORIGINAL model, original inputs
    m.load_state_dict(sd)
    ref = run_chain(m, h_t, ids, [sd["fc.weight"]] * DEPTH, sd["fc.bias"])

    # rotated model (decoder conjugated, embedding rotated)
    def rot_chain(fc_seq, first_hidden):
        m.load_state_dict({k: v.to(m.fc.weight.dtype) for k, v in conv.items()})
        return run_chain(m, first_hidden, ids, fc_seq, b_rot)

    return m, sd, conv, R1, gamma, n_t, h_t, a_t, ids, ref, \
        W_first, W_rec, b_rot, rot_chain


def test_gamma_exactly_once_and_negative_controls():
    (m, sd, conv, R1, gamma, n_t, h_t, a_t, ids, ref,
     W_first, W_rec, b_rot, rot_chain) = _chains()

    # ---- CORRECT split: first then recurrent -----------------------------
    outs = rot_chain([W_first] + [W_rec] * (DEPTH - 1), a_t)
    errs_ok = [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]
    assert max(errs_ok) < TOL_OK, f"correct split diverges: {errs_ok}"

    # ---- NC single_folded: fc_ext every step ------------------------------
    outs = rot_chain([W_first] * DEPTH, a_t)
    errs = [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]
    assert errs[0] < TOL_OK and max(errs[1:]) > TOL_BAD, \
        f"single_folded did not diverge after depth 0: {errs}"

    # ---- NC gamma_omitted: recurrent fc every step -------------------------
    outs = rot_chain([W_rec] * DEPTH, a_t)
    errs = [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]
    assert errs[0] > TOL_BAD, f"gamma omission had no effect at depth 0: {errs}"

    # ---- NC elementwise gamma in rotated basis ----------------------------
    outs = rot_chain([W_rec] * DEPTH, a_t * gamma)
    errs = [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]
    assert errs[0] > TOL_BAD, f"(a_t*γ) accidentally correct: {errs}"

    # ---- NC recurrent weight used for first, first for later --------------
    outs = rot_chain([W_rec, W_first] + [W_rec] * (DEPTH - 2), a_t)
    errs = [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]
    assert errs[0] > TOL_BAD and errs[1] > TOL_BAD, f"swap control passed: {errs}"


if __name__ == "__main__":
    test_gamma_exactly_once_and_negative_controls()
    print("OK")
