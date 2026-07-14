"""Test 1 (spec §13): first-step algebra.

Arch B :  (n_t·R1) @ W_feature_firstᵀ @ R1ᵀ == (n_t·γ) @ W_featureᵀ
          with W_feature_first = R1ᵀ·W_f·D_γ·R1  (via convert_draft_state fc_ext)
Arch A :  a_t @ W_feature_first_Aᵀ            == (n_t·γ) @ W_featureᵀ
          with W_feature_first_A = W_f·D_γ·R1  (via fold_matrix)
Full-fc versions include the embedding block + bias.
"""

import torch

from b2_common import (D_TINY, rand_orthogonal, rand_gamma, rms_normalize,
                       rel_l2, tiny_setup)
from eagle_spinquant import rotation_aware as ra
from eagle_spinquant.b2_projection import (build_b2_weights_arch_a,
                                           build_b2_weights_arch_b)

TOL = 1e-10


def test_first_feature_block_arch_b():
    Dh = 32
    R1 = rand_orthogonal(Dh, 5)
    gamma = rand_gamma(Dh, 6)
    W_f = torch.randn(Dh, Dh, dtype=torch.float64)
    n = rms_normalize(torch.randn(4, Dh, dtype=torch.float64))
    original = (n * gamma) @ W_f.t()
    W_first = R1.t() @ W_f @ torch.diag(gamma) @ R1
    rotated = ((n @ R1) @ W_first.t()) @ R1.t()
    assert rel_l2(rotated, original) < TOL


def test_first_full_fc_arch_b_matches_fc_ext():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    _, W_first, _W_rec, b_rot = build_b2_weights_arch_b(sd, R1, gamma)
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    z_orig = torch.cat([e, h_t], -1)
    y = z_orig @ W.t() + b                                     # original fc
    z_rot = torch.cat([e @ R1, a_t], -1)                       # rotated inputs
    y_R = z_rot @ W_first.double().t() + b_rot.double()
    assert rel_l2(y_R @ R1.t(), y) < TOL, "fc_ext != R1ᵀ[W_e R1 | W_f D_γ R1]"


def test_first_full_fc_arch_a():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    _, W_first, _W_rec, b = build_b2_weights_arch_a(sd, R1, gamma)
    W = sd["fc.weight"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    y = torch.cat([e, h_t], -1) @ W.t() + sd["fc.bias"].double()
    y_A = torch.cat([e, a_t], -1) @ W_first.double().t() + b.double()
    assert rel_l2(y_A, y) < TOL, "arch-A first fold wrong"


if __name__ == "__main__":
    test_first_feature_block_arch_b()
    test_first_full_fc_arch_b_matches_fc_ext()
    test_first_full_fc_arch_a()
    print("OK")
