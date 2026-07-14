"""Test 2 (spec §13): recurrent algebra + embedding branch (Test 4).

    (h_d·R1) @ W_feature_recurrentᵀ @ R1ᵀ == h_d @ W_featureᵀ
    (e·R1)   @ W_embedding_rotᵀ   @ R1ᵀ == e @ W_embeddingᵀ
NO gamma anywhere on the recurrent path.
"""

import torch

from b2_common import D_TINY, rand_orthogonal, rel_l2, tiny_setup
from eagle_spinquant.b2_projection import build_b2_weights_arch_b

TOL = 1e-10


def test_recurrent_feature_block():
    Dh = 32
    R1 = rand_orthogonal(Dh, 7)
    W_f = torch.randn(Dh, Dh, dtype=torch.float64)
    h_d = torch.randn(4, Dh, dtype=torch.float64)
    W_rec = R1.t() @ W_f @ R1
    assert rel_l2(((h_d @ R1) @ W_rec.t()) @ R1.t(), h_d @ W_f.t()) < TOL


def test_recurrent_full_fc_and_embedding_branch():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    _, _W_first, W_rec, b_rot = build_b2_weights_arch_b(sd, R1, gamma)
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    h_d = torch.randn(1, 3, D_TINY, dtype=torch.float64)   # any recurrent feature
    y = torch.cat([e, h_d], -1) @ W.t() + b
    y_R = torch.cat([e @ R1, h_d @ R1], -1) @ W_rec.double().t() + b_rot.double()
    assert rel_l2(y_R @ R1.t(), y) < TOL

    # embedding-only sensitivity (h zeroed) — checks W_embedding_rot block alone
    y_e = torch.cat([e, torch.zeros_like(h_d)], -1) @ W.t()
    y_eR = torch.cat([e @ R1, torch.zeros_like(h_d)], -1) @ W_rec.double().t()
    assert rel_l2(y_eR @ R1.t(), y_e) < TOL, "embedding block fold wrong"


if __name__ == "__main__":
    test_recurrent_feature_block()
    test_recurrent_full_fc_and_embedding_branch()
    print("OK")
