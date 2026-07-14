"""§12.3: recurrent explicit reference — ((e W_eᵀ + (r_d R1ᵀ) W_hᵀ + b) R1;
NO gamma anywhere on the recurrent path."""

import torch

from b2_common import D_TINY, rel_l2, tiny_setup
from eagle_spinquant.concat_selective_projection import concat_selective_explicit

TOL = 1e-5


def test_recurrent_explicit_reference():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    W_e, W_h = W[:, :D_TINY], W[:, D_TINY:]
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    h_d = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    r_d = h_d @ R1
    reference = (e @ W_e.t() + (r_d @ R1.t()) @ W_h.t() + b) @ R1
    z = torch.cat([e, r_d], -1).float()
    out = concat_selective_explicit(z, W.float(), b.float(), R1, gamma,
                                    "recurrent")
    assert rel_l2(out.double(), reference) < TOL
    # gamma must NOT appear: doubling gamma changes nothing on recurrent
    out2 = concat_selective_explicit(z, W.float(), b.float(), R1, gamma * 3.0,
                                     "recurrent")
    assert rel_l2(out.double(), out2.double()) < 1e-7


if __name__ == "__main__":
    test_recurrent_explicit_reference()
    print("OK")
