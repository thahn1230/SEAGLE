"""§12.2: first folded pre-R projection [W_e | W_h·D_γ·R1] + explicit output R1
must equal the explicit first path (and the fp64 closed form)."""

import torch

from b2_common import D_TINY, rel_l2, tiny_setup
from eagle_spinquant.concat_selective_projection import (
    build_concat_selective_weights, concat_selective_explicit)

TOL = 1e-5


def test_first_folded_equals_explicit():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W_first, W_rec, b = build_concat_selective_weights(sd, R1, gamma)
    W, bias = sd["fc.weight"].double(), sd["fc.bias"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    z = torch.cat([e, a_t], -1)
    folded = (z @ W_first.double().t() + b.double()) @ R1
    explicit = concat_selective_explicit(z.float(), W.float(), bias.float(),
                                         R1, gamma, "first")
    assert rel_l2(folded, explicit.double()) < TOL
    # embedding block untouched in the folded weight
    assert rel_l2(W_first.double()[:, :D_TINY], W[:, :D_TINY]) < 1e-12, \
        "W_e was transformed — forbidden"
    # bias is the ORIGINAL b (pre-R), not b·R1
    assert rel_l2(b.double(), bias) < 1e-12


if __name__ == "__main__":
    test_first_folded_equals_explicit()
    print("OK")
