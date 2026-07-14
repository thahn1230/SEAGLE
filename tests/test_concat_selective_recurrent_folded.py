"""§12.4: recurrent folded pre-R projection [W_e | W_h·R1] + explicit output R1
must equal the explicit recurrent path; hidden-side R1ᵀ fully absorbed."""

import torch

from b2_common import D_TINY, rel_l2, tiny_setup
from eagle_spinquant.concat_selective_projection import (
    build_concat_selective_weights, concat_selective_explicit)

TOL = 1e-5


def test_recurrent_folded_equals_explicit():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W_first, W_rec, b = build_concat_selective_weights(sd, R1, gamma)
    W, bias = sd["fc.weight"].double(), sd["fc.bias"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    h_d = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    z = torch.cat([e, h_d @ R1], -1)
    folded = (z @ W_rec.double().t() + b.double()) @ R1
    explicit = concat_selective_explicit(z.float(), W.float(), bias.float(),
                                         R1, gamma, "recurrent")
    assert rel_l2(folded, explicit.double()) < TOL
    # equals the pure original computation rotated: (eW_eᵀ + h_d W_hᵀ + b)R1
    ref = (torch.cat([e, h_d], -1) @ W.t() + bias) @ R1
    assert rel_l2(folded, ref) < 1e-9
    assert rel_l2(W_rec.double()[:, :D_TINY], W[:, :D_TINY]) < 1e-12
    # W_first != W_rec (differ by inner D_gamma on the hidden block only)
    assert rel_l2(W_first.double()[:, D_TINY:], W_rec.double()[:, D_TINY:]) > 1e-2
    # negative control: orig_PL_recurrent (no hidden absorption) must differ
    _, W_rec_bad, _b = build_concat_selective_weights(sd, R1, gamma,
                                                      nc="orig_PL_recurrent")
    bad = (z @ W_rec_bad.double().t() + b.double()) @ R1
    assert rel_l2(bad, ref) > 1e-2, "orig-PL-on-rotated-hidden control passed?!"


if __name__ == "__main__":
    test_recurrent_folded_equals_explicit()
    print("OK")
