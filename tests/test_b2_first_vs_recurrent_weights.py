"""Critical invariant (spec §8): W_first != W_recurrent (they differ by the
inner D_gamma), for both architectures; equal only if gamma == identity."""

import torch

from b2_common import rel_l2, tiny_setup
from eagle_spinquant.b2_projection import (build_b2_weights_arch_a,
                                           build_b2_weights_arch_b)


def test_weights_differ_arch_b():
    m, sd, R1, gamma, *_ = tiny_setup()
    _, W_first, W_rec, _b = build_b2_weights_arch_b(sd, R1, gamma)
    diff = rel_l2(W_first, W_rec)
    assert diff > 1e-2, f"first==recurrent (diff {diff}) — gamma lost"
    # identity gamma ⇒ identical
    _, Wf_id, Wr_id, _ = build_b2_weights_arch_b(sd, R1, torch.ones_like(gamma))
    assert rel_l2(Wf_id, Wr_id) < 1e-9


def test_weights_differ_arch_a():
    m, sd, R1, gamma, *_ = tiny_setup()
    _, W_first, W_rec, _b = build_b2_weights_arch_a(sd, R1, gamma)
    assert rel_l2(W_first, W_rec) > 1e-2
    # embedding block untouched in arch A first weight
    Dh = W_rec.shape[0]
    assert rel_l2(W_first[:, :Dh], W_rec[:, :Dh]) < 1e-12, \
        "arch-A first fold leaked into the embedding block"


if __name__ == "__main__":
    test_weights_differ_arch_b()
    test_weights_differ_arch_a()
    print("OK")
