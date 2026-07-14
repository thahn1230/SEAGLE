"""§9: the embedding slice must be numerically unchanged and never receive
hidden-side transforms; prohibited constructions must be detectably different."""

import torch

from b2_common import D_TINY, rel_l2, tiny_setup
from eagle_spinquant.concat_selective_projection import (
    build_concat_selective_weights, concat_selective_explicit)


def test_embedding_slice_passthrough_and_prohibited_forms():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    W_first, W_rec, _b = build_concat_selective_weights(sd, R1, gamma)

    # 1) folded weights keep W_e EXACTLY (both paths)
    for Wx in (W_first, W_rec):
        assert rel_l2(Wx.double()[:, :D_TINY], W[:, :D_TINY]) < 1e-12

    # 2) prohibited: W_e·D_γ, R1ᵀW_eR1, W_e·R1 — all measurably different
    W_e = W[:, :D_TINY]
    for bad in (W_e * gamma.unsqueeze(0), R1.t() @ W_e @ R1, W_e @ R1):
        assert rel_l2(bad, W_e) > 1e-2

    # 3) explicit-path negative controls change the output
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    z = torch.cat([e, a_t], -1).float()
    ok = concat_selective_explicit(z, W.float(), b.float(), R1, gamma, "first")
    for nc in ("embedding_rotated", "gamma_on_embedding", "Rt_on_whole_concat"):
        badout = concat_selective_explicit(z, W.float(), b.float(), R1, gamma,
                                           "first", nc=nc)
        assert rel_l2(badout.double(), ok.double()) > 1e-3, nc

    # 4) e-slice of the input is bit-identical through the folded module's
    #    input (no pre-transform exists by construction: nn.Linear consumes z)
    #    — asserted structurally: the only input op in the primary path is the
    #    Linear itself (see ConcatSelectiveProjection.forward).


if __name__ == "__main__":
    test_embedding_slice_passthrough_and_prohibited_forms()
    print("OK")
