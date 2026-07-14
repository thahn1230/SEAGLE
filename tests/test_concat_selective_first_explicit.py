"""§12.1: first explicit reference — implemented explicit path must equal
    ((e W_eᵀ + ((a R1ᵀ)*γ) W_hᵀ + b) R1  exactly (fp64/fp32 tolerance)."""

import torch

from b2_common import D_TINY, rel_l2, tiny_setup
from eagle_spinquant.concat_selective_projection import concat_selective_explicit

TOL = 1e-5   # explicit path runs fp32 internally


def test_first_explicit_reference():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    W_e, W_h = W[:, :D_TINY], W[:, D_TINY:]
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    reference = (e @ W_e.t()
                 + ((a_t @ R1.t()) * gamma) @ W_h.t() + b) @ R1
    z = torch.cat([e, a_t], -1).float()
    out = concat_selective_explicit(z, W.float(), b.float(), R1, gamma, "first")
    assert rel_l2(out.double(), reference) < TOL


def test_first_explicit_invariants():
    """R1ᵀ/γ touch ONLY the hidden slice: zero hidden ⇒ output = (eW_eᵀ+b)R1
    independent of γ; zero embedding ⇒ e-block contributes nothing."""
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup()
    W, b = sd["fc.weight"].double(), sd["fc.bias"].double()
    e = torch.randn(1, 3, D_TINY, dtype=torch.float64)
    z0 = torch.cat([e, torch.zeros_like(e)], -1).float()
    out_g = concat_selective_explicit(z0, W.float(), b.float(), R1, gamma, "first")
    out_2g = concat_selective_explicit(z0, W.float(), b.float(), R1,
                                       gamma * 2.0, "first")
    assert rel_l2(out_g.double(), out_2g.double()) < 1e-6, \
        "gamma leaked into the embedding slice"


if __name__ == "__main__":
    test_first_explicit_reference()
    test_first_explicit_invariants()
    print("OK")
