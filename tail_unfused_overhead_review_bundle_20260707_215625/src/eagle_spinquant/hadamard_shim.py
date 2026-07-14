"""Hadamard transform availability + a pure-PyTorch reference.

fast_hadamard_transform provides the CUDA kernel SpinQuant uses for online R3/R4.
If it is ever unavailable, `reference_hadamard` gives a functionally-identical
(slower) result via an explicit normalized Hadamard matrix, so the pipeline still
runs correctly (speed numbers are fake-quant anyway; see docs/00 S4). This module
is also used by the math sanity checks to cross-validate the kernel.
"""

from __future__ import annotations

import torch

try:
    import fast_hadamard_transform as _fht
    HAVE_FHT = True
except Exception:
    _fht = None
    HAVE_FHT = False


def _sylvester_hadamard(n: int, device, dtype=torch.float64) -> torch.Tensor:
    """Unnormalized 2^k Hadamard matrix via Sylvester construction."""
    assert n & (n - 1) == 0, f"{n} is not a power of two"
    H = torch.ones((1, 1), device=device, dtype=dtype)
    size = 1
    while size < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        size *= 2
    return H


def reference_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalized Hadamard transform over the last dim (matches fht kernel:
    H x / sqrt(n) with the Sylvester/Walsh ordering). For power-of-two last dim."""
    n = x.shape[-1]
    H = _sylvester_hadamard(n, x.device, torch.float64)
    y = (x.to(torch.float64) @ H.t()) / (n ** 0.5)
    return y.to(x.dtype)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Prefer the CUDA kernel; fall back to the reference. Note the kernel returns
    the *unnormalized* transform (H x); SpinQuant divides by sqrt(n) at call
    sites, so we mirror the kernel's convention here."""
    if HAVE_FHT and x.is_cuda:
        return _fht.hadamard_transform(x)
    # reference is normalized; undo normalization to match kernel convention
    n = x.shape[-1]
    return reference_hadamard(x) * (n ** 0.5)
