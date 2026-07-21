"""Correct speculative rejection sampling (study spec 16.M1, Gate I).

Standard speculative sampling identity: draft proposes x ~ q; accept with
probability min(1, p(x)/q(x)); on rejection resample from the residual
normalize(max(p - q, 0)). The output token is then EXACTLY distributed as
p, and P(accept) = sum_v min(p_v, q_v) = alpha(p, q).
Used by eval_eagle_lk_chain (M1) and verified analytically in
tests/test_rejection_sampling.py.
"""
from __future__ import annotations

import torch


def accept_prob(p, q, x):
    """min(1, p(x)/q(x)) for token x."""
    return (p[x] / q[x].clamp_min(1e-20)).clamp(max=1.0)


def residual_dist(p, q):
    """normalize(max(p - q, 0)); falls back to p if the residual is
    numerically empty (q >= p everywhere can only happen at q == p)."""
    r = (p - q).clamp_min(0)
    s = r.sum()
    if float(s) <= 0:
        return p
    return r / s


def spec_step(p, q, x, gen=None, device="cpu"):
    """One accept/reject decision. Returns (accepted: bool,
    correction_token or None)."""
    u = torch.rand((), device=device, generator=gen)
    if u < accept_prob(p, q, x):
        return True, None
    r = residual_dist(p, q)
    corr = int(torch.multinomial(r, 1, generator=gen))
    return False, corr
