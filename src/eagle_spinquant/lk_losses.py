"""LK acceptance losses (study spec section 5) — full-vocabulary, from logits.

Conventions
-----------
All losses take TARGET logits ``zT`` (teacher, constant — always detached
internally) and DRAFT logits ``zD`` (differentiable) with shape
``(..., V)`` over the FULL vocabulary, temperature 1 unless stated.

  p = softmax(zT)          deployed-target distribution
  q = softmax(zD)          quantized-draft distribution

  alpha(p, q) = sum_v min(p_v, q_v)        speculative acceptance rate
  TV(p, q)    = 0.5 * sum_v |p_v - q_v|    total variation
  alpha == 1 - TV                          (verified in tests, Gate C)

Everything reduces over the vocab dim only; callers apply batch/position/
depth weighting. ``eps`` guards the log at alpha -> 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def _pq(zT, zD):
    p = F.softmax(zT.detach().float(), dim=-1)
    q = F.softmax(zD.float(), dim=-1)
    return p, q


def overlap_alpha(zT, zD):
    """alpha = sum_v min(p, q), shape (...,). Differentiable in zD
    (min is differentiable a.e.; subgradient at ties)."""
    p, q = _pq(zT, zD)
    return torch.minimum(p, q).sum(-1)


def tv(zT, zD):
    """Total variation distance, shape (...,)."""
    p, q = _pq(zT, zD)
    return 0.5 * (p - q).abs().sum(-1)


def kl_full(zT, zD):
    """Forward KL(p || q) over the full vocabulary, shape (...,).
    Computed in log space: sum p * (log p - log q)."""
    p = F.softmax(zT.detach().float(), dim=-1)
    logp = F.log_softmax(zT.detach().float(), dim=-1)
    logq = F.log_softmax(zD.float(), dim=-1)
    return (p * (logp - logq)).sum(-1)


def kl_topk(zT, zD, k=64):
    """Previous-study proxy: KL restricted to teacher top-k, renormalized.
    Kept ONLY for the top-64-vs-full ablation."""
    zT = zT.detach().float()
    tv_, ti = zT.topk(k, dim=-1)
    pT = F.softmax(tv_, dim=-1)
    zDk = torch.gather(zD.float(), -1, ti)
    logq = F.log_softmax(zDk, dim=-1)
    return (pT * (pT.clamp_min(1e-9).log() - logq)).sum(-1)


def neg_log_alpha(zT, zD, eps=EPS):
    """L_LK_alpha = -log(alpha + eps), shape (...,)."""
    return -(overlap_alpha(zT, zD) + eps).log()


def adaptive_lambda(alpha_k, eta):
    """lambda_k = exp(-eta * stopgrad(mean(alpha_k))) — scalar per depth.
    alpha_k: tensor of alphas for one depth (batch/positions)."""
    return torch.exp(-eta * alpha_k.detach().mean())


def hybrid_lk(zT, zD, eta=3.0, fixed_lambda=None, eps=EPS):
    """Adaptive hybrid LK for ONE depth: lam*KL + (1-lam)*TV, where
    lam = exp(-eta*stopgrad(mean alpha)) or a fixed constant.
    Returns (loss_per_elem, lam_scalar, alpha_per_elem)."""
    a = overlap_alpha(zT, zD)
    lam = (torch.as_tensor(float(fixed_lambda), device=zD.device)
           if fixed_lambda is not None else adaptive_lambda(a, eta))
    loss = lam * kl_full(zT, zD) + (1.0 - lam) * tv(zT, zD)
    return loss, lam, a


def expected_tau(alphas):
    """alphas: (..., K) per-depth acceptance probs along a CHAIN.
    expected_tau = 1 + sum_{k=1..K} prod_{j<=k} alpha_j, shape (...,)."""
    cum = torch.cumprod(alphas, dim=-1)
    return 1.0 + cum.sum(-1)


def expected_tau_loss(alphas, eps=EPS):
    """L = -log(expected_tau / (K+1)); alphas (..., K)."""
    K = alphas.shape[-1]
    et = expected_tau(alphas)
    return -((et / (K + 1)).clamp_min(eps)).log()


def teacher_token_logp(zD, tok):
    """log q(teacher token) for one depth: zD (..., V), tok (...,).
    The LRGF-validated ACC surrogate consumes these along a chain."""
    return torch.log_softmax(zD.float(), dim=-1) \
        .gather(-1, tok.long().unsqueeze(-1)).squeeze(-1)


def prefix_survival(logps):
    """Soft prefix survival (LRGF ACC surrogate): logps (..., K) log-probs
    of the teacher-forced greedy token per depth. Returns
    sum_{k=1..K} prod_{j<=k} q_j, shape (...,) — the expected accepted
    length under greedy verification, soft version. Clamp at 0 keeps each
    prefix probability <= 1."""
    return torch.exp(torch.cumsum(logps, dim=-1).clamp(max=0.0)).sum(-1)


def greedy_ce(zT, zD):
    """Auxiliary: -log q(argmax p), shape (...,)."""
    y = zT.detach().argmax(-1)
    return F.cross_entropy(
        zD.float().reshape(-1, zD.shape[-1]), y.reshape(-1),
        reduction="none").reshape(y.shape)


def top1_margin(zT, zD, margin=1.0):
    """Auxiliary: hinge on the target-top-1 logit margin."""
    y = zT.detach().argmax(-1)
    zy = zD.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    zother = zD.scatter(-1, y.unsqueeze(-1),
                        torch.finfo(zD.dtype).min).amax(-1)
    return F.relu(margin - zy + zother)


def depth_weights(K, gamma=0.8, normalize=False, device="cpu"):
    """w_k = gamma^(k-1), k=1..K. NOT normalized by default (spec section 6:
    do not silently normalize away the intended relative weighting)."""
    w = torch.tensor([gamma ** k for k in range(K)], device=device)
    return w / w.sum() if normalize else w
