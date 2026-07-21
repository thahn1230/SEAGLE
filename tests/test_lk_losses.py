"""Gate C: LK loss mathematics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from eagle_spinquant import lk_losses as L


def _logits(B=5, V=97, seed=0):
    g = torch.Generator().manual_seed(seed)
    zT = torch.randn(B, V, generator=g) * 3
    zD = (torch.randn(B, V, generator=g) * 3).requires_grad_(True)
    return zT, zD


def test_alpha_equals_one_minus_tv():
    zT, zD = _logits()
    a = L.overlap_alpha(zT, zD)
    t = L.tv(zT, zD)
    assert torch.allclose(a, 1 - t, atol=1e-6)


def test_alpha_bounds_and_selfmatch():
    zT, zD = _logits()
    a = L.overlap_alpha(zT, zD)
    assert (a > 0).all() and (a < 1).all()
    a_self = L.overlap_alpha(zT, zT)
    assert torch.allclose(a_self, torch.ones_like(a_self), atol=1e-6)


def test_neg_log_alpha_gradient_nonzero_finite():
    zT, zD = _logits()
    loss = L.neg_log_alpha(zT, zD).mean()
    loss.backward()
    g = zD.grad
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_kl_full_matches_torch_reference():
    zT, zD = _logits()
    ref = torch.nn.functional.kl_div(
        torch.log_softmax(zD, -1), torch.softmax(zT, -1),
        reduction="none").sum(-1)
    assert torch.allclose(L.kl_full(zT, zD), ref, atol=1e-5)


def test_objectives_are_distinct():
    zT, zD = _logits()
    kl = L.kl_full(zT, zD).mean()
    tv_ = L.tv(zT, zD).mean()
    nla = L.neg_log_alpha(zT, zD).mean()
    hyb, lam, _ = L.hybrid_lk(zT, zD, eta=3.0)
    vals = torch.stack([kl, tv_, nla, hyb.mean()])
    assert len(torch.unique(torch.round(vals * 1e6))) == 4


def test_adaptive_lambda_stop_gradient():
    zT, zD = _logits()
    a = L.overlap_alpha(zT, zD)
    lam = L.adaptive_lambda(a, eta=3.0)
    assert not lam.requires_grad          # stopgrad: no graph into lambda
    # but the hybrid loss still differentiates through KL and TV terms
    loss, lam2, _ = L.hybrid_lk(zT, zD, eta=3.0)
    loss.mean().backward()
    assert torch.isfinite(zD.grad).all() and float(zD.grad.abs().sum()) > 0


def test_per_depth_lambda_independent():
    zT, zD = _logits(B=8)
    lam_lo = L.adaptive_lambda(torch.full((8,), 0.1), eta=3.0)
    lam_hi = L.adaptive_lambda(torch.full((8,), 0.9), eta=3.0)
    assert float(lam_lo) > float(lam_hi)   # low alpha -> more KL weight
    assert abs(float(lam_lo) - torch.exp(torch.tensor(-0.3))) < 1e-5


def test_expected_tau_formula():
    a = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    et = L.expected_tau(a)
    # 1 + .5 + .25 + .125 + .0625 = 1.9375
    assert torch.allclose(et, torch.tensor([1.9375]))
    a1 = torch.ones(1, 4)
    assert torch.allclose(L.expected_tau(a1), torch.tensor([5.0]))
    assert torch.allclose(L.expected_tau_loss(a1),
                          torch.tensor([0.0]), atol=1e-5)


def test_expected_tau_early_rejection_dominates():
    early = torch.tensor([[0.1, 0.9, 0.9, 0.9]])
    late = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
    assert L.expected_tau(late) > L.expected_tau(early)


def test_topk_vs_full_kl_differ():
    zT, zD = _logits(V=500)
    full = L.kl_full(zT, zD)
    tk = L.kl_topk(zT, zD, k=64)
    assert not torch.allclose(full, tk, atol=1e-3)


def test_greedy_ce_and_margin():
    zT, zD = _logits()
    ce = L.greedy_ce(zT, zD)
    assert ce.shape == (5,) and torch.isfinite(ce).all()
    m = L.top1_margin(zT, zD)
    assert (m >= 0).all()


def test_depth_weights_unnormalized():
    w = L.depth_weights(4, gamma=0.8)
    assert torch.allclose(w, torch.tensor([1.0, 0.8, 0.64, 0.512]))
