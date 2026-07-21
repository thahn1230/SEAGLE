"""Gate C/G: residual Cayley rotation + geometry."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from eagle_spinquant.residual_rotation import (
    ResidualRotation, FullRotation, SharedRotation, cayley,
    rotation_geometry)


def _rt(D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(D, D, generator=g))[0]


def test_cayley_orthogonal_for_skew():
    g = torch.Generator().manual_seed(1)
    W = torch.randn(16, 16, generator=g) * 0.3
    A = W - W.t()
    C = cayley(A)
    I = torch.eye(16)
    assert float((C.t() @ C - I).norm()) < 1e-5


def test_residual_zero_A_is_exactly_RT():
    R_T = _rt()
    m = ResidualRotation(R_T)
    assert torch.allclose(m.R(), R_T.float(), atol=1e-7)
    assert m.orth_error() < 1e-5


def test_residual_gradients_flow():
    R_T = _rt()
    m = ResidualRotation(R_T)
    x = torch.randn(4, 16)
    (x @ m.R()).pow(2).sum().backward()
    assert m.W.grad is not None and float(m.W.grad.abs().sum()) > 0


def test_residual_stays_orthogonal_after_updates():
    R_T = _rt()
    m = ResidualRotation(R_T)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(5):
        loss = (torch.randn(4, 16) @ m.R()).pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); m.post_step()
    assert m.orth_error() < 1e-4      # by construction, no retraction


def test_hard_trust_region_projects():
    R_T = _rt()
    m = ResidualRotation(R_T, trust="hard", radius=0.01)
    with torch.no_grad():
        m.W += 1.0
    m.post_step()
    assert float(m.A().norm()) <= 0.0101


def test_penalty_modes():
    R_T = _rt()
    with torch.no_grad():
        none = ResidualRotation(R_T, trust="none")
        weak = ResidualRotation(R_T, trust="weak")
        weak.W[0, 1] += 0.1                     # asymmetric: changes A
        assert float(none.penalty()) == 0.0
        assert float(weak.penalty()) > 0.0


def test_full_rotation_qr_retraction():
    m = FullRotation(_rt())
    with torch.no_grad():
        m.R_D += torch.randn_like(m.R_D) * 0.05
    m.post_step()
    assert m.orth_error() < 1e-4


def test_shared_rotation_frozen():
    m = SharedRotation(_rt())
    assert not any(p.requires_grad for p in m.parameters())


def test_geometry_report():
    R_T = _rt()
    m = ResidualRotation(R_T)
    with torch.no_grad():
        m.W[0, 1] += 0.01                       # asymmetric: changes A
    g = rotation_geometry(m.R(), R_T)
    assert g["orth_error"] < 1e-5
    assert 0 < g["geodesic_dist"] < 1.0
    assert g["frob_dist"] > 0
    g0 = rotation_geometry(R_T, R_T)
    assert g0["geodesic_dist"] < 1e-5 and g0["frob_dist"] < 1e-6
