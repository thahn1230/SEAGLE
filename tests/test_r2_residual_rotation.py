"""Gate C-class unit tests for the draft-aware R2 parameterization
(GS/R2 study): R2_D = R2_B @ C(B), B = W - W^T, one [128,128] shared
across heads — granularity of the deployed baseline."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from eagle_spinquant.fake_w4a4_draft import baseline_r2  # noqa: E402
from eagle_spinquant.residual_rotation import (  # noqa: E402
    ResidualR2Rotation, cayley, rotation_geometry)


def test_baseline_r2_deterministic():
    a, b = baseline_r2(0), baseline_r2(0)
    assert torch.equal(a, b)
    assert a.dtype == torch.float64 and a.shape == (128, 128)
    I = torch.eye(128, dtype=torch.float64)
    assert float((a.t() @ a - I).norm()) < 1e-12


def test_r2d_init_reproduces_baseline_bitwise():
    """A2 at init must equal A0 exactly: B=0 -> C(B)=I -> R2_D == R2_B."""
    r = ResidualR2Rotation(baseline_r2(0))
    assert torch.equal(r.R64(), r.R2_B64)
    assert torch.equal(r.R(torch.float32), r.R2_B64.float())


def test_r2d_orthogonal_for_any_generator():
    torch.manual_seed(3)
    r = ResidualR2Rotation(baseline_r2(0))
    with torch.no_grad():
        r.W += 0.05 * torch.randn(128, 128)
    assert r.orth_error() < 1e-5          # fp32 Cayley factor
    A = r.A()
    assert float((A + A.t()).abs().max()) == 0.0   # skew by construction


def test_r2d_gradient_flows_to_generator():
    r = ResidualR2Rotation(baseline_r2(0))
    y = r.R(torch.float32).sum()
    y.backward()
    assert r.W.grad is not None
    assert torch.isfinite(r.W.grad).all()
    # sum of an orthogonal-matrix-valued function still moves with B
    with torch.no_grad():
        gnorm = float(r.W.grad.norm())
    assert gnorm > 0


def test_r2d_generator_frob_norm():
    r = ResidualR2Rotation(baseline_r2(0))
    with torch.no_grad():
        r.W[0, 1] = 0.5                    # A = W - W^T has +-0.5 pair
    assert abs(r.generator_frob_norm() - (2 * 0.25) ** 0.5) < 1e-6


def test_r2d_trust_penalty():
    r0 = ResidualR2Rotation(baseline_r2(0), trust="none")
    rw = ResidualR2Rotation(baseline_r2(0), trust="weak")
    with torch.no_grad():
        rw.W[0, 1] += 0.1                  # asymmetric -> A != 0
        r0.W[0, 1] += 0.1
    assert float(r0.penalty()) == 0.0
    assert float(rw.penalty()) > 0.0


def test_r2d_hard_radius_projection():
    r = ResidualR2Rotation(baseline_r2(0), trust="hard", radius=0.1)
    with torch.no_grad():
        r.W += torch.randn(128, 128)
    r.post_step()
    assert float(r.A().norm()) <= 0.1 + 1e-6


def test_r2d_geometry_vs_baseline():
    r = ResidualR2Rotation(baseline_r2(0))
    g0 = rotation_geometry(r.R64(), r.R2_B64)
    assert g0["frob_dist"] == 0.0 and g0["geodesic_dist"] < 1e-6
    with torch.no_grad():
        r.W[:16, 16:32] += 0.02
    g1 = rotation_geometry(r.R64(), r.R2_B64)
    assert g1["frob_dist"] > 0 and g1["orth_error"] < 1e-5


def test_cayley_identity_at_zero():
    A = torch.zeros(64, 64)
    C = cayley(A)
    assert torch.equal(C, torch.eye(64))
