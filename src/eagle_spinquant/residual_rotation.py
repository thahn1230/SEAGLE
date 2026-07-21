"""Rotation parameterizations for R_D (study spec section 8).

  SharedRotation      : R_D = R_T frozen (candidate A / C0).
  FullRotation        : unrestricted orthogonal R_D, initialized anywhere;
                        optimizer steps + QR retraction (validated recipe).
  ResidualRotation    : R_D = R_T @ C(A), A skew, C = Cayley transform
                        C(A) = (I - A/2)^{-1} (I + A/2). A=0 -> R_D = R_T
                        exactly. Orthogonal BY CONSTRUCTION for every A, so
                        no retraction is needed and gradients stay on the
                        manifold.

Trust regions for ResidualRotation (spec 8.3): the Cayley rotation angles
are 2*atan(s_i/2) for the singular values s_i of A, so ||A||_F is a faithful
small-rotation radius. Modes:
  none    : unconstrained
  weak    : + 1e-3 * ||A||_F^2 penalty
  medium  : + 1e-2 * ||A||_F^2 penalty
  hard    : project ||A||_F <= radius after every step

Geometry metrics (used by analyze_rotation_geometry): geodesic distance
(sum of principal-rotation angles of R_T^T R_D, computed via the skew log),
Frobenius distance, max |element| change, orthogonality error.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def cayley(A):
    """C(A) = (I - A/2)^{-1} (I + A/2); orthogonal for skew A."""
    D = A.shape[0]
    I = torch.eye(D, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - A / 2, I + A / 2)


class SharedRotation(nn.Module):
    trainable = False

    def __init__(self, R_T):
        super().__init__()
        self.register_buffer("R_T", R_T.float())

    def R(self):
        return self.R_T

    def penalty(self):
        return torch.zeros((), device=self.R_T.device)

    def post_step(self):
        pass


class FullRotation(nn.Module):
    """Unrestricted orthogonal R_D (candidate E). QR retraction after step."""
    trainable = True

    def __init__(self, R_init):
        super().__init__()
        self.R_D = nn.Parameter(R_init.float().clone())

    def R(self):
        return self.R_D

    def penalty(self):
        return torch.zeros((), device=self.R_D.device)

    @torch.no_grad()
    def post_step(self):
        Q, Rr = torch.linalg.qr(self.R_D.data)
        self.R_D.data = Q * torch.sign(torch.diagonal(Rr))

    def orth_error(self):
        I = torch.eye(self.R_D.shape[0], device=self.R_D.device)
        return float((self.R_D.t() @ self.R_D - I).norm())


class ResidualRotation(nn.Module):
    """R_D = R_T @ C(A) with A = W - W^T (skew by construction)."""
    trainable = True
    PENALTY = dict(none=0.0, weak=1e-3, medium=1e-2)

    def __init__(self, R_T, trust="none", radius=None):
        super().__init__()
        D = R_T.shape[0]
        self.register_buffer("R_T", R_T.float())
        self.W = nn.Parameter(torch.zeros(D, D))
        assert trust in ("none", "weak", "medium", "hard")
        if trust == "hard":
            assert radius is not None and radius > 0
        self.trust = trust
        self.radius = radius

    def A(self):
        return self.W - self.W.t()

    def R(self):
        return self.R_T @ cayley(self.A())

    def penalty(self):
        lam = self.PENALTY.get(self.trust, 0.0)
        if lam == 0.0:
            return torch.zeros((), device=self.W.device)
        return lam * self.A().pow(2).sum()

    @torch.no_grad()
    def post_step(self):
        if self.trust == "hard":
            n = self.A().norm() / 2 ** 0.5   # ||A||_F = sqrt(2)*||W_skew||
            a_norm = float(self.A().norm())
            if a_norm > self.radius:
                self.W.data *= self.radius / a_norm

    def orth_error(self):
        R = self.R()
        I = torch.eye(R.shape[0], device=R.device)
        return float((R.t() @ R - I).norm())


@torch.no_grad()
def rotation_geometry(R_D, R_T):
    """Geometry report for one rotation (spec sections 8.3 / 17)."""
    R_D, R_T = R_D.double(), R_T.double()
    D = R_D.shape[0]
    Q = R_T.t() @ R_D                        # relative rotation
    # principal angles via eigenvalues of Q (orthogonal): e^{±i theta}
    ev = torch.linalg.eigvals(Q)
    theta = torch.atan2(ev.imag, ev.real).abs()
    geo = float(theta.pow(2).sum().sqrt() / 2 ** 0.5)   # each angle twice
    I = torch.eye(D, dtype=torch.float64)
    return dict(
        geodesic_dist=geo,
        frob_dist=float((R_D - R_T).norm()),
        max_elem_change=float((R_D - R_T).abs().max()),
        orth_error=float((R_D.t() @ R_D - I).norm()),
        max_abs_elem=float(R_D.abs().max()),
    )
