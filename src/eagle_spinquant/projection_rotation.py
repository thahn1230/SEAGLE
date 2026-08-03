"""Projection-local structured orthogonal rotations for R-EP3-P.

Q is represented implicitly as   Q = P_out . H_b . D_sign . P_in
(right-multiplication convention: apply(x) returns x Q for row-vector
x), where H_b is a normalized blockwise Walsh-Hadamard transform,
D_sign a +-1 diagonal, and P_in / P_out deterministic permutations.
All pieces are orthogonal, so Q is orthogonal by construction.

Orientation contract (audited, test_rotated_ep3p_weight_orientation):
  storage  W_pt [out_features, in_features],  y = x @ W_pt.T (+ bias)
  transform T = S Q (SR) or Q S (RS), x' = x T
  requirement x T W'_pt.T = x W_pt.T  =>  W'_pt = W_pt T^{-T}
  SR: T^{-T} = (Q^T S^{-1})^T = S^{-1} Q  => W'_pt = (W_pt S^{-1}) Q
  RS: T^{-T} = (S^{-1} Q^T)^T = Q S^{-1}  => W'_pt = (W_pt Q) S^{-1}
Both cases reduce to "scale input columns by 1/m where S applies, and
right-multiply rows by Q with the same apply() used for activations";
only the order differs (RS scales in the ROTATED basis).

This Q is NOT the target SpinQuant rotation R_T, NOT the shared
interface rotation R1, and NOT the residual draft rotation R_D: it is
an input preconditioner of the feature-fusion projection applied after
the validated projection input has been assembled.
"""
import hashlib

import torch

D = 4096


def fwht(x, block):
    """Normalized blockwise Walsh-Hadamard on the last dim.
    x last dim must be divisible by block; block power of two."""
    shp = x.shape
    n = shp[-1]
    assert n % block == 0 and (block & (block - 1)) == 0
    y = x.reshape(-1, n // block, block).clone()
    h = 1
    while h < block:
        y = y.reshape(-1, n // block, block // (2 * h), 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
    y = y.reshape(-1, n // block, block) / (block ** 0.5)
    return y.reshape(shp)


def interleave_perm(n, chunk=1, device="cpu"):
    """[e0..e_{D-1}, h0..h_{D-1}] -> chunkwise interleave so every
    contiguous window of 2*chunk holds both branches."""
    half = n // 2
    e = torch.arange(half, device=device).reshape(-1, chunk)
    h = (torch.arange(half, device=device) + half).reshape(-1, chunk)
    return torch.stack([e, h], dim=1).reshape(-1)


class StructuredRotation:
    """spec keys:
      family : identity|e_only|h_only|dual|full|cross
      block  : Hadamard block size (power of two)
      seed   : int (sign diagonal + optional extra permutation)
      seed_h : optional independent seed for the h half (dual)
      interleave_chunk : cross family chunk size (default 1)
    """

    def __init__(self, spec, n=2 * D, device="cpu"):
        self.spec = dict(spec)
        self.n = n
        self.family = spec.get("family", "identity")
        self.block = int(spec.get("block", 0) or 0)
        self.seed = int(spec.get("seed", 0))
        dev = device
        g = torch.Generator().manual_seed(self.seed)
        half = n // 2
        if self.family == "identity":
            self.sign = None
            self.perm_in = None
            self.perm_out = None
            return
        if self.family in ("e_only", "h_only", "dual"):
            se = (torch.randint(0, 2, (half,), generator=g) * 2
                  - 1).float()
            g2 = torch.Generator().manual_seed(
                int(spec.get("seed_h", self.seed + 7919)))
            sh = (torch.randint(0, 2, (half,), generator=g2) * 2
                  - 1).float()
            if self.family == "e_only":
                sh = torch.ones(half)
            if self.family == "h_only":
                se = torch.ones(half)
            self.sign = torch.cat([se, sh]).to(dev)
            self.perm_in = None
            self.perm_out = None
        elif self.family == "full":
            self.sign = ((torch.randint(0, 2, (n,), generator=g) * 2
                          - 1).float()).to(dev)
            self.perm_in = None
            self.perm_out = None
        elif self.family == "cross":
            self.sign = ((torch.randint(0, 2, (n,), generator=g) * 2
                          - 1).float()).to(dev)
            self.perm_in = interleave_perm(
                n, int(spec.get("interleave_chunk", 1))).to(dev)
            self.perm_out = None
        else:
            raise ValueError(self.family)

    # -- which coordinates the blockwise Hadamard acts on ------------
    def _apply_core(self, y):
        half = self.n // 2
        if self.family == "e_only":
            y = torch.cat([fwht(y[..., :half], self.block),
                           y[..., half:]], -1)
        elif self.family == "h_only":
            y = torch.cat([y[..., :half],
                           fwht(y[..., half:], self.block)], -1)
        elif self.family in ("dual", "full", "cross"):
            y = fwht(y, self.block)
        return y

    def apply(self, x):
        """returns x Q (right multiplication, rows are vectors)."""
        if self.family == "identity":
            return x
        y = x
        if self.perm_in is not None:
            y = y[..., self.perm_in]
        y = y * self.sign.to(y.dtype).to(y.device)
        y = self._apply_core(y)
        if self.perm_out is not None:
            y = y[..., self.perm_out]
        return y

    def apply_T(self, x):
        """returns x Q^T (inverse). Q = Pout H S Pin as operators on
        row-vectors applied left-to-right: y = Pin -> sign -> H ->
        Pout. Inverse: un-Pout -> H (self-inverse) -> sign -> un-Pin."""
        if self.family == "identity":
            return x
        y = x
        if self.perm_out is not None:
            inv = torch.argsort(self.perm_out)
            y = y[..., inv]
        y = self._apply_core(y)          # H is its own inverse
        y = y * self.sign.to(y.dtype).to(y.device)
        if self.perm_in is not None:
            inv = torch.argsort(self.perm_in.to(y.device))
            y = y[..., inv]
        return y

    def to_dense(self, dtype=torch.float64):
        eye = torch.eye(self.n, dtype=dtype)
        return self.apply(eye)           # rows: e_i Q

    def meta(self):
        m = dict(self.spec, n=self.n)
        if self.sign is not None:
            m["sign_hash"] = hashlib.sha256(
                self.sign.to(torch.int8).numpy().tobytes()) \
                .hexdigest()[:16]
        if self.perm_in is not None:
            m["perm_in_hash"] = hashlib.sha256(
                self.perm_in.cpu().numpy().tobytes()).hexdigest()[:16]
        return m


def scale_vec(n, m_val, device="cpu", dtype=torch.float64):
    s = torch.ones(n, device=device, dtype=dtype)
    s[: n // 2] = m_val
    return s


def transform_xw(X, W_pt, m_val, rot, order="SR", bias=None):
    """Return (X', W'_pt) implementing x T, W_pt T^{-T}.
    X [N, n] raw (UNSCALED) input; W_pt [out, n] original fold."""
    s = scale_vec(X.shape[-1], m_val, X.device, X.dtype)
    if order == "SR":
        Xt = rot.apply(X * s)
        Wt = rot.apply(W_pt / s)
    else:                                  # RS: T = Q S
        Xt = rot.apply(X) * s
        Wt = rot.apply(W_pt) / s
    return Xt, Wt
