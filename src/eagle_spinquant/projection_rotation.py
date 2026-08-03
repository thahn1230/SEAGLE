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


class DeployedLearnedRotation:
    """Deployment-side reconstruction of a trained LearnedRotation
    checkpoint (no dense 8192^2 matrix; params stay in their compact
    parameterization). spec: {"learned_ckpt": path, "which": "rot_f"|
    "rot_r"}."""

    def __init__(self, spec, n=2 * D, device="cpu"):
        self.spec = dict(spec)
        self.n = n
        ck = torch.load(spec["learned_ckpt"], map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        sd = ck[spec.get("which", "rot_f")] or ck["rot_f"]
        self.param = cfg["param"]
        self.block = int(cfg.get("block", 32))
        self.perm = interleave_perm(n)
        self.inv_perm = torch.argsort(self.perm)
        self.fixed = (StructuredRotation(
            dict(family="cross", block=8192, seed=12,
                 interleave_chunk=1), n=n)
            if cfg.get("fixed_init") else None)
        if self.param == "cayley":
            a = sd["a"].float()
            A = a - a.transpose(1, 2)
            eye = torch.eye(self.block).expand_as(A)
            self.Q = torch.linalg.solve(eye + A, eye - A)
        elif self.param == "givens":
            self.theta = sd["theta"].float()
        elif self.param == "householder":
            self.v = sd["v"].float()
        else:
            raise ValueError(self.param)

    def apply(self, x):
        y = x
        if self.fixed is not None:
            y = self.fixed.apply(y)
        y = y[..., self.perm.to(y.device)]
        if self.param == "cayley":
            Q = self.Q.to(y.device).to(
                torch.float32 if y.dtype != torch.float64
                else torch.float64)
            z = y.reshape(*y.shape[:-1], self.n // self.block,
                          self.block)
            z = torch.einsum("...bi,bij->...bj",
                             z.to(Q.dtype), Q).to(y.dtype)
            y = z.reshape(*y.shape)
        elif self.param == "givens":
            c = torch.cos(self.theta).to(y.device).to(y.dtype)
            s = torch.sin(self.theta).to(y.device).to(y.dtype)
            z = y.reshape(*y.shape[:-1], self.n // 2, 2)
            e, h = z[..., 0], z[..., 1]
            y = torch.stack([e * c + h * s, -e * s + h * c],
                            dim=-1).reshape(*y.shape)
        else:
            z = y.float()
            for i in range(self.v.shape[0]):
                v = self.v[i].to(y.device)
                v = v / v.norm().clamp_min(1e-8)
                z = z - 2.0 * torch.outer(
                    z.reshape(-1, self.n) @ v, v).reshape(z.shape)
            y = z.to(y.dtype)
        return y[..., self.inv_perm.to(y.device)]

    def meta(self):
        return dict(self.spec, param=self.param, block=self.block)


def build_rotation(spec, n=2 * D, device="cpu"):
    """Factory: fixed structured spec dict OR learned ckpt spec."""
    if spec.get("learned_ckpt"):
        return DeployedLearnedRotation(spec, n=n, device=device)
    return StructuredRotation(spec, n=n, device=device)


class LearnedRotation(torch.nn.Module):
    """Orthogonal-by-construction learned rotation on the 2D concat.
    param:
      cayley      blockwise Cayley over interleaved layout
      givens      one angle per interleaved (e_i, h_i) pair (block 2)
      householder K reflections over the full 8192 vector
    An optional FIXED structured cross rotation (spec dict) composes
    first: Q = Q_fixed Q_learned.
    """

    def __init__(self, param="cayley", block=32, K=8, fixed_spec=None,
                 n=2 * D, seed=0, device="cuda:0"):
        super().__init__()
        self.param, self.block, self.n = param, block, n
        torch.manual_seed(seed)
        self.fixed = (StructuredRotation(fixed_spec, n=n,
                                         device=device)
                      if fixed_spec else None)
        self.perm = interleave_perm(n).to(device)
        self.inv_perm = torch.argsort(self.perm)
        if param == "cayley":
            nb = n // block
            self.a = torch.nn.Parameter(
                torch.zeros(nb, block, block))
        elif param == "givens":
            self.theta = torch.nn.Parameter(torch.zeros(n // 2))
        elif param == "householder":
            self.v = torch.nn.Parameter(
                torch.randn(K, n) * 0.02)
        else:
            raise ValueError(param)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _learned(self, y):
        if self.param == "cayley":
            A = self.a - self.a.transpose(1, 2)
            eye = torch.eye(self.block, device=y.device,
                            dtype=torch.float32).expand_as(A)
            Q = torch.linalg.solve(eye + A, eye - A).to(y.dtype)
            z = y.reshape(*y.shape[:-1], self.n // self.block,
                          self.block)
            z = torch.einsum("...bi,bij->...bj", z, Q)
            return z.reshape(*y.shape)
        if self.param == "givens":
            c = torch.cos(self.theta).to(y.dtype)
            s = torch.sin(self.theta).to(y.dtype)
            z = y.reshape(*y.shape[:-1], self.n // 2, 2)
            e, h = z[..., 0], z[..., 1]
            return torch.stack([e * c + h * s, -e * s + h * c],
                               dim=-1).reshape(*y.shape)
        # householder
        z = y.float()
        for i in range(self.v.shape[0]):
            v = self.v[i]
            v = v / v.norm().clamp_min(1e-8)
            z = z - 2.0 * torch.outer(z.reshape(-1, self.n) @ v,
                                      v).reshape(z.shape)
        return z.to(y.dtype)

    def apply(self, x):
        y = x
        if self.fixed is not None:
            y = self.fixed.apply(y)
        y = y[..., self.perm.to(y.device)]
        y = self._learned(y)
        return y[..., self.inv_perm.to(y.device)]

    def orth_error(self):
        with torch.no_grad():
            eye = torch.eye(min(self.n, 512), self.n,
                            device=self.perm.device)
            q = self.apply(eye.float())
            g = q @ q.t()
            return float((g - torch.eye(q.shape[0],
                                        device=q.device)).norm()
                         / math.sqrt(q.shape[0]))



def build_rotation(spec, n=2 * D, device="cpu"):
    """Factory: structured spec dict, or {'learned_ckpt': path,
    'which': 'rot_f'|'rot_r'} for a trained LearnedRotation."""
    if spec and "learned_ckpt" in spec:
        ck = torch.load(spec["learned_ckpt"], map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        rot = LearnedRotation(cfg["param"], cfg["block"],
                              cfg.get("hh_k", 8),
                              (dict(family="cross", block=8192,
                                    seed=cfg["seed"],
                                    interleave_chunk=1)
                               if cfg.get("fixed_init") else None),
                              n=n, seed=cfg["seed"], device=device)
        which = spec.get("which", "rot_f")
        sd = ck[which] if ck.get(which) is not None else ck["rot_f"]
        rot.load_state_dict(sd)
        for p in rot.parameters():
            p.requires_grad_(False)
        return rot
    return StructuredRotation(spec, n=n, device=device)
