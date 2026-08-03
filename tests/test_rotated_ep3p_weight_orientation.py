"""Audited orientation: W_pt [out,in], y = x @ W.T; the SR transform
must satisfy (x T) @ W'.T == x @ W.T with W' = (W S^-1) Q."""
import torch
from _rep3p_common import small_rot, scale_vec, N


def test_weight_orientation():
    torch.manual_seed(1)
    X = torch.randn(5, N, dtype=torch.float64)
    W = torch.randn(7, N, dtype=torch.float64)     # [out, in]
    rot = small_rot("cross", block=8)
    m = 42.22
    s = scale_vec(N, m).double()
    Q = rot.to_dense()
    Xt = (X * s) @ Q
    Wt = (W / s) @ Q                                # rows right-mult
    assert torch.allclose(Xt @ Wt.t(), X @ W.t(), atol=1e-9)
    # explicit T^{-T} check: T = S Q  ->  W' = W_pt T^{-T}
    T = torch.diag(s) @ Q
    Wt2 = W @ torch.linalg.inv(T).t()
    assert torch.allclose(Wt, Wt2, atol=1e-9)
