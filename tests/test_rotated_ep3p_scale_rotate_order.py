"""SR: x' = (x S) Q, W' = (W S^-1) Q — verified against dense T."""
import torch
from _rep3p_common import small_rot, transform_xw, scale_vec, N


def test_scale_rotate_order():
    torch.manual_seed(2)
    X = torch.randn(4, N, dtype=torch.float64)
    W = torch.randn(5, N, dtype=torch.float64)
    rot = small_rot("full", block=N)
    m = 9.7
    Xt, Wt = transform_xw(X, W, m, rot, "SR")
    Q = rot.to_dense()
    s = scale_vec(N, m)
    assert torch.allclose(Xt, (X * s) @ Q, atol=1e-10)
    assert torch.allclose(Wt, (W / s) @ Q, atol=1e-10)
    assert torch.allclose(Xt @ Wt.t(), X @ W.t(), atol=1e-9)
