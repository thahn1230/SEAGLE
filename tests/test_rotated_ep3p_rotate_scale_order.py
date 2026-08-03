"""RS: x' = (x Q) S; distinct from SR for cross-branch Q, identical
for block-diagonal Q (commutes with S)."""
import torch
from _rep3p_common import small_rot, transform_xw, N


def test_rotate_scale_order():
    torch.manual_seed(3)
    X = torch.randn(4, N, dtype=torch.float64)
    W = torch.randn(5, N, dtype=torch.float64)
    m = 27.86
    full = small_rot("full", block=N)
    a = transform_xw(X, W, m, full, "SR")[0]
    b = transform_xw(X, W, m, full, "RS")[0]
    assert not torch.allclose(a, b)          # non-commuting
    for o in ("SR", "RS"):
        Xt, Wt = transform_xw(X, W, m, full, o)
        assert torch.allclose(Xt @ Wt.t(), X @ W.t(), atol=1e-9)
    dual = small_rot("dual", block=8)
    a = transform_xw(X, W, m, dual, "SR")[0]
    b = transform_xw(X, W, m, dual, "RS")[0]
    assert torch.allclose(a, b, atol=1e-10)  # commuting
