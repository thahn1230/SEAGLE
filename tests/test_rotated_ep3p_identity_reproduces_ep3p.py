"""Q = I must exactly recover EP3-P; beta=0 + Q=I recovers NPTQ."""
import torch
from _rep3p_common import small_rot, transform_xw, N


def test_identity_reproduces_ep3p():
    torch.manual_seed(0)
    X = torch.randn(8, N, dtype=torch.float64)
    W = torch.randn(6, N, dtype=torch.float64)
    rot = small_rot("identity")
    m = 27.86
    Xt, Wt = transform_xw(X, W, m, rot, "SR")
    Xe = X.clone(); Xe[:, : N // 2] *= m
    We = W.clone(); We[:, : N // 2] /= m
    assert torch.equal(Xt, Xe) and torch.equal(Wt, We)
    Xt0, Wt0 = transform_xw(X, W, 1.0, rot, "SR")
    assert torch.equal(Xt0, X) and torch.equal(Wt0, W)   # NPTQ
