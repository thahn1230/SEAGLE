"""Different specs for the two paths yield different transforms."""
import torch
from _rep3p_common import small_rot, N


def test_first_recurrent_independent():
    X = torch.randn(3, N, dtype=torch.float64)
    qa = small_rot("cross", block=8, seed=1)
    qb = small_rot("cross", block=8, seed=2)
    assert not torch.allclose(qa.apply(X), qb.apply(X))
    assert qa.meta()["sign_hash"] != qb.meta()["sign_hash"]
