"""Same spec -> bitwise-identical transform + identical hashes."""
import torch
from _rep3p_common import small_rot


def test_seed_determinism():
    X = torch.randn(3, 64, dtype=torch.float64)
    a = small_rot("cross", block=8, seed=11)
    b = small_rot("cross", block=8, seed=11)
    assert torch.equal(a.apply(X), b.apply(X))
    assert a.meta() == b.meta()
