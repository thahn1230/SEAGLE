"""FWHT is normalized (norm-preserving) and self-inverse."""
import torch
from _rep3p_common import fwht


def test_hadamard_normalization():
    torch.manual_seed(4)
    for blk in (2, 16, 64, 256):
        x = torch.randn(7, 256, dtype=torch.float64)
        y = fwht(x, blk)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1),
                              atol=1e-10)
        assert torch.allclose(fwht(y, blk), x, atol=1e-10)
