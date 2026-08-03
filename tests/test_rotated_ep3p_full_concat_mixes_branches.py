"""Full/cross Q must transfer contribution across the e/h boundary."""
import torch
from _rep3p_common import small_rot, N


def test_full_concat_mixes_branches():
    for fam, blk in (("full", N), ("cross", 4)):
        Q = small_rot(fam, block=blk).to_dense()
        cross_mass = Q[: N // 2, N // 2:].abs().sum() \
            + Q[N // 2:, : N // 2].abs().sum()
        assert cross_mass > 1.0, fam
        e_in = torch.zeros(1, N, dtype=torch.float64)
        e_in[0, : N // 2] = torch.randn(N // 2)
        out = small_rot(fam, block=blk).apply(e_in)
        assert out[0, N // 2:].abs().sum() > 1e-3, fam
