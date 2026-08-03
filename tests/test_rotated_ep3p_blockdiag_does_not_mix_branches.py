"""dual / e_only / h_only must NOT transfer across the boundary."""
import torch
from _rep3p_common import small_rot, N


def test_blockdiag_does_not_mix_branches():
    for fam in ("dual", "e_only", "h_only"):
        Q = small_rot(fam, block=8).to_dense()
        cross = Q[: N // 2, N // 2:].abs().sum() \
            + Q[N // 2:, : N // 2].abs().sum()
        assert float(cross) == 0.0, fam
