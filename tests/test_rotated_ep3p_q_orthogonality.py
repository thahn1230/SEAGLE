"""Q^T Q = I within tolerance for every family."""
import torch
from _rep3p_common import small_rot, N


def test_q_orthogonality():
    for fam, blk in (("e_only", 8), ("h_only", 16), ("dual", 8),
                     ("full", N), ("cross", 4)):
        Q = small_rot(fam, block=blk).to_dense()
        err = (Q.t() @ Q - torch.eye(N, dtype=torch.float64)).norm() \
            / (N ** 0.5)
        assert err < 1e-12, (fam, float(err))
