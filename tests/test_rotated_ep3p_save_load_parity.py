"""Rotation reconstructed from saved spec metadata is identical."""
import json
import torch
from _rep3p_common import StructuredRotation, small_rot


def test_save_load_parity():
    a = small_rot("cross", block=16, seed=9)
    spec = json.loads(json.dumps(a.spec))     # round-trip
    b = StructuredRotation(spec, n=a.n)
    X = torch.randn(4, a.n, dtype=torch.float64)
    assert torch.equal(a.apply(X), b.apply(X))
