"""First path uses m_first = D**0.40: e-slice x m, W_e / m."""
from _ep3p_viz_common import D, load_blob, load_manifest, run_dir

import torch


def test_ep3p_first_migration_values():
    rd = run_dir()
    if rd is None:
        return
    man = load_manifest(rd)
    assert man["beta_first"] == 0.40
    m = man["m_first"]
    assert abs(m - D ** 0.40) < 1e-9
    assert abs(m - 27.8576) < 1e-3
    blob = load_blob(rd)
    X = blob["X_first_raw"].float()
    Xa = X.clone(); Xa[:, :D] *= m
    assert torch.allclose(Xa[:, :D], X[:, :D] * m)
    W = blob["W_before"].float()
    Wf = W.clone(); Wf[:, :D] /= m
    assert torch.allclose(Wf[:, :D] * m, W[:, :D], rtol=1e-6)
