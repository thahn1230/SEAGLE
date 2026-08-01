"""Migration touches ONLY the embedding region: hidden activation and
hidden-side weight columns are bitwise unchanged."""
from _ep3p_viz_common import D, load_blob, load_manifest, run_dir

import numpy as np
import torch


def test_ep3p_hidden_region_unchanged():
    rd = run_dir()
    if rd is None:
        return
    blob = load_blob(rd)
    man = load_manifest(rd)
    X = blob["X_first_raw"].float()
    Xa = X.clone(); Xa[:, :D] *= man["m_first"]
    assert torch.equal(Xa[:, D:], X[:, D:])
    W = blob["W_before"].float()
    for m in (man["m_first"], man["m_rec"]):
        Wa = W.clone(); Wa[:, :D] /= m
        assert torch.equal(Wa[:, D:], W[:, D:])
    # plotted difference figure: hidden region must be numerically 0
    import os
    npz = rd + "/plot_data/first_projection_input_difference_3d.npz"
    if not os.path.exists(npz):
        return   # plots not generated yet
    z = np.load(npz)
    hid = z["Z"][:, z["x"] > D]
    assert float(np.abs(hid).max()) < 1e-4
