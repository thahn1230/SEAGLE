"""The complete projection input has 2D channels; the complete weight
holds embedding + hidden input regions in ONE matrix, boundary at D."""
from _ep3p_viz_common import D, load_blob, run_dir

import numpy as np
import torch


def test_ep3p_full_projection_boundary():
    rd = run_dir()
    if rd is None:
        return
    blob = load_blob(rd)
    for k in ("X_first_raw", "X_rec1_raw", "X_rec2_raw",
              "X_rec3_raw", "X_rec4_raw"):
        assert blob[k].shape[1] == 2 * D, k
    W = blob["W_before"]
    assert W.shape[1] == 2 * D
    # both regions live in the same matrix and are non-trivial
    assert float(W[:, :D].abs().sum()) > 0
    assert float(W[:, D:].abs().sum()) > 0
    # plotted matrices keep the full 2D-channel extent
    import os
    npz = rd + "/plot_data/first_projection_input_before_3d.npz"
    if not os.path.exists(npz):
        return   # plots not generated yet
    z = np.load(npz)
    assert z["x"].max() > D and z["x"].min() < D
