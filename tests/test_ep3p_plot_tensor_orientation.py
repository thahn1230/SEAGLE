"""Weight orientation audit: storage [out,in], forward Y = X @ W.T,
plot x-axis = input channels (dim 1)."""
from _ep3p_viz_common import D, load_blob, load_manifest, run_dir

import torch


def test_ep3p_plot_tensor_orientation():
    rd = run_dir()
    if rd is None:
        return
    blob = load_blob(rd)
    man = load_manifest(rd)
    W = blob["W_before"]
    assert list(W.shape) == [D, 2 * D]        # [out, in]
    X = blob["X_first_raw"].float()
    assert X.shape[1] == 2 * D
    # forward orientation must be X @ W.T (shapes only compose this way)
    Y = X[:4] @ W.t()
    assert Y.shape == (4, D)
    assert "in_channel" in man["weight_orientation"]["storage"]
    assert man["weight_orientation"]["input_channel_axis"].startswith(
        "columns")
