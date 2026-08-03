"""FP output preserved on REAL recurrent tensors, depths 1-4."""
import os
import torch
from _rep3p_common import StructuredRotation, transform_xw, run_dir


def test_recurrent_fp_equivalence():
    rd = run_dir()
    if rd is None or not os.path.exists(
            os.path.join(rd, "tensors", "calib_int4.pt")):
        return
    t = torch.load(os.path.join(rd, "tensors", "calib_int4.pt"),
                   map_location="cpu", weights_only=False)
    W = t["W"].double()
    rot = StructuredRotation(dict(family="cross", block=64, seed=10,
                                  interleave_chunk=1))
    for k in (1, 2, 3, 4):
        X = t[f"X_rec{k}"][:64].double()
        Xt, Wt = transform_xw(X, W, 42.22, rot, "SR")
        rel = float((Xt @ Wt.t() - X @ W.t()).abs().max()
                    / (X @ W.t()).abs().max())
        assert rel < 1e-12, (k, rel)
