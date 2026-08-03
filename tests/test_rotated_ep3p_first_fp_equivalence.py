"""FP output preserved on REAL first-path tensors (fp32/fp16)."""
import os
import torch
from _rep3p_common import StructuredRotation, transform_xw, run_dir


def test_first_fp_equivalence():
    rd = run_dir()
    if rd is None or not os.path.exists(
            os.path.join(rd, "tensors", "calib_int4.pt")):
        return
    t = torch.load(os.path.join(rd, "tensors", "calib_int4.pt"),
                   map_location="cpu", weights_only=False)
    X = t["X_first"][:128].float()
    W = t["W"].float()
    rot = StructuredRotation(dict(family="cross", block=64, seed=12,
                                  interleave_chunk=1))
    Xt, Wt = transform_xw(X.double(), W.double(), 27.86, rot, "SR")
    rel = float((Xt @ Wt.t() - X.double() @ W.double().t()).abs()
                .max() / (X.double() @ W.double().t()).abs().max())
    assert rel < 1e-12
    Xh, Wh = transform_xw(X, W, 27.86, rot, "SR")
    relh = float((Xh.half().float() @ Wh.half().float().t()
                  - X @ W.t()).abs().max()
                 / (X @ W.t()).abs().max())
    assert relh < 5e-2    # fp16 dtype-appropriate
