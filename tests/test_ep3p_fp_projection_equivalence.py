"""FP projection output is preserved by the migration within fp32
roundoff (scale in-activation / scale-out-of-weight cancels)."""
from _ep3p_viz_common import D, load_blob, load_manifest, run_dir

import torch


def test_ep3p_fp_projection_equivalence():
    rd = run_dir()
    if rd is None:
        return
    blob = load_blob(rd)
    man = load_manifest(rd)
    b = blob.get("bias")
    for key, m in (("X_first_raw", man["m_first"]),
                   ("X_rec1_raw", man["m_rec"])):
        X = blob[key].float()[:256].double()
        W = blob["W_before"].double()
        Xa = X.clone(); Xa[:, :D] *= m
        Wa = W.clone(); Wa[:, :D] /= m
        Yb = X @ W.t()
        Ya = Xa @ Wa.t()
        if b is not None:
            Yb = Yb + b.double()
            Ya = Ya + b.double()
        rel = float((Ya - Yb).abs().max() /
                    Yb.abs().max().clamp_min(1e-30))
        assert rel < 1e-12, (key, rel)
