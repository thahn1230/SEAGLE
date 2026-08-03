"""Rotation composes AFTER the validated fold: the gamma_R1 first
interface lives in W (build_concat_selective_weights) and Q is applied
to the assembled input/W only — asserted by construction: adapter
applies rot to W AFTER the fold and never touches gamma or upstream
hidden states (source assertion)."""
import os
from _rep3p_common import ROOT


def test_gamma_once():
    src = open(os.path.join(
        ROOT, "src", "eagle_spinquant",
        "concat_selective_projection.py")).read()
    i_fold = src.index("W_first, W_rec, bias = build_concat_selective_weights")
    i_rot = src.index("self._rot_f = StructuredRotation")
    assert i_rot > i_fold          # rotation strictly after gamma fold
    assert "gamma" not in src[i_rot:i_rot + 400].lower()
