"""Proxy search uses the deployment quantizers (fake_w4a4_draft),
not a private re-implementation."""
import os
from _rep3p_common import ROOT


def test_quantizer_contract():
    for f in ("calibrate_rotated_ep3p.py",
              "build_structured_projection_rotation.py"):
        src = open(os.path.join(ROOT, "scripts", f)).read()
        assert "fake_w4a4_draft" in src, f
        assert "_weight_fake_quant" in src and "_act_quantizer" in src
        assert "def _weight_fake_quant" not in src   # no duplicate
