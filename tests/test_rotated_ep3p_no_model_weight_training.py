"""R6 optimizer trains ONLY the Cayley skew params (model frozen)."""
import os
from _rep3p_common import ROOT


def test_no_model_weight_training():
    src = open(os.path.join(
        ROOT, "scripts",
        "build_structured_projection_rotation.py")).read()
    assert "Adam(cay.parameters()" in src
    assert "model" not in src.split("Adam(")[1][:80]
    assert "load_state_dict" not in src   # never writes model weights
