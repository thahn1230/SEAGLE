#!/usr/bin/env python
"""FP16 retraining CONTROL arms (C6, C8) for the PTQ-vs-QAT study.

This is NOT QAT: no fake quantization anywhere in the draft forward.
Identical data order, budget, optimizer, schedule, and stopping rule as
the QAT arms (spec section 9.3) — it delegates to the shared trainer in
train_eagle_draft_int4_qat.py, which disables every quantizer for these
arms (w_bits = a_bits = 16, alpha = 1) and trains the stock-basis (C8)
or restored-basis target-adapted (C6) draft with the original EAGLE
objective."""
import sys

sys.argv[0] = "train_eagle_draft_fp16_control.py"
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "qat_trainer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "train_eagle_draft_int4_qat.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    arm = None
    for i, a in enumerate(sys.argv):
        if a == "--arm" and i + 1 < len(sys.argv):
            arm = sys.argv[i + 1]
    assert arm in ("C6", "C8"), \
        "FP16 control script only runs the non-QAT arms C6/C8"
    sys.exit(mod.main())
