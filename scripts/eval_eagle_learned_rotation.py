#!/usr/bin/env python
"""Learned-rotation deployment eval wrapper: delegates to the
validated universal evaluator; --proj-rot-* now accepts
{"learned_ckpt": path, "which": "rot_f"|"rot_r"} specs via the
build_rotation factory. No duplicated eval logic."""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable,
         os.path.join(ROOT, "scripts",
                      "eval_eagle_acceptance_length.py")]
        + sys.argv[1:]))
