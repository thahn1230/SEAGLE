#!/usr/bin/env python
"""R-EP3-P acceptance wrapper: delegates to the validated universal
evaluator (scripts/eval_eagle_acceptance_length.py) with the
projection-local rotation specs. No duplicated eval logic."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable,
         os.path.join(ROOT, "scripts",
                      "eval_eagle_acceptance_length.py")]
        + sys.argv[1:]))
