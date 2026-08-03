#!/usr/bin/env python
"""R-EP3-P RCAL wrapper: delegates to the validated lockstep capture
(scripts/capture_eagle_proposal_cycles.py) + metrics. No duplicated
protocol logic."""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable,
         os.path.join(ROOT, "scripts",
                      "capture_eagle_proposal_cycles.py")]
        + sys.argv[1:]))
