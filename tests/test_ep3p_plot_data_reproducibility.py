"""Saved plot data regenerates the reported statistics (spec §15.9):
run the verifier end to end on the real run."""
import os
import subprocess
import sys

from _ep3p_viz_common import ROOT, run_dir


def test_ep3p_plot_data_reproducibility():
    rd = run_dir()
    if rd is None:
        return
    if not os.path.exists(os.path.join(
            rd, "tables", "ep3p_first_statistics.csv")):
        return   # plots not generated yet
    r = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "scripts",
                      "verify_ep3p_visualization_data.py"),
         "--run-dir", rd], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
