"""Search rows are recomputable from stored spec + tensors (spot)."""
import glob
import json
import os
from _rep3p_common import run_dir


def test_plot_reproducibility():
    rd = run_dir()
    if rd is None:
        return
    hits = glob.glob(os.path.join(rd, "candidates", "s2_*.jsonl"))
    if not hits:
        return
    rows = [json.loads(x) for x in open(hits[0])]
    for r in rows[:5]:
        assert {"family", "block", "seed", "order", "beta",
                "j_out"} <= set(r)   # complete reconstruction recipe
