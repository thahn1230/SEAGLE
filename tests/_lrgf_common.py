import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
D = 4096


def run_dir():
    p = os.path.join(ROOT, "runs", "LRGF_RUN_DIR")
    return (os.path.join(ROOT, open(p).read().strip())
            if os.path.exists(p) else None)


def tbl(name):
    rd = run_dir()
    if rd is None:
        return None
    p = os.path.join(rd, "tables", name)
    return json.load(open(p)) if os.path.exists(p) else None
