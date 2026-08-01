"""Shared fixtures for the EP3-P visualization tests: load the real
collected run (runs/EP3PVIZ_RUN_DIR) when present."""
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 4096


def run_dir():
    p = os.path.join(ROOT, "runs", "EP3PVIZ_RUN_DIR")
    if not os.path.exists(p):
        return None
    rd = os.path.join(ROOT, open(p).read().strip())
    return rd if os.path.exists(
        os.path.join(rd, "plot_data", "ep3p_tensors.pt")) else None


def load_blob(rd):
    return torch.load(os.path.join(rd, "plot_data",
                                   "ep3p_tensors.pt"),
                      map_location="cpu", weights_only=False)


def load_manifest(rd):
    return json.load(open(os.path.join(
        rd, "metadata", "collection_manifest.json")))
