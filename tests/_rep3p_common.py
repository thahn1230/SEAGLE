"""Shared fixtures for R-EP3-P tests."""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eagle_spinquant.projection_rotation import (StructuredRotation,   # noqa
                                                 transform_xw, fwht,
                                                 scale_vec)
D = 4096
N = 64   # small dim for dense checks


def small_rot(family, block=16, seed=3, n=N):
    return StructuredRotation(dict(family=family, block=block,
                                   seed=seed, interleave_chunk=1),
                              n=n)


def run_dir():
    p = os.path.join(ROOT, "runs", "REP3P_RUN_DIR")
    if not os.path.exists(p):
        return None
    return os.path.join(ROOT, open(p).read().strip())
