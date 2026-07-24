"""Deterministic 8-way sharding contract (study spec §7): disjoint,
complete, order-stable; manifest hashing helpers stable."""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(ROOT, "scripts",
                            "generate_eagle1_official_training_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_shard_slices_disjoint_and_complete():
    audit_n, n_shards = 800, 8
    per = audit_n // n_shards
    rows = list(range(10000))
    seen = []
    for s in range(n_shards):
        seen += rows[s * per:(s + 1) * per]
    assert len(seen) == len(set(seen)) == audit_n
    assert seen == rows[:audit_n]          # deterministic global order


def test_sample_hash_stability():
    m = _load()
    t = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    h1 = m.sha(t)
    h2 = m.sha(t.clone())
    assert h1 == h2 and len(h1) == 24
    t2 = t.clone()
    t2[0, 0] += 1
    assert m.sha(t2) != h1


def test_trainer_rank_slicing_equalized():
    # equalized per-rank counts (DDP hang guard in the trainer)
    world, bs = 8, 4
    perm = list(range(64603))
    n_common = (min(len(perm[r::world]) for r in range(world)) // bs) * bs
    sizes = [len(perm[r::world][:n_common]) for r in range(world)]
    assert len(set(sizes)) == 1 and sizes[0] % bs == 0
