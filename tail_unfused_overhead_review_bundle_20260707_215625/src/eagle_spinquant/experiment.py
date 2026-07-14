"""Shared experiment helpers: config loading, path resolution, MT-bench prompts."""

from __future__ import annotations

import json
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default_experiment.yaml")
EAGLE_DIR = os.path.join(PROJECT_ROOT, "third_party", "EAGLE")
MT_BENCH = os.path.join(EAGLE_DIR, "eagle", "data", "mt_bench", "question.jsonl")


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict:
    from . import model_discovery
    return model_discovery.resolve_target_and_draft(cfg)


def load_mt_bench_prompts(n: int | None = None) -> list[dict]:
    """Return MT-bench turn-1 prompts (question_id, category, text)."""
    prompts = []
    if not os.path.isfile(MT_BENCH):
        return prompts
    with open(MT_BENCH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            prompts.append({
                "question_id": q["question_id"],
                "category": q.get("category", ""),
                "text": q["turns"][0],
            })
            if n is not None and len(prompts) >= n:
                break
    return prompts
