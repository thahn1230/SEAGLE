"""Logging, run metadata, seeding, and JSONL/CSV writers.

Every experiment run records: exact git commits of both third_party repos, the
resolved environment, the config used, and per-measurement metric rows. This is
the single source of truth for reproducibility (docs/05 report is generated from
these files).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EAGLE_DIR = os.path.join(PROJECT_ROOT, "third_party", "EAGLE")
SPINQUANT_DIR = os.path.join(PROJECT_ROOT, "third_party", "SpinQuant")


def git_commit(repo: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def git_branch(repo: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "branch", "--show-current"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic where cheap; do NOT force deterministic algos globally
    # because some fused kernels have no deterministic variant and would raise.
    torch.backends.cudnn.benchmark = False


def env_summary() -> dict[str, Any]:
    """Environment snapshot. Import torch lazily so this module stays importable
    even in a minimal interpreter."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eagle_commit": git_commit(EAGLE_DIR),
        "eagle_branch": git_branch(EAGLE_DIR),
        "spinquant_commit": git_commit(SPINQUANT_DIR),
        "spinquant_branch": git_branch(SPINQUANT_DIR),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {"index": i, "name": props.name,
                 "total_gib": round(props.total_memory / 2**30, 2)}
            )
    except Exception as e:  # torch missing / broken
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        import fast_hadamard_transform  # noqa: F401
        info["fast_hadamard_transform"] = True
    except Exception:
        info["fast_hadamard_transform"] = False
    return info


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)  # tensors, dtypes, paths, etc.


class RunLogger:
    """Append-only JSONL logger for one run directory, plus a metadata sidecar."""

    def __init__(self, run_dir: str, run_name: str, config: dict | None = None):
        self.run_dir = run_dir
        self.run_name = run_name
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, f"{run_name}.jsonl")
        self.meta_path = os.path.join(run_dir, f"{run_name}.meta.json")
        self._rows: list[dict] = []
        meta = {"run_name": run_name, "env": env_summary(), "config": _jsonable(config or {})}
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log(self, **row: Any) -> dict:
        row = _jsonable(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._rows.append(row)
        return row

    def rows(self) -> list[dict]:
        return list(self._rows)


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(_jsonable(r)) + "\n")


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [_jsonable(r) for r in rows]
    if not rows:
        # Still create the file so downstream tools don't choke on a missing path.
        open(path, "w").close()
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
