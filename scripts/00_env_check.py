#!/usr/bin/env python
"""Environment check -> results/env_check.json.

Reports python/torch/CUDA, GPUs + per-GPU memory, fast_hadamard_transform,
EAGLE import, SpinQuant import, and HF access for the selected model. Also runs
the download-free rotation math sanity checks so a broken environment is caught
before any heavy run. Exit 0 = ready, 1 = a blocking problem.

Usage: python scripts/00_env_check.py [--config configs/default_experiment.yaml]
"""

import argparse
import importlib
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "SpinQuant"))

import yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "default_experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "env_check.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    report = {"blocking_problems": [], "warnings": []}

    def block(msg):
        report["blocking_problems"].append(msg)

    def warn(msg):
        report["warnings"].append(msg)

    # --- python / torch / cuda ---
    report["python"] = sys.version.split()[0]
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            block("torch.cuda.is_available() is False")
        report["device_count"] = torch.cuda.device_count()
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({"index": i, "name": p.name,
                         "total_gib": round(total / 2**30, 2),
                         "free_gib": round(free / 2**30, 2)})
        report["gpus"] = gpus
    except Exception as e:
        block(f"torch import/query failed: {e}")

    # --- packages ---
    pkgs = {}
    for mod, blocking in [("transformers", True), ("accelerate", True),
                          ("datasets", True), ("safetensors", True),
                          ("sentencepiece", True), ("fastchat", False),
                          ("lm_eval", False)]:
        try:
            m = importlib.import_module(mod)
            pkgs[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pkgs[mod] = None
            (block if blocking else warn)(f"package {mod} not importable")
    report["packages"] = pkgs

    # --- fast_hadamard_transform (real CUDA kernel) ---
    try:
        import fast_hadamard_transform  # noqa: F401
        report["fast_hadamard_transform"] = True
    except Exception:
        report["fast_hadamard_transform"] = False
        block("fast_hadamard_transform not importable (needed for R3/R4)")

    # --- EAGLE import ---
    try:
        from eagle.model.ea_model import EaModel  # noqa: F401
        from eagle.model.cnets import Model  # noqa: F401
        report["eagle_import"] = True
    except Exception as e:
        report["eagle_import"] = False
        block(f"EAGLE import failed: {e}")

    # --- SpinQuant import ---
    try:
        import eval_utils.rotation_utils  # noqa: F401
        import utils.quant_utils  # noqa: F401
        import utils.fuse_norm_utils  # noqa: F401
        report["spinquant_import"] = True
    except Exception as e:
        report["spinquant_import"] = False
        block(f"SpinQuant import failed: {e}")

    # --- HF access for selected model ---
    from eagle_spinquant import model_discovery
    resolved = model_discovery.resolve_target_and_draft(cfg)
    report["model"] = resolved
    if not resolved["target_weights_cached"]:
        tgt_cfg = model_discovery.fetch_config(resolved["target_id"])
        if isinstance(tgt_cfg, str) and "gated" in tgt_cfg:
            block(f"target model {resolved['target_id']} gated and not cached: {tgt_cfg}")
        elif isinstance(tgt_cfg, str):
            warn(f"target weights not cached yet (config status: {tgt_cfg})")
        else:
            warn("target weights not fully cached yet (config reachable)")
    if not resolved["draft_weights_cached"]:
        warn("draft weights not cached yet")

    # --- git commits (provenance) ---
    from eagle_spinquant import logging_utils
    report["eagle_commit"] = logging_utils.git_commit(logging_utils.EAGLE_DIR)
    report["eagle_branch"] = logging_utils.git_branch(logging_utils.EAGLE_DIR)
    report["spinquant_commit"] = logging_utils.git_commit(logging_utils.SPINQUANT_DIR)

    # --- rotation math sanity (download-free) ---
    try:
        from eagle_spinquant import rotation_interface as ri
        math_checks = {
            "unrotation_roundtrip": ri.check_unrotation_roundtrip(),
            "attention_preservation": ri.check_attention_preservation(),
        }
        report["math_sanity"] = math_checks
        if not math_checks["unrotation_roundtrip"]["ok"]:
            block("unrotation roundtrip failed")
        if not math_checks["attention_preservation"]["preserved"]:
            block("attention preservation (R3) failed")
    except Exception as e:
        warn(f"math sanity checks failed to run: {e}")

    report["ready"] = len(report["blocking_problems"]) == 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in
                      ["python", "torch", "device_count", "fast_hadamard_transform",
                       "eagle_import", "spinquant_import", "ready"] if k in report}, indent=2))
    print(f"\nfull report -> {args.out}")
    if report["warnings"]:
        print("WARNINGS:")
        for w in report["warnings"]:
            print("  -", w)
    if report["blocking_problems"]:
        print("BLOCKING:")
        for b in report["blocking_problems"]:
            print("  -", b)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
