#!/usr/bin/env python
"""SpinQuant rotation optimization wrapper (target 4 / task R-1).

Wraps SpinQuant's optimize_rotation.py (torchrun). Produces the learned R1/R2
checkpoint (R.bin) for the CHAT target. Because PTQ will use GPTQ weights, the
default optimizes rotations with w_bits=16 (SpinQuant README guidance); a
w_bits=4 variant can be produced for ablation.

Starts on 1 GPU; escalate --nproc-per-node on OOM (DDP replicates the model, so
more GPUs only speed wall-clock — see docs/00 S5). Logs the full command, wall
clock, and a metadata JSON. This is a heavy run: use --background or nohup.

Usage:
  python scripts/10_optimize_rotation.py --w-bits 16 --a-bits 4 --kv-bits 4 \
      --nproc-per-node 1 --max-steps 100
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eagle_spinquant import experiment, logging_utils, spinquant_bridge as sb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--w-bits", type=int, default=16)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=4)
    ap.add_argument("--nproc-per-node", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", default=None, help="output rotation dir (holds R.bin)")
    ap.add_argument("--timeout", type=int, default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg = experiment.load_config(args.config)
    # Optimize on the base model id (gated); the cache already holds the chat model.
    input_model = cfg["model"]["target"]

    tag = f"w{args.w_bits}a{args.a_bits}kv{args.kv_bits}"
    out_dir = args.out or os.path.join(PROJECT_ROOT, "outputs", "rotations", tag)
    # MUST be absolute: the SpinQuant subprocess runs with cwd=third_party/SpinQuant,
    # so a relative --output_rotation_path would save R.bin under the SpinQuant repo.
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # log path derived from the output dir basename so distinct runs don't collide
    log_path = os.path.join(PROJECT_ROOT, "runs", "rotation_opt",
                            f"{os.path.basename(out_dir.rstrip('/'))}.log")
    # unique master port per output dir to avoid rendezvous collisions
    master_port = 29500 + (abs(hash(out_dir)) % 200)

    access_token = os.environ.get("HF_TOKEN")
    cmd = sb.build_optimize_rotation_cmd(
        input_model=input_model, output_rotation_path=out_dir,
        w_bits=args.w_bits, a_bits=args.a_bits, kv_bits=args.kv_bits,
        nproc_per_node=args.nproc_per_node, max_steps=args.max_steps,
        seqlen=args.seqlen, access_token=access_token, master_port=master_port,
    )
    print("CMD:", " ".join(cmd))
    t0 = time.time()
    res = sb.run_spinquant_cmd(cmd, log_path, timeout=args.timeout)
    elapsed = time.time() - t0

    r_bin = os.path.join(out_dir, "R.bin")
    meta = {
        "tag": tag, "returncode": res["returncode"], "elapsed_s": elapsed,
        "cmd": cmd, "log": log_path, "r_bin": r_bin,
        "r_bin_exists": os.path.isfile(r_bin),
        "input_model": input_model, "nproc_per_node": args.nproc_per_node,
        "max_steps": args.max_steps, "env": logging_utils.env_summary(),
    }
    meta_path = os.path.join(out_dir, "rotation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(logging_utils._jsonable(meta), f, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ["tag", "returncode", "elapsed_s", "r_bin_exists", "r_bin"]}, indent=2))
    print(f"log -> {log_path}\nmeta -> {meta_path}")
    return 0 if (res["returncode"] == 0 and os.path.isfile(r_bin)) else 1


if __name__ == "__main__":
    sys.exit(main())
