#!/usr/bin/env python
"""Single-step low-LR QAT from the fresh FP16 anchor (spec §16-18).

Thin adapter over the validated leaf-grad QAT trainer
(scripts/train_eagle_draft_int4_qat.py): original EAGLE-1 single-step
objective, runtime-matched STE W4A4 at the deployed sites only, teacher
matched to deployment (Q0 fp16 target -> arm C3 structure; Q1 W4A4
target -> arm C7 structure), anchor initialization, step-stamped
checkpoints for offline best-validation-AL selection.

  python scripts/train_eagle1_single_step_qat.py --variant Q0|Q1 \
      --anchor checkpoints/eagle1_fresh_fp16_anchor/anchor.pt \
      --alpha A --lr 3e-6 --seed 0 --steps 3000 --run-dir RD
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = {"Q0": "C3", "Q1": "C7"}   # identity/fp16-teacher, gammaR1/int4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["Q0", "Q1"])
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt-every", type=int, default=500)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.variant}lr{args.lr:g}_s{args.seed}"
    cmd = [sys.executable,
           os.path.join(ROOT, "scripts", "train_eagle_draft_int4_qat.py"),
           "--arm", ARM[args.variant], "--seed", str(args.seed),
           "--run-dir", args.run_dir, "--alpha", str(args.alpha),
           "--lr", str(args.lr), "--warmup", str(args.warmup),
           "--steps", str(args.steps), "--init-sd", args.anchor,
           "--save-ckpt-every", str(args.save_ckpt_every),
           "--tag", tag]
    print("[qat-single]", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
