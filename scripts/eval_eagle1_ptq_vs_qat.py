#!/usr/bin/env python
"""Cell evaluator for the official from-scratch study (spec §21).

Maps study cells to the validated universal acceptance evaluator
(scripts/eval_eagle_acceptance_length.py), always deploying from an
explicit checkpoint (--draft-sd), so every quantized method shares the
exact same quantizer contract:

  C0  public FP16 draft            (stock ea weights, no --draft-sd)
  C1  fresh FP16 anchor            (fp16_deploy, anchor sd)
  C2  fresh naive W4A4 (no P3)     (naive deploy of anchor)
  C3  fresh D4P3 TF-PTQ            (d4p3_deploy, calibrated alpha)
  C4  fresh D4P3 + local R_D       (rot deploy with anchor sd)
  C5  single-step QAT ckpt         (d4p3_deploy, QAT sd)
  C6  multi-step QAT ckpt          (d4p3_deploy, QAT sd)

Usage: --cell C3 --target fp16|int4 --alpha A [--sd path] [--ckpt R_D]
       --run-dir RD [--datasets ...] [--pool eval|calib] [--tag TAG]
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELL_CFG = {
    "C0": dict(cfg="stock", sd=False),
    "C1": dict(cfg="fp16_deploy", sd=True),
    "C2": dict(cfg="naive_w4a4_deploy", sd=True),
    "C3": dict(cfg="d4p3_deploy", sd=True),
    "C4": dict(cfg="rot", sd=True),
    "C5": dict(cfg="d4p3_deploy", sd=True),
    "C6": dict(cfg="d4p3_deploy", sd=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELL_CFG))
    ap.add_argument("--target", required=True, choices=["fp16", "int4"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sd", default=None)
    ap.add_argument("--ckpt", default=None, help="R_D ckpt for C4")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--datasets", default="mtbench")
    ap.add_argument("--pool", default="eval")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cc = CELL_CFG[args.cell]
    cfg = cc["cfg"]
    if cfg == "naive_w4a4_deploy":
        # naive = universal evaluator's naive_w4a4 applied to a loaded sd
        cfg = "naive_w4a4"
    tag = args.tag or f"OF_{args.cell}_{args.target}"
    cmd = [sys.executable,
           os.path.join(ROOT, "scripts", "eval_eagle_acceptance_length.py"),
           "--run-dir", args.run_dir, "--target", args.target,
           "--draft-cfg", cfg, "--datasets", args.datasets,
           "--pool", args.pool, "--n-prompts", str(args.n_prompts),
           "--tag", tag]
    if cc["sd"]:
        assert args.sd, f"{args.cell} needs --sd (anchor/QAT checkpoint)"
        cmd += ["--draft-sd", args.sd]
    if args.alpha is not None:
        cmd += ["--alpha", str(args.alpha)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    print("[eval-cell]", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
