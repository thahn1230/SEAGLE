#!/usr/bin/env python
"""GENERIC W4A4 draft-core QAT — no P3, no migration (study §6).

Wrapper over the validated single-step QAT trainer with the migration
factor pinned to exactly 1.0 (m = D**0 = 1): no embedding multiply, no
W_e division, no P2 branch scales, no trainable migration scalar, no
rotation training. The valid target-draft rotation/gamma interface is
kept (correctness requirement, not part of P3). Same quantizer contract,
same anchor, same teacher-matched deployment, same objective/budget as
the D4P3 QAT arms.

  migration_mode = "none"; migration_factor = 1.0
  beta_first = beta_recurrent = 0.0

Variants: T0 (fp16 teacher/deploy, identity interface) -> arm C3;
          T1 (w4a4 teacher/deploy, gamma_R1 interface) -> arm C7.
Writes a no-P3 audit record per run.
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM = {"T0": "C3", "T1": "C7"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["T0", "T1"])
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt-every", type=int, default=500)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"GQAT_{args.variant}_s{args.seed}"
    audit = dict(tag=tag, migration_mode="none", migration_factor=1.0,
                 beta_first=0.0, beta_recurrent=0.0,
                 p3=False, p2=False, lk=False, rotation_training=False,
                 interface="valid rotation/gamma (correctness, not P3)",
                 note="alpha pinned 1.0: E*1.0 and W_e/1.0 are exact "
                      "identities -> bitwise-generic projection")
    os.makedirs(os.path.join(args.run_dir, "manifests"), exist_ok=True)
    with open(os.path.join(args.run_dir, "manifests",
                           f"gqat_audit_{tag}.json"), "w") as f:
        json.dump(audit, f, indent=1)
    cmd = [sys.executable,
           os.path.join(ROOT, "scripts", "train_eagle_draft_int4_qat.py"),
           "--arm", ARM[args.variant], "--seed", str(args.seed),
           "--run-dir", args.run_dir, "--alpha", "1.0",
           "--lr", str(args.lr), "--warmup", str(args.warmup),
           "--steps", str(args.steps), "--init-sd", args.anchor,
           "--save-ckpt-every", str(args.save_ckpt_every),
           "--tag", tag]
    print("[gqat]", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
