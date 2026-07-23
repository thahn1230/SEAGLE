#!/usr/bin/env python
"""One-shot driver for the complete PTQ-vs-QAT matrix (spec sections 5-6).

Phases (each idempotent; done work is skipped via existing artifacts):
  0  audit + Gate A provenance (audit_eagle_qat_pipeline.py)
  1  training-free PTQ family on all GPUs (C1,C2,C4,C5,N1-N3(+int4),
     C11,C12) + calibration-pool alpha sweeps       [eval script]
  2  trainings + dependent evals + oracle + fidelity + cross-domain via
     the dynamic GPU scheduler (schedule_eagle_ptq_qat_jobs.py)
  3  Gate D parity, bootstrap battery, attainment tables, plots

This driver serializes the phases; the scheduler parallelizes inside
phase 2. Run under nohup; expect ~24h wall-clock on 6x RTX 4090.
"""
import argparse, os, subprocess, sys

ROOT = "/home/thahn1230/eagle_spinquant_w4a4"
PY = sys.executable


def sh(cmd, env_extra=None, check=True):
    env = dict(os.environ, TMPDIR="/data/thahn1230/tmp")
    if env_extra:
        env.update(env_extra)
    print(f"[matrix] RUN {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
    if check and r.returncode != 0:
        raise SystemExit(f"[matrix] FAILED ({r.returncode}): {cmd}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha-fp16", type=float, default=45.254834)
    ap.add_argument("--alpha-int4", type=float, default=45.254834)
    ap.add_argument("--skip-phase", default="",
                    help="comma list of phases to skip, e.g. 0,1")
    args = ap.parse_args()
    rd = args.run_dir
    skip = set(args.skip_phase.split(",")) if args.skip_phase else set()

    if "0" not in skip:
        sh(f"{PY} scripts/audit_eagle_qat_pipeline.py --run-dir {rd}")

    if "1" not in skip:
        E = (f"{PY} scripts/eval_eagle_acceptance_length.py --run-dir "
             f"{rd} --n-prompts 80 --datasets mtbench")
        ptq = [
            f"{E} --target fp16 --draft-cfg stock --tag C1_fp16_stock",
            f"{E} --target fp16 --draft-cfg naive_w4a4 "
            f"--tag N1_fp16_naive",
            f"{E} --target fp16 --draft-cfg d4p3 "
            f"--alpha {args.alpha_fp16} --tag C2_fp16_d4p3",
            f"{E} --target fp16 --draft-cfg p2 --tag N3_fp16_p2",
            f"{E} --target int4 --draft-cfg stock "
            f"--tag C4_int4_stockrestored",
            f"{E} --target int4 --draft-cfg naive_w4a4 "
            f"--tag N2_int4_naive",
            f"{E} --target int4 --draft-cfg d4p3 "
            f"--alpha {args.alpha_int4} --tag C5_int4_d4p3",
            f"{E} --target int4 --draft-cfg p2 --tag N3b_int4_p2",
        ]
        for i, cmd in enumerate(ptq):
            sh(cmd, env_extra={"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                               "CUDA_VISIBLE_DEVICES": str(i % 6)},
               check=False)

    if "2" not in skip:
        sh(f"{PY} scripts/schedule_eagle_ptq_qat_jobs.py --run-dir {rd} "
           f"--alpha-fp16 {args.alpha_fp16} "
           f"--alpha-int4 {args.alpha_int4}")

    if "3" not in skip:
        sh(f"{PY} scripts/check_qat_deploy_parity.py --run-dir {rd} "
           f"--mode both --alpha {args.alpha_fp16}",
           env_extra={"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                      "CUDA_VISIBLE_DEVICES": "0"}, check=False)
        sh(f"{PY} scripts/eval_eagle_oracle_ceiling.py --self-test")
        sh(f"{PY} scripts/bootstrap_eagle_tau.py --run-dir {rd} "
           f"--battery")
        sh(f"{PY} scripts/analyze_qat_attainment.py --run-dir {rd}")
        sh(f"{PY} scripts/plot_eagle_ptq_qat_results.py --run-dir {rd}")
    print("[matrix] COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
