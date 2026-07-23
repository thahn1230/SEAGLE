#!/usr/bin/env python
"""Dynamic GPU scheduler for the PTQ-vs-QAT study (spec section 2).

Maintains a dependency DAG of training/eval jobs and assigns each pending
job to the first genuinely free GPU in {0..5} (GPU 6/7 never touched:
6 reserved, 7 lost). A GPU is free when nvidia-smi reports <1 GiB used and
no tracked job is running on it. State is persisted to
<run>/manifests/scheduler_state.json so the scheduler can be restarted.
GPU utilization snapshots are appended to <run>/gpu_logs/sched_gpu.csv.

Job list: python scripts/schedule_eagle_ptq_qat_jobs.py --run-dir <rd>
          [--phase train|eval|all] [--dry-run]
"""
import argparse, json, os, subprocess, sys, time

ROOT = "/home/thahn1230/eagle_spinquant_w4a4"
GPUS = [0, 1, 2, 3, 4, 5]
PY = sys.executable


def build_jobs(rd, alpha_fp16, alpha_int4, with_sensitivity=False):
    T = f"{PY} scripts/train_eagle_draft_int4_qat.py --run-dir {rd}"
    TC = f"{PY} scripts/train_eagle_draft_fp16_control.py --run-dir {rd}"
    E = (f"{PY} scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
         f"--n-prompts 80 --datasets mtbench")
    XD = (f"{PY} scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
          f"--datasets sharegpt:80,c4:200,gsm8k:200,humaneval:200")
    ck = f"{rd}/ckpts"
    jobs = []

    def J(name, cmd, deps=()):
        jobs.append(dict(name=name, cmd=cmd, deps=list(deps)))

    # ---- trainings (9); C6/C8 first (critical path: C6 -> C7b) ----
    J("train_C6", f"{TC} --arm C6 --seed 0")
    J("train_C8", f"{TC} --arm C8 --seed 0")
    for s in (0, 1, 2):
        J(f"train_C3_s{s}", f"{T} --arm C3 --seed {s} "
          f"--alpha {alpha_fp16}")
        J(f"train_C7_s{s}", f"{T} --arm C7 --seed {s} "
          f"--alpha {alpha_int4}")
    J("train_C7b", f"{T} --arm C7b --seed 0 --alpha {alpha_int4} "
      f"--init-sd {ck}/C6_s0_last.pt", deps=["train_C6"])

    # ---- deployment evals (primary mtbench) ----
    for s in (0, 1, 2):
        J(f"eval_C3_s{s}", f"{E} --target fp16 --draft-cfg d4p3_deploy "
          f"--alpha {alpha_fp16} --draft-sd {ck}/C3_s{s}_last.pt "
          f"--tag C3_s{s}_qat", deps=[f"train_C3_s{s}"])
        J(f"eval_C7_s{s}", f"{E} --target int4 --draft-cfg d4p3_deploy "
          f"--alpha {alpha_int4} --draft-sd {ck}/C7_s{s}_last.pt "
          f"--tag C7_s{s}_qat", deps=[f"train_C7_s{s}"])
    J("eval_C7b", f"{E} --target int4 --draft-cfg d4p3_deploy "
      f"--alpha {alpha_int4} --draft-sd {ck}/C7b_s0_last.pt "
      f"--tag C7b_qat", deps=["train_C7b"])
    J("eval_C8", f"{E} --target fp16 --draft-cfg fp16_deploy "
      f"--draft-sd {ck}/C8_s0_last.pt --tag C8_fp16retrain",
      deps=["train_C8"])
    J("eval_C6", f"{E} --target int4 --draft-cfg fp16_deploy "
      f"--draft-sd {ck}/C6_s0_last.pt --tag C6_tgtadapt",
      deps=["train_C6"])
    # derived: retrain-then-PTQ and teacher-mismatch cells
    J("eval_C9", f"{E} --target fp16 --draft-cfg d4p3_deploy "
      f"--alpha {alpha_fp16} --draft-sd {ck}/C8_s0_last.pt "
      f"--tag C9_c8ptq", deps=["train_C8"])
    J("eval_C10", f"{E} --target int4 --draft-cfg d4p3_deploy "
      f"--alpha {alpha_int4} --draft-sd {ck}/C6_s0_last.pt "
      f"--tag C10_c6ptq", deps=["train_C6"])
    J("eval_C13", f"{E} --target int4 --draft-cfg fp16_deploy "
      f"--draft-sd {ck}/C8_s0_last.pt --tag C13_c8_at_int4",
      deps=["train_C8"])
    J("eval_C14", f"{E} --target int4 --draft-cfg d4p3_deploy "
      f"--alpha {alpha_int4} --draft-sd {ck}/C3_s0_last.pt "
      f"--tag C14_c3_at_int4", deps=["train_C3_s0"])

    # ---- calibrated-alpha strict-PTQ primary (C5 rerun at alpha*) ----
    J("eval_C5_recal", f"{E} --target int4 --draft-cfg d4p3 "
      f"--alpha {alpha_int4} --tag C5_int4_d4p3")
    # ---- complete the 9-point alpha candidate grid on the calib pool ----
    CALE = (f"{PY} scripts/eval_eagle_acceptance_length.py --run-dir {rd}"
            f" --n-prompts 20 --datasets c4 --pool calib")
    for tgt in ("fp16", "int4"):
        for a in (8, 11.3137085, 90.509668, 128):
            J(f"cal_{tgt}_a{a}", f"{CALE} --target {tgt} --draft-cfg "
              f"d4p3 --alpha {a} --tag CAL_{tgt}_a{a}")

    # ---- oracle ceilings + speculative fidelity ----
    for tgt in ("fp16", "int4"):
        J(f"oracle_{tgt}", f"{PY} scripts/eval_eagle_oracle_ceiling.py "
          f"--run-dir {rd} --target {tgt} --datasets mtbench "
          f"--n-prompts 80")
        J(f"fidelity_{tgt}",
          f"{PY} scripts/eval_eagle_speculative_fidelity.py "
          f"--run-dir {rd} --target {tgt} --n-prompts 40")

    # ---- cross-domain for the key configs ----
    xd = [("C1_fp16_stock", "fp16", "stock", None, None),
          ("C2_fp16_d4p3", "fp16", "d4p3", alpha_fp16, None),
          ("C4_int4_stockrestored", "int4", "stock", None, None),
          ("C5_int4_d4p3", "int4", "d4p3", alpha_int4, None),
          ("C3_s0_qat", "fp16", "d4p3_deploy", alpha_fp16,
           f"{ck}/C3_s0_last.pt"),
          ("C7_s0_qat", "int4", "d4p3_deploy", alpha_int4,
           f"{ck}/C7_s0_last.pt"),
          ("C8_fp16retrain", "fp16", "fp16_deploy", None,
           f"{ck}/C8_s0_last.pt"),
          ("C6_tgtadapt", "int4", "fp16_deploy", None,
           f"{ck}/C6_s0_last.pt")]
    for tag, tgt, dc, al, sd in xd:
        cmd = f"{XD} --target {tgt} --draft-cfg {dc} --tag {tag}"
        if al is not None:
            cmd += f" --alpha {al}"
        deps = []
        if sd is not None:
            cmd += f" --draft-sd {sd}"
            deps = [f"train_{tag.split('_qat')[0]}"
                    if "qat" in tag else
                    ("train_C8" if "C8" in tag else "train_C6")]
        J(f"xd_{tag}", cmd, deps=deps)

    if with_sensitivity:
        # (a) best-val checkpoint deployments (labeled sensitivity: the
        # primary uses the final checkpoint per original EAGLE semantics)
        best = [("C3_s0", "fp16", "d4p3_deploy", alpha_fp16),
                ("C3_s1", "fp16", "d4p3_deploy", alpha_fp16),
                ("C3_s2", "fp16", "d4p3_deploy", alpha_fp16),
                ("C7_s0", "int4", "d4p3_deploy", alpha_int4),
                ("C7_s1", "int4", "d4p3_deploy", alpha_int4),
                ("C7_s2", "int4", "d4p3_deploy", alpha_int4),
                ("C7b_s0", "int4", "d4p3_deploy", alpha_int4),
                ("C8_s0", "fp16", "fp16_deploy", None),
                ("C6_s0", "int4", "fp16_deploy", None)]
        for t, tgt, dc, al in best:
            dep = {"C8_s0": "train_C8", "C6_s0": "train_C6",
                   "C7b_s0": "train_C7b"}.get(t, f"train_{t}")
            cmd = (f"{E} --target {tgt} --draft-cfg {dc} "
                   f"--draft-sd {ck}/{t}_best.pt --tag {t}_bestval")
            if al is not None:
                cmd += f" --alpha {al}"
            J(f"evalbest_{t}", cmd, deps=[dep])
        # (b) fine-tuning-LR sensitivity trainings (NOT primary; probes
        # whether the faithful from-scratch recipe (lr 3e-5, warmup 2000)
        # is what degrades a converged init at this budget)
        J("train_C3lr_s0", f"{T} --arm C3 --seed 0 --alpha {alpha_fp16} "
          f"--lr 3e-6 --warmup 200 --steps 3000 --tag C3lr_s0")
        J("train_C7lr_s0", f"{T} --arm C7 --seed 0 --alpha {alpha_int4} "
          f"--lr 3e-6 --warmup 200 --steps 3000 --tag C7lr_s0")
        J("eval_C3lr_s0", f"{E} --target fp16 --draft-cfg d4p3_deploy "
          f"--alpha {alpha_fp16} --draft-sd {ck}/C3lr_s0_last.pt "
          f"--tag C3lr_s0_qat", deps=["train_C3lr_s0"])
        J("eval_C7lr_s0", f"{E} --target int4 --draft-cfg d4p3_deploy "
          f"--alpha {alpha_int4} --draft-sd {ck}/C7lr_s0_last.pt "
          f"--tag C7lr_s0_qat", deps=["train_C7lr_s0"])
    return jobs


def free_gpus(busy):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    free = []
    for ln in out.stdout.strip().splitlines():
        i, mem = [int(x) for x in ln.split(",")]
        if i in GPUS and i not in busy and mem < 1024:
            free.append(i)
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha-fp16", type=float, default=45.254834)
    ap.add_argument("--alpha-int4", type=float, default=32.0)
    ap.add_argument("--phase", default="all",
                    choices=["train", "eval", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-sensitivity", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    rd = args.run_dir
    jobs = build_jobs(rd, args.alpha_fp16, args.alpha_int4,
                      with_sensitivity=args.with_sensitivity)
    if args.phase == "train":
        jobs = [j for j in jobs if j["name"].startswith("train_")]
    elif args.phase == "eval":
        pass  # eval jobs depend on train jobs; keep all, trains may be done
    state_p = os.path.join(rd, "manifests", "scheduler_state.json")
    state = {}
    if os.path.exists(state_p):
        state = json.load(open(state_p))
    for j in jobs:
        j["state"] = state.get(j["name"], {}).get("state", "pending")
        if j["state"] in ("running", "blocked", "failed"):
            # a restarted scheduler holds no Popen for running jobs, and
            # failed jobs may have been fixed since; all jobs are
            # idempotent (evals skip existing shards, trainer skips on
            # final manifest) so re-queue them
            j["state"] = "pending"
    if args.dry_run:
        for j in jobs:
            print(f"{j['state']:9s} {j['name']}: {j['cmd']}"
                  f"  deps={j['deps']}")
        return 0
    os.makedirs(os.path.join(rd, "gpu_logs"), exist_ok=True)
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    gpu_log = open(os.path.join(rd, "gpu_logs", "sched_gpu.csv"), "a")
    running = {}          # name -> (Popen, gpu, t_start)
    by_name = {j["name"]: j for j in jobs}

    def save():
        json.dump({j["name"]: {"state": j["state"]} for j in jobs},
                  open(state_p, "w"), indent=1)

    def deps_ok(j):
        return all(by_name[d]["state"] == "done" for d in j["deps"]
                   if d in by_name)

    print(f"[sched] {len(jobs)} jobs "
          f"({sum(j['state'] == 'done' for j in jobs)} already done)")
    while True:
        # reap
        for name in list(running):
            p, g, ts = running[name]
            rc = p.poll()
            if rc is None:
                continue
            j = by_name[name]
            j["state"] = "done" if rc == 0 else "failed"
            hrs = (time.time() - ts) / 3600
            print(f"[sched] {name} -> {j['state']} (rc={rc}, "
                  f"{hrs:.2f}h, gpu{g})", flush=True)
            del running[name]
            save()
        # snapshot GPU state
        snap = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip() \
            .replace("\n", ";")
        gpu_log.write(f"{int(time.time())},{snap}\n")
        gpu_log.flush()
        # launch
        busy = {g for (_, g, _) in running.values()}
        for g in free_gpus(busy):
            nxt = next((j for j in jobs if j["state"] == "pending"
                        and deps_ok(j)), None)
            if nxt is None:
                break
            log = os.path.join(rd, "logs", f"sched_{nxt['name']}.log")
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES=str(g),
                       TMPDIR="/data/thahn1230/tmp",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            p = subprocess.Popen(
                nxt["cmd"], shell=True, cwd=ROOT, env=env,
                stdout=open(log, "w"), stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            nxt["state"] = "running"
            running[nxt["name"]] = (p, g, time.time())
            print(f"[sched] LAUNCH {nxt['name']} on gpu{g} "
                  f"pid={p.pid}", flush=True)
            save()
        if not running and all(j["state"] in ("done", "failed")
                               for j in jobs):
            break
        # deadlock guard: pending jobs whose deps failed
        for j in jobs:
            if j["state"] == "pending" and any(
                    by_name.get(d, {}).get("state") == "failed"
                    for d in j["deps"]):
                j["state"] = "blocked"
                print(f"[sched] BLOCKED {j['name']} (failed dep)")
                save()
        if all(j["state"] in ("done", "failed", "blocked")
               for j in jobs) and not running:
            break
        time.sleep(args.poll)
    n_done = sum(j["state"] == "done" for j in jobs)
    print(f"[sched] FINISHED: {n_done}/{len(jobs)} done, "
          f"{[j['name'] for j in jobs if j['state'] != 'done']}"
          f" not done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
