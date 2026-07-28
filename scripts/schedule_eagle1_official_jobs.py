#!/usr/bin/env python
"""Dynamic GPU scheduler for the official from-scratch study phase 2+
(same engine pattern as the validated PTQ-vs-QAT scheduler; GPUs 0-7,
foreign processes respected via the <1 GiB free check).

DAG: alpha_select (waits for the 18 running sweep shards) ->
P0/P1/P2 PTQ evals + LR pilots (runtime-read alphas) -> pilot calib
evals -> lr_select -> 3-seed QAT x Q0/Q1 -> per-seed final evals +
best-ckpt selection -> multistep pilots (M1, M2g08) + evals ->
first/recurrent audits, fidelity, step-0 parity, cross-domain.

State: <run>/manifests/of_scheduler_state.json (restart-safe; running/
failed re-queued; jobs idempotent).
"""
import argparse, json, os, subprocess, sys, time

ROOT = "/home/thahn1230/eagle_spinquant_w4a4"
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PY = sys.executable
ANCHOR = "checkpoints/eagle1_fresh_fp16_anchor/anchor.pt"


def rd_alpha(rd, cond):
    """Shell fragment: read the calibrated alpha for a condition."""
    return (f"$({PY} -c \"import json;print(json.load(open('"
            f"{rd}/tables/alpha_calibration.json'))['{cond}']"
            f"['selected_alpha'])\")")


def rd_lr(rd, var):
    return (f"$({PY} -c \"import json;print(json.load(open('"
            f"{rd}/tables/lr_pilot.json'))['{var}']['selected_lr'])\")")


def build_jobs(rd):
    EV = (f"{PY} scripts/eval_eagle1_ptq_vs_qat.py --run-dir {rd} "
          f"--sd {ANCHOR}")
    XD = "--datasets sharegpt:80,c4:200,gsm8k:200,humaneval:164"
    jobs = []

    def J(name, cmd, deps=()):
        jobs.append(dict(name=name, cmd=cmd, deps=list(deps)))

    J("alpha_select",
      f"bash -c 'until [ $(ls {rd}/shards/al__ALCAL_* 2>/dev/null | "
      f"wc -l) -ge 18 ]; do sleep 120; done; {PY} "
      f"scripts/calibrate_eagle1_p3_alpha.py --stage select "
      f"--run-dir {rd}'")
    # PTQ baselines
    J("p0_fp16", f"{EV} --cell C2 --target fp16 --tag OF_P0_fp16")
    J("p0_int4", f"{EV} --cell C2 --target int4 --tag OF_P0_int4")
    aA, aC = rd_alpha(rd, "A"), rd_alpha(rd, "C")
    J("p1_fp16", f"bash -c '{EV} --cell C3 --target fp16 --alpha {aA} "
      f"--tag OF_P1_fp16'", deps=["alpha_select"])
    J("p1_int4", f"bash -c '{EV} --cell C3 --target int4 --alpha {aC} "
      f"--tag OF_P1_int4'", deps=["alpha_select"])
    lk = ("$(cat runs/LK_RUN_DIR)/rotations/LK2_HYBRID_s2.pt")
    J("p2_int4", f"bash -c '{EV} --cell C4 --target int4 --alpha {aC} "
      f"--ckpt {lk} --tag OF_P2_int4'", deps=["alpha_select"])
    # LR pilots (1000 steps)
    for var, cond in (("Q0", "A"), ("Q1", "C")):
        for lr in ("1e-06", "3e-06", "1e-05"):
            t = f"{var}p_lr{lr}"
            J(f"pilot_{t}", f"bash -c '{PY} "
              f"scripts/train_eagle1_single_step_qat.py --variant {var} "
              f"--anchor {ANCHOR} --alpha {rd_alpha(rd, cond)} "
              f"--lr {lr} --steps 1000 --seed 0 --run-dir {rd} "
              f"--tag {t}'", deps=["alpha_select"])
            J(f"pev_{t}", f"bash -c '{PY} "
              f"scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
              f"--target {'fp16' if var == 'Q0' else 'int4'} "
              f"--draft-cfg d4p3_deploy --alpha {rd_alpha(rd, cond)} "
              f"--draft-sd {rd}/ckpts/{t}_last.pt --datasets c4:20 "
              f"--pool calib --tag PILOT_{t}'", deps=[f"pilot_{t}"])
    J("lr_select", f"{PY} scripts/analyze_eagle1_lr_pilot.py "
      f"--run-dir {rd}",
      deps=[f"pev_{v}p_lr{l}" for v in ("Q0", "Q1")
            for l in ("1e-06", "3e-06", "1e-05")])
    # 3-seed QAT per variant
    for var, cond, tgt in (("Q0", "A", "fp16"), ("Q1", "C", "int4")):
        for s in (0, 1, 2):
            t = f"{var}_s{s}"
            J(f"qat_{t}", f"bash -c '{PY} "
              f"scripts/train_eagle1_single_step_qat.py --variant {var} "
              f"--anchor {ANCHOR} --alpha {rd_alpha(rd, cond)} "
              f"--lr {rd_lr(rd, var)} --steps 3000 --seed {s} "
              f"--save-ckpt-every 500 --run-dir {rd} --tag {t}'",
              deps=["lr_select"])
            J(f"cksel_{t}", f"bash -c '{PY} "
              f"scripts/select_best_qat_ckpt.py --tag {t} "
              f"--target {tgt} --alpha {rd_alpha(rd, cond)} "
              f"--run-dir {rd}'", deps=[f"qat_{t}"])
    # multistep pilots (Q1 primary, seed 0)
    for wts in ("M1", "M2g08"):
        t = f"MS{wts}_Q1_s0"
        J(f"ms_{wts}", f"bash -c '{PY} "
          f"scripts/train_eagle1_multistep_qat.py --variant Q1 "
          f"--weights {wts} --anchor {ANCHOR} "
          f"--alpha {rd_alpha(rd, 'C')} --lr {rd_lr(rd, 'Q1')} "
          f"--steps 3000 --seed 0 --run-dir {rd} --tag {t}'",
          deps=["lr_select"])
        J(f"msev_{wts}", f"bash -c '{PY} "
          f"scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
          f"--target int4 --draft-cfg d4p3_deploy "
          f"--alpha {rd_alpha(rd, 'C')} "
          f"--draft-sd {rd}/ckpts/{t}_last.pt --datasets mtbench "
          f"--tag {t}_final'", deps=[f"ms_{wts}"])
    # audits / parity / fidelity
    J("firstrec_int4", f"bash -c '{PY} "
      f"scripts/audit_eagle1_first_recurrent.py --target int4 "
      f"--anchor {ANCHOR} --alpha {rd_alpha(rd, 'C')} --run-dir {rd}'",
      deps=["alpha_select"])
    J("firstrec_fp16", f"bash -c '{PY} "
      f"scripts/audit_eagle1_first_recurrent.py --target fp16 "
      f"--anchor {ANCHOR} --alpha {rd_alpha(rd, 'A')} --run-dir {rd}'",
      deps=["alpha_select"])
    J("parity_anchor_id", f"bash -c '{PY} "
      f"scripts/check_qat_deploy_parity.py --run-dir {rd} "
      f"--mode identity --alpha {rd_alpha(rd, 'A')} "
      f"--draft-sd {ANCHOR}'", deps=["alpha_select"])
    J("parity_anchor_rot", f"bash -c '{PY} "
      f"scripts/check_qat_deploy_parity.py --run-dir {rd} "
      f"--mode gamma_R1 --alpha {rd_alpha(rd, 'C')} "
      f"--draft-sd {ANCHOR}'", deps=["alpha_select"])
    J("fidelity_fp16", f"{PY} scripts/eval_eagle_speculative_fidelity.py"
      f" --run-dir {rd} --target fp16 --n-prompts 40")
    J("fidelity_int4", f"{PY} scripts/eval_eagle_speculative_fidelity.py"
      f" --run-dir {rd} --target int4 --n-prompts 40")
    # cross-domain for P1 cells + best QAT seed
    J("xd_p1_fp16", f"bash -c '{EV} --cell C3 --target fp16 "
      f"--alpha {aA} --tag OF_P1_fp16 {XD}'", deps=["p1_fp16"])
    J("xd_p1_int4", f"bash -c '{EV} --cell C3 --target int4 "
      f"--alpha {aC} --tag OF_P1_int4 {XD}'", deps=["p1_int4"])
    J("xd_q0_s0", f"bash -c '{PY} "
      f"scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
      f"--target fp16 --draft-cfg d4p3_deploy --alpha {aA} "
      f"--draft-sd {rd}/ckpts/Q0_s0_last.pt --tag Q0_s0_final {XD}'",
      deps=["cksel_Q0_s0"])
    J("xd_q1_s0", f"bash -c '{PY} "
      f"scripts/eval_eagle_acceptance_length.py --run-dir {rd} "
      f"--target int4 --draft-cfg d4p3_deploy --alpha {aC} "
      f"--draft-sd {rd}/ckpts/Q1_s0_last.pt --tag Q1_s0_final {XD}'",
      deps=["cksel_Q1_s0"])
    return jobs


def free_gpus(busy):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True,
        text=True)
    free = []
    for ln in out.stdout.strip().splitlines():
        i, mem = [int(x) for x in ln.split(",")]
        if i in GPUS and i not in busy and mem < 1024:
            free.append(i)
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    rd = args.run_dir
    jobs = build_jobs(rd)
    state_p = os.path.join(rd, "manifests", "of_scheduler_state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    for j in jobs:
        j["state"] = state.get(j["name"], {}).get("state", "pending")
        if j["state"] in ("running", "blocked", "failed"):
            j["state"] = "pending"
    if args.dry_run:
        for j in jobs:
            print(f"{j['state']:8s} {j['name']}  deps={j['deps']}")
        return 0
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    os.makedirs(os.path.join(rd, "gpu_logs"), exist_ok=True)
    gpu_log = open(os.path.join(rd, "gpu_logs", "of_sched_gpu.csv"), "a")
    running, by = {}, {j["name"]: j for j in jobs}

    def save():
        json.dump({j["name"]: {"state": j["state"]} for j in jobs},
                  open(state_p, "w"), indent=1)

    def ok(j):
        return all(by[d]["state"] == "done" for d in j["deps"]
                   if d in by)

    print(f"[of-sched] {len(jobs)} jobs "
          f"({sum(j['state'] == 'done' for j in jobs)} done)")
    while True:
        for name in list(running):
            p, g, ts = running[name]
            rc = p.poll()
            if rc is None:
                continue
            by[name]["state"] = "done" if rc == 0 else "failed"
            print(f"[of-sched] {name} -> {by[name]['state']} (rc={rc}, "
                  f"{(time.time()-ts)/3600:.2f}h, gpu{g})", flush=True)
            del running[name]
            save()
        gpu_log.write(f"{int(time.time())}," + subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True).stdout.strip().replace("\n", ";") + "\n")
        gpu_log.flush()
        busy = {g for (_, g, _) in running.values()}
        for g in free_gpus(busy):
            nxt = next((j for j in jobs if j["state"] == "pending"
                        and ok(j)), None)
            if nxt is None:
                break
            log = os.path.join(rd, "logs", f"ofsched_{nxt['name']}.log")
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES=str(g),
                       TMPDIR="/data/thahn1230/tmp",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            p = subprocess.Popen(nxt["cmd"], shell=True, cwd=ROOT,
                                 env=env, stdout=open(log, "w"),
                                 stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,
                                 start_new_session=True)
            nxt["state"] = "running"
            running[nxt["name"]] = (p, g, time.time())
            print(f"[of-sched] LAUNCH {nxt['name']} gpu{g} "
                  f"pid={p.pid}", flush=True)
            save()
        for j in jobs:
            if j["state"] == "pending" and any(
                    by.get(d, {}).get("state") == "failed"
                    for d in j["deps"]):
                j["state"] = "blocked"
                save()
        if not running and all(j["state"] in
                               ("done", "failed", "blocked")
                               for j in jobs):
            break
        time.sleep(args.poll)
    print(f"[of-sched] FINISHED: "
          f"{sum(j['state'] == 'done' for j in jobs)}/{len(jobs)} done; "
          f"not-done: {[j['name'] for j in jobs if j['state'] != 'done']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
