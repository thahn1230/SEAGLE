#!/usr/bin/env python
"""Continuous dependency-aware GPU dispatcher for the GS R1/R2 study.

Keeps every GPU busy for the whole post-training pipeline. Each job
declares how many GPUs it needs (RCAL captures need 2: deployed + FP16
reference in lockstep), what must exist first (deps), and a completion
MARKER file — so the dispatcher is idempotent and can be restarted at
any point without redoing finished work.

Job waves (later waves become eligible as their deps land):
  W1 eval      : 9 trained ckpts x 4 datasets  (1 GPU)
  W2 rcal      : A0/A1/A2/A3 same-proposal replay, mtbench-80 (2 GPUs)
  W3 runtime   : per-arm fake-quant timing (1 GPU)
  W4 mechanism : R2 geometry / quant proxies / per-depth overlap (1 GPU)
  W5 composed  : A4 = best R1_D + best R2_D, no joint training (1 GPU)
  W6 randctrl  : matched-generator-norm random R2 controls (1 GPU)

Median-val-seed selection (the pre-registered primary per arm) and the
composed/random control checkpoints are built by the dispatcher itself
once their inputs exist, on CPU, between dispatch polls.
"""
import argparse, glob, json, os, subprocess, sys, time

import torch

ALPHA = "32.89964245299412"          # 4096**0.42 (canonical T4 beta_GS)
DSETS = [("mtbench", "80"), ("gsm8k", "200"), ("sharegpt", "80"),
         ("humaneval", "164")]
ARMS = ("A1", "A2", "A3")
SEEDS = (1001, 1002, 1003)
FREE_MB = 2000


def gpu_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    return {int(l.split(",")[0]): int(l.split(",")[1])
            for l in out.strip().splitlines()}


class Dispatcher:
    def __init__(self, rd, gpus, poll=30):
        self.rd = rd
        self.gpus = gpus
        self.poll = poll
        self.claimed = set()
        self.running = []            # (job, popen, gpus)
        self.done = set()
        self.failed = {}

    def free_gpus(self):
        mem = gpu_mem()
        return [g for g in self.gpus
                if g not in self.claimed and mem.get(g, 99999) < FREE_MB]

    @staticmethod
    def externally_running(job):
        """True when an equivalent process is already alive (e.g. started
        by a previous dispatcher instance) — makes restarting the
        dispatcher safe while work is in flight. Matches on the script
        name plus --tag/--datasets values, which identify a job uniquely.
        """
        t = job["cmd"].split()
        script = next((os.path.basename(x) for x in t
                       if x.startswith("scripts/")), None)
        if script is None:
            return False
        need = [script]
        for flag in ("--tag", "--datasets"):
            if flag in t:
                need.append(f"{flag} {t[t.index(flag) + 1]}")
        try:
            ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True,
                                text=True, timeout=20).stdout
        except Exception:
            return False
        return any(all(n in line for n in need)
                   for line in ps.splitlines())

    def launch(self, job, gpus):
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus))
        log = os.path.join(self.rd, "logs", f"disp__{job['id']}.log")
        lf = open(log, "w")
        lf.write(f"# gpus={gpus}\n# {job['cmd']}\n")
        lf.flush()
        p = subprocess.Popen(job["cmd"], shell=True, stdout=lf,
                             stderr=subprocess.STDOUT, env=env,
                             cwd="/home/thahn1230/SEAGLE")
        self.claimed.update(gpus)
        self.running.append((job, p, gpus, time.time(), lf))
        print(f"[disp] gpu{gpus} <- {job['id']}", flush=True)

    def reap(self):
        still = []
        for job, p, gpus, t0, lf in self.running:
            rc = p.poll()
            if rc is None:
                still.append((job, p, gpus, t0, lf))
                continue
            lf.close()
            self.claimed.difference_update(gpus)
            dt = (time.time() - t0) / 60
            ok = rc == 0 and (job["marker"] is None
                              or os.path.exists(job["marker"]))
            print(f"[disp] {job['id']} {'OK' if ok else f'FAIL(rc={rc})'} "
                  f"{dt:.1f}min", flush=True)
            if ok:
                self.done.add(job["id"])
            else:
                self.failed[job["id"]] = rc
        self.running = still

    def run(self, build_jobs, max_idle_polls=200):
        idle = 0
        while True:
            self.reap()
            jobs = build_jobs(self)
            pending = [j for j in jobs
                       if j["id"] not in self.done
                       and j["id"] not in self.failed
                       and not any(r[0]["id"] == j["id"]
                                   for r in self.running)
                       and (j["marker"] is None
                            or not os.path.exists(j["marker"]))
                       and all(os.path.exists(d) for d in j["deps"])]
            for j in pending:
                fg = self.free_gpus()
                if len(fg) < j["gpus"]:
                    continue
                if self.externally_running(j):
                    continue         # a previous instance owns this job
                self.launch(j, fg[:j["gpus"]])
                time.sleep(20)       # let it claim VRAM before rescan
            if not self.running and not pending:
                remaining = [j for j in jobs
                             if j["id"] not in self.done
                             and j["id"] not in self.failed
                             and (j["marker"] is None
                                  or not os.path.exists(j["marker"]))]
                if not remaining:
                    print("[disp] ALL_JOBS_DONE", flush=True)
                    return 0
                idle += 1
                if idle > max_idle_polls:
                    print(f"[disp] STALLED, waiting on deps of "
                          f"{[j['id'] for j in remaining][:6]}", flush=True)
                    return 1
            else:
                idle = 0
            time.sleep(self.poll)


def ckpt(rd, arm, seed):
    return os.path.join(rd, "rotations", f"RD_GS_{arm}_s{seed}.pt")


def shard(rd, tag, ds):
    return os.path.join(rd, "shards", f"al__{tag}__int4__{ds}.csv")


def median_seed(rd, arm):
    """Pre-registered primary: seed with the MEDIAN best-val loss."""
    rows = []
    for s in SEEDS:
        p = ckpt(rd, arm, s)
        if not (os.path.exists(p) and os.path.exists(p + ".sha256")):
            return None
        d = torch.load(p, map_location="cpu", weights_only=False)
        bv = (d["meta"].get("best_val") or {})
        rows.append((bv.get("val_loss", float("inf")), s, p))
    rows.sort()
    return rows[len(rows) // 2]


def build_controls(rd):
    """CPU-side artifact builds, PER ARM as soon as that arm's three
    seeds exist — so A1/A2 downstream work (RCAL, runtime, composed
    control) starts while a slower arm is still training."""
    meds = {a: median_seed(rd, a) for a in ARMS}
    ready = {a: v for a, v in meds.items() if v is not None}
    if ready:
        sel = os.path.join(rd, "tables", "median_seeds.json")
        prev = json.load(open(sel)) if os.path.exists(sel) else {}
        cur = {a: dict(seed=v[1], val_loss=v[0], ckpt=v[2])
               for a, v in ready.items()}
        if cur != prev:
            json.dump(cur, open(sel, "w"), indent=1)
            print(f"[disp] median seeds so far: "
                  f"{ {a: v['seed'] for a, v in cur.items()} }",
                  flush=True)
    if "A1" in ready and "A2" in ready:
        comp = os.path.join(rd, "rotations", "COMPOSED_A1R1_A2R2.pt")
        if not os.path.exists(comp):
            subprocess.run(
                [sys.executable, "scripts/make_r2_control_ckpts.py",
                 "--mode", "composed", "--a1-ckpt", ready["A1"][2],
                 "--a2-ckpt", ready["A2"][2], "--out-dir",
                 os.path.join(rd, "rotations")],
                cwd="/home/thahn1230/SEAGLE", check=False)
    if "A2" in ready:
        rand0 = os.path.join(rd, "rotations", "RANDR2_s7000.pt")
        if not os.path.exists(rand0):
            subprocess.run(
                [sys.executable, "scripts/make_r2_control_ckpts.py",
                 "--mode", "random", "--a2-ckpt", ready["A2"][2],
                 "--n", "3", "--out-dir",
                 os.path.join(rd, "rotations")],
                cwd="/home/thahn1230/SEAGLE", check=False)
    return meds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    rd = args.run_dir
    gpus = [int(g) for g in args.gpus.split(",")]

    def build_jobs(_d):
        jobs = []
        meds = build_controls(rd)
        # ---- W1: per-seed evals (deps: the checkpoint + its sha) -----
        for arm in ARMS:
            for s in SEEDS:
                c = ckpt(rd, arm, s)
                tag = f"{arm}_s{s}"
                for ds, n in DSETS:
                    jobs.append(dict(
                        id=f"eval_{tag}_{ds}", gpus=1,
                        deps=[c, c + ".sha256"],
                        marker=shard(rd, tag, ds),
                        cmd=(f"python scripts/eval_eagle_acceptance_length.py "
                             f"--target int4 --draft-cfg rot --ckpt {c} "
                             f"--alpha {ALPHA} --tag {tag} "
                             f"--datasets {ds}:{n} --max-new-tokens 128 "
                             f"--run-dir {rd}")))
        comp = os.path.join(rd, "rotations", "COMPOSED_A1R1_A2R2.pt")
        # ---- W2/W3: RCAL replay (2 GPUs) + runtime, per ready arm ----
        for arm in ARMS:
            if meds[arm] is None:
                continue
            ck_, tag = meds[arm][2], f"{arm}_MED"
            jobs.append(dict(
                id=f"rcal_{tag}", gpus=2, deps=[ck_],
                marker=os.path.join(rd, "cycles",
                                    f"cyc__{tag}__mtbench.jsonl"),
                cmd=(f"python scripts/capture_eagle_proposal_cycles.py "
                     f"--target int4 --draft-cfg rot --ckpt {ck_} "
                     f"--alpha {ALPHA} --tag {tag} --datasets mtbench "
                     f"--n-prompts 80 --run-dir {rd} "
                     f"--ref-device cuda:1")))
            jobs.append(dict(
                id=f"runtime_{tag}", gpus=1, deps=[ck_],
                marker=os.path.join(rd, "tables", f"runtime__{tag}.json"),
                cmd=(f"python scripts/bench_pmg_runtime.py --target int4 "
                     f"--draft-cfg rot_ep3p --ckpt {ck_} --alpha {ALPHA} "
                     f"--tag {tag} --run-dir {rd}")))
        # ---- W4: mechanism (needs all three arms + composed) ---------
        if all(meds[a] is not None for a in ARMS):
            m1, m2, m3 = (meds[a][2] for a in ARMS)
            jobs.append(dict(
                id="mechanism", gpus=1, deps=[m1, m2, m3, comp],
                marker=os.path.join(rd, "geometry", "r2_mechanism.json"),
                cmd=(f"python scripts/analyze_r2_mechanism.py "
                     f"--run-dir {rd} "
                     f"--ckpts A1={m1},A2={m2},A3={m3},A4COMP={comp}")))
        # ---- W5: composed-arm evals (needs A1+A2 only) ---------------
        for ds, n in DSETS:
            jobs.append(dict(
                id=f"eval_A4COMPOSED_{ds}", gpus=1, deps=[comp],
                marker=shard(rd, "A4_COMPOSED", ds),
                cmd=(f"python scripts/eval_eagle_acceptance_length.py "
                     f"--target int4 --draft-cfg rot --ckpt {comp} "
                     f"--alpha {ALPHA} --tag A4_COMPOSED "
                     f"--datasets {ds}:{n} --max-new-tokens 128 "
                     f"--run-dir {rd}")))
        # ---- W6: matched random-R2 controls (mtbench) ----------------
        for i in range(3):
            rp = os.path.join(rd, "rotations", f"RANDR2_s{7000 + i}.pt")
            jobs.append(dict(
                id=f"eval_RANDR2_s{7000 + i}", gpus=1, deps=[rp],
                marker=shard(rd, f"RANDR2_s{7000 + i}", "mtbench"),
                cmd=(f"python scripts/eval_eagle_acceptance_length.py "
                     f"--target int4 --draft-cfg rot --ckpt {rp} "
                     f"--alpha {ALPHA} --tag RANDR2_s{7000 + i} "
                     f"--datasets mtbench:80 --max-new-tokens 128 "
                     f"--run-dir {rd}")))
        return jobs

    d = Dispatcher(rd, gpus)
    rc = d.run(build_jobs)
    print(f"[disp] done={len(d.done)} failed={d.failed}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
