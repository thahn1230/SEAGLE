#!/usr/bin/env python
"""PMG persistent GPU scheduler.

Keeps all 8 GPUs saturated from a jsonl job queue; coexists with
externally-launched jobs (skips GPUs with >1 GiB used). Jobs support
dependencies (on job ids) and wait_files (existence preconditions).
Failures are logged loudly and retried once; the queue file can be
APPENDED to while the scheduler runs (it re-reads every poll).

Job line schema (queue.jsonl):
  {"id": str, "cmd": str (shell), "gpus": int (default 1),
   "deps": [ids], "wait_files": [paths], "prio": int (higher first),
   "log": path}

State: state.json heartbeat every poll; events.log one line per
transition (START/DONE/FAIL/RETRY/GAVE-UP) — monitors tail this.
"""
import json, os, shlex, subprocess, sys, time

POLL = 20
MEM_FREE_MIB = 1024
FREE_STREAK = 3   # consecutive free polls before a GPU is claimable


def gpu_free(idx, owned):
    if idx in owned:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out) < MEM_FREE_MIB
    except Exception:
        return False


def main():
    rd = sys.argv[1]
    qpath = os.path.join(rd, "scheduler", "queue.jsonl")
    spath = os.path.join(rd, "scheduler", "state.json")
    epath = os.path.join(rd, "scheduler", "events.log")
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    open(qpath, "a").close()

    def ev(msg):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
        with open(epath, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    donefile = os.path.join(rd, "scheduler", "done.json")
    status = {}                    # id -> pending|running|done|failed|gaveup
    if os.path.exists(donefile):
        for jid in json.load(open(donefile)):
            status[jid] = "done"
    tries = {}
    running = {}                   # id -> (Popen, [gpus], job)
    free_streak = {g: 0 for g in range(8)}
    ev("SCHEDULER-START")
    while True:
        jobs = []
        for ln in open(qpath):
            ln = ln.strip()
            if ln:
                jobs.append(json.loads(ln))
        byid = {j["id"]: j for j in jobs}
        for j in jobs:
            status.setdefault(j["id"], "pending")

        for jid in list(running):
            pr, gpus, j = running[jid]
            rc = pr.poll()
            if rc is None:
                continue
            del running[jid]
            if rc == 0:
                status[jid] = "done"
                json.dump([k for k, v in status.items()
                           if v == "done"], open(donefile, "w"))
                ev(f"DONE {jid}")
            else:
                tries[jid] = tries.get(jid, 0) + 1
                if tries[jid] <= 1:
                    status[jid] = "pending"
                    ev(f"FAIL {jid} rc={rc} -> RETRY")
                else:
                    status[jid] = "gaveup"
                    ev(f"FAIL {jid} rc={rc} -> GAVE-UP (see {j.get('log')})")

        owned = [g for _, gpus, _ in running.values() for g in gpus]
        for g in range(8):
            free_streak[g] = (free_streak[g] + 1
                              if gpu_free(g, owned) else 0)
        free = [g for g in range(8) if free_streak[g] >= FREE_STREAK]

        # idempotence: a job whose artifact already exists is done
        for j in jobs:
            if status[j["id"]] == "pending" and j.get("artifact") \
                    and os.path.exists(j["artifact"]):
                status[j["id"]] = "done"
                json.dump([k for k, v in status.items()
                           if v == "done"], open(donefile, "w"))
                ev(f"SKIP-ARTIFACT {j['id']}")
        ready = [j for j in jobs if status[j["id"]] == "pending"
                 and all(status.get(d) == "done" for d in
                         j.get("deps", []))
                 and all(os.path.exists(w) for w in
                         j.get("wait_files", []))]
        ready.sort(key=lambda j: -j.get("prio", 0))
        for j in ready:
            need = j.get("gpus", 1)
            if need == 0:
                asg = []
            elif len(free) >= need:
                asg, free = free[:need], free[need:]
            else:
                continue
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES=",".join(map(str, asg)))
            lf = open(j.get("log", os.path.join(
                rd, "logs", f"sched_{j['id']}.log")), "a")
            pr = subprocess.Popen(["bash", "-c", j["cmd"]], env=env,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  cwd="/home/thahn1230/dflash_workspace/dflash")
            running[j["id"]] = (pr, asg, j)
            for g in asg:
                free_streak[g] = 0
            status[j["id"]] = "running"
            ev(f"START {j['id']} gpus={asg}")

        json.dump(dict(ts=time.time(),
                       running={k: v[1] for k, v in running.items()},
                       counts={s: sum(1 for v in status.values()
                                      if v == s)
                               for s in ("pending", "running", "done",
                                         "failed", "gaveup")}),
                  open(spath, "w"))
        if jobs and all(status[j["id"]] in ("done", "gaveup")
                        for j in jobs) and not running:
            ev("QUEUE-DRAINED (idle, waiting for appends)")
            time.sleep(POLL * 3)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
