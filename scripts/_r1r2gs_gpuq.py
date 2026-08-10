#!/usr/bin/env python
"""Tiny per-GPU job queue for the GS R1/R2 study.

Jobs file: one shell command per line; the literal string {GPU} is
replaced by the assigned physical GPU index, and CUDA_VISIBLE_DEVICES is
set to that index (the repo's scripts assert a single-digit CVD). Blank
lines / #-comments skipped. Each job's output goes to
<log-dir>/q__<idx>__<slug>.log. Exit code = number of failed jobs.
"""
import argparse, os, re, subprocess, sys, threading, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()
    jobs = [ln.strip() for ln in open(args.jobs)
            if ln.strip() and not ln.strip().startswith("#")]
    gpus = [g for g in args.gpus.split(",") if g != ""]
    os.makedirs(args.log_dir, exist_ok=True)
    lock = threading.Lock()
    queue = list(enumerate(jobs))
    fails = []

    def gpu_free(gpu, thresh_mb=2000):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(gpu)],
                capture_output=True, text=True, timeout=15)
            return int(out.stdout.strip().splitlines()[0]) < thresh_mb
        except Exception:
            return True

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                idx, cmd = queue.pop(0)
            # wait until the physical GPU is actually free (another
            # process — e.g. an early-started training job — may still
            # own it); avoids VRAM collisions on shared queues
            waited = 0
            while not gpu_free(gpu):
                if waited == 0:
                    print(f"[gpuq] gpu{gpu} busy — waiting before "
                          f"job{idx}", flush=True)
                time.sleep(30)
                waited += 30
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", cmd.split("--tag")[-1]
                          if "--tag" in cmd else cmd.split()[-1])[:60]
            log = os.path.join(args.log_dir, f"q__{idx:03d}__{slug}.log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            full = cmd.replace("{GPU}", str(gpu))
            t0 = time.time()
            print(f"[gpuq] gpu{gpu} job{idx}: {full[:160]}", flush=True)
            with open(log, "w") as lf:
                lf.write(f"# gpu={gpu}\n# {full}\n")
                lf.flush()
                r = subprocess.run(full, shell=True, stdout=lf,
                                   stderr=subprocess.STDOUT, env=env)
            dt = time.time() - t0
            status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
            print(f"[gpuq] gpu{gpu} job{idx} {status} {dt/60:.1f}min "
                  f"-> {log}", flush=True)
            if r.returncode != 0:
                with lock:
                    fails.append((idx, full, log))

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for idx, cmd, log in fails:
        print(f"[gpuq] FAILED job{idx}: {cmd} (log {log})", flush=True)
    print(f"[gpuq] done: {len(jobs) - len(fails)}/{len(jobs)} ok",
          flush=True)
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
