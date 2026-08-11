#!/usr/bin/env python
"""Idle-gap probe for the GS+R5 campaign watchdog.

Prints one line: "<free_gpus> <ready_jobs> <running> <pending>" where
ready_jobs counts queue entries that are pending, dependency-satisfied,
wait-files present and NOT already satisfied by an existing artifact —
i.e. work the scheduler could start right now. A real idle gap is
free_gpus > 0 AND ready_jobs > 0; anything else (e.g. 5 GPUs free while
only 3 QAT jobs remain) is expected serialization, not a stall.
"""
import json, os, subprocess, sys

rd = sys.argv[1]
qp = os.path.join(rd, "scheduler", "queue.jsonl")
dp = os.path.join(rd, "scheduler", "done.json")
sp = os.path.join(rd, "scheduler", "state.json")
jobs = [json.loads(l) for l in open(qp) if l.strip()]
done = set(json.load(open(dp))) if os.path.exists(dp) else set()
state = json.load(open(sp)) if os.path.exists(sp) else {}
running = set(state.get("running", {}))

ready = 0
for j in jobs:
    jid = j["id"]
    if jid in done or jid in running:
        continue
    if j.get("artifact") and os.path.exists(j["artifact"]):
        continue
    if not all(d in done for d in j.get("deps", [])):
        continue
    if not all(os.path.exists(w) for w in j.get("wait_files", [])):
        continue
    ready += 1

out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.split()
# a GPU assigned to a running job is NOT free even while its process is
# still loading and reports ~0 MiB used
owned = {g for gs in state.get("running", {}).values() for g in gs}
free = sum(1 for i, x in enumerate(out)
           if x.isdigit() and int(x) < 1024 and i not in owned)
pending = len(jobs) - len(done) - len(running)
print(f"{free} {ready} {len(running)} {pending}")
