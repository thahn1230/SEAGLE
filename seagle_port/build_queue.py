"""Build the DFST scheduler queue (wave 1 + learned-R wave 2).

Wave 1 (runnable now, Hadamard R where rotation is needed for diagnostics):
  gateC, wc_stats, component sensitivity @T16, T16 grid row, A0 leftovers.
Wave 2 (wait_files = learned R.bin):
  gateB_lrn, interface arms A1/A2 (mtbench), folded grid rows T8/T4 x 4 ds.
Append-safe: scheduler re-reads queue.jsonl every poll; artifacts make
re-enqueueing idempotent.
"""
import json
import os
import sys

WS = "/home/thahn1230/dflash_workspace"
PY = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../venv/bin/python"
HAD = f"{WS}/outputs/rotations/llama31_hadamard/R.bin"
LRN = f"{WS}/outputs/rotations/llama31_w16a4kv16/R.bin"
DATASETS = [("mtbench", 80), ("gsm8k", 200), ("humaneval", 164),
            ("sharegpt", 80)]
COMPS = ["fc", "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"]


def main():
    rd = sys.argv[1]
    absrd = os.path.abspath(rd)
    jobs = []

    def add(jid, cmd, prio=5, deps=(), wait=(), artifact=None):
        jobs.append({"id": jid, "cmd": cmd, "gpus": 1, "deps": list(deps),
                     "wait_files": list(wait), "prio": prio,
                     "artifact": artifact,
                     "log": f"{absrd}/logs/sched_{jid}.log"})

    def ev(tag, tgt, iface, ds, n, rbin=None, extra="", prio=5, deps=(),
           wait=()):
        cmd = (f"{PY} -m seagle_port.eval_al --tag {tag} --target-mode {tgt}"
               f" --interface {iface} --dataset {ds} --max-samples {n}"
               f" --max-new-tokens 1024 --run-dir {absrd}")
        if rbin:
            cmd += f" --rbin {rbin}"
        cmd += extra
        add(f"{tag}__{ds}", cmd, prio=prio, deps=deps, wait=wait,
            artifact=f"{absrd}/shards/al__{tag}__{tgt}__{ds}.csv")

    # ---- wave 1
    add("gateC",
        f"{PY} -m seagle_port.gate_c --rbin {HAD} --run-dir {absrd}",
        prio=9, artifact=f"{absrd}/tables/gate_c.json")
    add("wc_stats",
        f"{PY} -m seagle_port.wc_stats --rbin {HAD} --run-dir {absrd}",
        prio=8, artifact=f"{absrd}/tables/wc_branch_stats.csv")
    for c in COMPS:
        ev(f"S16_{c}", "fp16", "stock", "mtbench", 40,
           extra=f" --draft-mode w4a4 --draft-components {c}", prio=7)
    ev("S16_fc_p2", "fp16", "stock", "mtbench", 40,
       extra=" --draft-mode w4a4 --draft-components fc --fc-p2", prio=7)
    ev("S16_full", "fp16", "stock", "mtbench", 40,
       extra=" --draft-mode w4a4", prio=7)
    for ds, n in DATASETS:
        ev("G_T16D8", "fp16", "stock", ds, n,
           extra=" --draft-mode w8a8", prio=6)
        ev("G_T16D4", "fp16", "stock", ds, n,
           extra=" --draft-mode w4a4", prio=6)
    ev("A0_T16D16", "fp16", "stock", "sharegpt", 80, prio=6)
    ev("A0_T16D16", "fp16", "stock", "math500", 100, prio=4)

    # ---- wave 2 (learned R)
    gb = " && ".join(
        f"{PY} -m seagle_port.gate_b --arm {a} --rbin {LRN} --run-dir "
        f"{absrd}_lrn" for a in ("fp16", "rot_fp16", "w8a8", "w4a4",
                                 "compare"))
    add("gateB_lrn", f"mkdir -p {absrd}_lrn/tables && {gb}", prio=9,
        wait=[LRN], artifact=f"{absrd}_lrn/tables/gate_b.json")

    for tgt, T in (("rot_fp16", "TR"), ("w8a8", "T8"), ("w4a4", "T4")):
        ev(f"A1_{T}", tgt, "naive", "mtbench", 80, rbin=LRN, prio=8,
           deps=["gateB_lrn"])
        ev(f"A2_{T}", tgt, "explicit", "mtbench", 80, rbin=LRN, prio=8,
           deps=["gateB_lrn"])
    for tgt, T in (("w8a8", "T8"), ("w4a4", "T4")):
        for ds, n in DATASETS:
            ev(f"G_{T}D16", tgt, "folded", ds, n, rbin=LRN, prio=7,
               deps=["gateB_lrn", "gateC"])
            ev(f"G_{T}D8", tgt, "folded", ds, n, rbin=LRN,
               extra=" --draft-mode w8a8", prio=6,
               deps=["gateB_lrn", "gateC"])
            ev(f"G_{T}D4", tgt, "folded", ds, n, rbin=LRN,
               extra=" --draft-mode w4a4", prio=6,
               deps=["gateB_lrn", "gateC"])
    # rot_fp16 folded interface sanity row (diagnostic, mtbench only)
    ev("A3_TR", "rot_fp16", "folded", "mtbench", 80, rbin=LRN, prio=7,
       deps=["gateB_lrn", "gateC"])

    qdir = os.path.join(absrd, "scheduler")
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, "queue.jsonl"), "a") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"enqueued {len(jobs)} jobs -> {qdir}/queue.jsonl")


if __name__ == "__main__":
    main()
