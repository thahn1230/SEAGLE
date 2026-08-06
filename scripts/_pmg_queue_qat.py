#!/usr/bin/env python
"""Append the 12-condition x 3-seed QAT campaign to the PMG queue.

Usage: _pmg_queue_qat.py <run_dir> <lr>

Per condition x seed: train (3000 steps, shared LR, anchor init,
ckpt/500) -> selection (calib-AL over {step1500, best-loss, last}) ->
mtbench eval of the selected ckpt. The 3 non-mtbench datasets run on
the MEDIAN-calib-AL seed (pre-registered; never the best seed).
Deps: T8 arms after calib (already done), B8 arms after their R_D.
"""
import json, os, sys

rd, LR = sys.argv[1], sys.argv[2]
D = 4096
ENV = ("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
       "TMPDIR=/data/thahn1230/tmp ")
TR = ENV + "python scripts/train_eagle_draft_int4_qat.py"
SEL = "python scripts/_pmg_qat_select.py"
t8 = json.load(open(f"{rd}/tables/ep3_selection_w8a8.json"))
RDROT = open("runs/RDROT_RUN_DIR").read().strip()

SC = {   # target -> dict(method -> (alpha, alpha_rec, rd_ckpt))
  "fp16": {"gen": (1.0, None, None),
           "ep3g": (D ** 0.46, None, None),
           "ep3p": (D ** 0.46, D ** 0.46, None),
           "rd": (D ** 0.46, D ** 0.46,
                  f"{rd}/rotations/RD_T16_HYB_s2v2.pt")},
  "w8a8": {"gen": (1.0, None, None),
           "ep3g": (t8["ep3g"]["m"], None, None),
           "ep3p": (t8["ep3p"]["m_f"], t8["ep3p"]["m_r"], None),
           "rd": (t8["ep3p"]["m_f"], t8["ep3p"]["m_r"],
                  f"{rd}/rotations/RD_T8_HYB_s2.pt")},
  "int4": {"gen": (1.0, None, None),
           "ep3g": (D ** 0.42, None, None),
           "ep3p": (D ** 0.40, D ** 0.45, None),
           "rd": (D ** 0.40, D ** 0.45,
                  f"{RDROT}/rotations/RD_HYB_s2.pt")},
}
ARM = {"fp16": "--arm C3", "w8a8": "--arm C7 --teacher-quant w8a8",
       "int4": "--arm C7"}
TN = {"fp16": "T16", "w8a8": "T8", "int4": "T4"}
DEP = {("fp16", "rd"): ["rd_train_t16v2"],
       ("w8a8", "rd"): ["rd_train_t8"]}

q = []
def J(jid, cmd, gpus=1, deps=(), wf=(), prio=0):
    q.append(dict(id=jid, cmd=cmd, gpus=gpus, deps=list(deps),
                  wait_files=list(wf), prio=prio,
                  log=f"{rd}/logs/sched_{jid}.log"))

for tgt in ("int4", "fp16", "w8a8"):
    for meth in ("gen", "ep3g", "ep3p", "rd"):
        a, ar, rdck = SC[tgt][meth]
        for seed in (0, 1, 2):
            tag = f"Q_{meth}_{TN[tgt]}_s{seed}"
            cmd = (f"{TR} {ARM[tgt]} --seed {seed} --run-dir {rd} "
                   f"--steps 3000 --bs 1 --accum 4 --lr {LR} "
                   f"--warmup 200 --val-every 500 --save-ckpt-every "
                   f"500 --alpha {a} "
                   + (f"--alpha-rec {ar} " if ar else "")
                   + (f"--rd-ckpt {rdck} " if rdck else "")
                   + f"--init-sd checkpoints/eagle1_fresh_fp16_anchor/"
                   f"anchor.pt --tag {tag}")
            J(f"qat_{tag}", cmd, gpus=1,
              deps=DEP.get((tgt, meth), []), prio=6)
            # selection + mtbench eval of selected ckpt (single job,
            # serial on one GPU)
            J(f"sel_{tag}",
              f"{SEL} {rd} {tag} {tgt} {a} "
              f"{ar if ar else 'none'} {rdck if rdck else 'none'}",
              gpus=1, deps=[f"qat_{tag}"], prio=5)
with open(f"{rd}/scheduler/queue.jsonl", "a") as f:
    for j in q:
        f.write(json.dumps(j) + "\n")
print(f"appended {len(q)} QAT jobs (LR={LR})")
