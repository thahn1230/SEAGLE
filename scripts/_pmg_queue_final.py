#!/usr/bin/env python
"""Final evaluation wave: median-seed multi-dataset QF evals + RCAL
captures (T4 x 8 methods) + RCAL metrics/bootstrap.

Median seed per (method,target): the seed whose QF mtbench tau is the
median of the three (pre-registered; never the best seed). Requires all
sel_ jobs of a condition done (QF mtbench shards present) — run this
once QAT/sel complete; safe to re-run (artifact-guarded queue ids).
"""
import csv, glob, json, os, sys

rd = sys.argv[1]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
D = 4096
t8 = json.load(open(f"{rd}/tables/ep3_selection_w8a8.json"))
RDROT = open("runs/RDROT_RUN_DIR").read().strip()
TN = {"fp16": "T16", "w8a8": "T8", "int4": "T4"}
SC = {
  "fp16": {"gen": (1.0, None, None), "ep3g": (D**0.46, None, None),
           "ep3p": (D**0.46, D**0.46, None),
           "rd": (D**0.46, D**0.46, f"{rd}/rotations/RD_T16_HYB_s2v2.pt")},
  "w8a8": {"gen": (1.0, None, None),
           "ep3g": (t8["ep3g"]["m"], None, None),
           "ep3p": (t8["ep3p"]["m_f"], t8["ep3p"]["m_r"], None),
           "rd": (t8["ep3p"]["m_f"], t8["ep3p"]["m_r"],
                  f"{rd}/rotations/RD_T8_HYB_s2.pt")},
  "int4": {"gen": (1.0, None, None), "ep3g": (D**0.42, None, None),
           "ep3p": (D**0.40, D**0.45, None),
           "rd": (D**0.40, D**0.45,
                  f"{RDROT}/rotations/RD_HYB_s2.pt")},
}


def tau_of(path):
    ts = [t for r in csv.DictReader(open(path))
          for t in json.loads(r["acceptance_list"])]
    return sum(ts) / max(len(ts), 1)


med = {}
missing = []
for tgt in ("fp16", "w8a8", "int4"):
    if ONLY and tgt not in ONLY:
        continue
    for meth in ("gen", "ep3g", "ep3p", "rd"):
        taus = {}
        for s in (0, 1, 2):
            p = os.path.join(rd, "shards",
                             f"al__QF_{meth}_{TN[tgt]}_s{s}__{tgt}"
                             f"__mtbench.csv")
            if os.path.exists(p):
                taus[s] = tau_of(p)
        if len(taus) < 3:
            missing.append((meth, tgt, sorted(taus)))
            continue
        order = sorted(taus, key=lambda s: taus[s])
        med[(meth, tgt)] = dict(seed=order[1],
                                taus={str(s): round(taus[s], 4)
                                      for s in (0, 1, 2)})
if missing:
    print("[final] skipping not-ready conditions:", missing)
mspath = f"{rd}/tables/median_seeds.json"
old = json.load(open(mspath)) if os.path.exists(mspath) else {}
old.update({f"{m}_{TN[t]}": v for (m, t), v in med.items()})
json.dump(old, open(mspath, "w"), indent=1)

EV = "python scripts/eval_eagle_acceptance_length.py"
CAP = "python scripts/capture_eagle_proposal_cycles.py"
q = []
def J(jid, cmd, gpus=1, deps=(), prio=0, artifact=None):
    q.append(dict(id=jid, cmd=cmd, gpus=gpus, deps=list(deps),
                  wait_files=[], prio=prio, artifact=artifact,
                  log=f"{rd}/logs/sched_{jid}.log"))


def deploy(meth, tgt, sd_path):
    a, ar, rdck = SC[tgt][meth]
    if rdck:
        s = (f"--draft-cfg rot_ep3p --ckpt {rdck} --alpha {a} "
             f"--alpha-rec {ar}")
    else:
        s = f"--draft-cfg d4p3_deploy --alpha {a}"
        if ar:
            s += f" --alpha-rec {ar}"
    return s + (f" --draft-sd {sd_path}" if sd_path else "")


for (meth, tgt), v in med.items():
    tag = f"Q_{meth}_{TN[tgt]}_s{v['seed']}"
    sel = json.load(open(f"{rd}/tables/qat_selection_{tag}.json"))
    sd = sel["selected"]["path"]
    ftag = f"QF_{meth}_{TN[tgt]}_s{v['seed']}"
    for ds, n in (("gsm8k", 200), ("sharegpt", 80),
                  ("humaneval", 164)):
        J(f"fds_{ftag}_{ds}",
          f"{EV} --target {tgt} {deploy(meth, tgt, sd)} --tag {ftag} "
          f"--datasets {ds} --pool eval --n-prompts {n} --run-dir {rd}",
          prio=5,
          artifact=f"{rd}/shards/al__{ftag}__{tgt}__{ds}.csv")

# RCAL: T4 x 8 methods, mtbench 80 captures (2 GPUs each)
rcal_tags = []
for meth in (("gen", "ep3g", "ep3p", "rd")
             if (not ONLY or "int4" in ONLY) and
             any(t == "int4" for (_, t) in med) else ()):
    a, ar, rdck = SC["int4"][meth]
    # PTQ arm
    ptq_map = {"gen": "B1_T4", "ep3g": "B3_T4", "ep3p": "B5_T4",
               "rd": "B7_T4"}
    vtag = "V" + ptq_map[meth]
    J(f"cap_{vtag}",
      f"{CAP} --target int4 {deploy(meth, 'int4', None)} --tag {vtag} "
      f"--datasets mtbench --n-prompts 80 --run-dir {rd} "
      f"--ref-device cuda:1", gpus=2, prio=6,
      artifact=f"{rd}/cycles/cyc__{vtag}__mtbench.jsonl")
    rcal_tags.append(vtag)
    # QAT arm (median seed selected ckpt)
    v = med[(meth, "int4")]
    tag = f"Q_{meth}_T4_s{v['seed']}"
    sel = json.load(open(f"{rd}/tables/qat_selection_{tag}.json"))
    vtag = f"VQF_{meth}_T4"
    J(f"cap_{vtag}",
      f"{CAP} --target int4 "
      f"{deploy(meth, 'int4', sel['selected']['path'])} --tag {vtag} "
      f"--datasets mtbench --n-prompts 80 --run-dir {rd} "
      f"--ref-device cuda:1", gpus=2, prio=6,
      artifact=f"{rd}/cycles/cyc__{vtag}__mtbench.jsonl")
    rcal_tags.append(vtag)
J("rcal_metrics",
  f"python scripts/compute_eagle_rcal_metrics.py --run-dir {rd} && "
  f"python scripts/bootstrap_eagle_rcal.py --run-dir {rd} "
  f"--tags {','.join(rcal_tags)} "
  f"--pairs " + ",".join(f"VB1_T4:{t}" for t in rcal_tags
                         if t != "VB1_T4") +
  f" --dataset mtbench --reps 3000",
  gpus=0, deps=[f"cap_{t}" for t in rcal_tags], prio=6,
  artifact=f"{rd}/stats/rcal_bootstrap_mtbench.json")
with open(f"{rd}/scheduler/queue.jsonl", "a") as f:
    for j in q:
        f.write(json.dumps(j) + "\n")
print(f"[final] median seeds {json.dumps({f'{m}_{TN[t]}': v['seed'] for (m,t),v in med.items()})}")
print(f"[final] appended {len(q)} jobs")
