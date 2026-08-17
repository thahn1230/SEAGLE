#!/bin/bash
# Strict SEAGLE-RT AUTOPILOT — fully session-independent orchestrator.
# Launch once with: setsid nohup bash scripts/strict_rt/_autopilot.sh &
#
# Phase W  watchdog: if the training launcher dies before the final
#          checkpoint exists, resume from strict_rt_last.pt (<=5x).
# Phase AB when strict_rt_final.pt exists: dispatch the 18-job Stage
#          A/B1 list over GPUs 0-7 (single dispatcher — never run a
#          second one or hand-launch evals concurrently).
# Phase S  per-tag micro-tau/mean4 table + preregistered bootstrap
#          pairs P1/P2/P4 (10k reps, ALL pairs in one invocation per
#          dataset — bootstrap_eagle_tau.py overwrites its JSON).
# Then writes AUTOPILOT_DONE. Stage-B2 branching (HAD/SQ/rescue) and
# the final report intentionally require the operator.
set -u
cd /home/thahn1230/SEAGLE
RUN=runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552
PY=/home/thahn1230/anaconda3/envs/seagle/bin/python
CKD=/data/thahn1230/strict_rt_ckpts
ST=$RUN/logs/autopilot.log
export HF_HOME=/data/thahn1230/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
log() { echo "$(date -u +%FT%TZ) $*" >> $ST; }
log "autopilot start (pid $$)"

launcher_alive() {
  pgrep -f "strict_rt/_launch_ddp[.]sh 8 hybrid on 0" >/dev/null 2>&1 \
  || pgrep -f "train_eagle1_strict_rt[.]py --run-dir .* --teacher hybrid --grad-ckpt on$" >/dev/null 2>&1 \
  || pgrep -f "train_eagle1_strict_rt" >/dev/null 2>&1
}

# ---------------- Phase W ----------------
RESTARTS=0
while [ ! -f $CKD/strict_rt_final.pt ]; do
  if launcher_alive; then
    sleep 300
    continue
  fi
  sleep 60   # settle: distinguish crash from just-finished save
  [ -f $CKD/strict_rt_final.pt ] && break
  if [ $RESTARTS -ge 5 ]; then
    log "TRAIN DEAD after 5 restarts — giving up; operator needed"
    echo "AUTOPILOT_TRAIN_FAILED" > $RUN/AUTOPILOT_STATE
    exit 1
  fi
  RESTARTS=$((RESTARTS+1))
  RES=""
  [ -f $CKD/strict_rt_last.pt ] && RES="--resume $CKD/strict_rt_last.pt"
  log "training not alive; RESTART #$RESTARTS ($RES)"
  nohup bash scripts/strict_rt/_launch_ddp.sh 8 hybrid on 0 $RES \
      >> $RUN/logs/train_launcher.log 2>&1 &
  sleep 900   # give the restart time to build teachers
done
log "training final checkpoint present"
echo "TRAIN_DONE" > $RUN/AUTOPILOT_STATE

# ---------------- Phase AB ----------------
sleep 60   # let all ranks exit cleanly
sha256sum $CKD/strict_rt_final.pt > $RUN/manifests/selected_checkpoint.sha256
log "dispatching Stage A/B1 (18 jobs)"
$PY scripts/_r1r2gs_gpuq.py --jobs $RUN/configs/jobs_stageAB1.txt \
    --gpus 0,1,2,3,4,5,6,7 --log-dir $RUN/logs >> $ST 2>&1
log "Stage A/B1 dispatcher exited rc=$?"
echo "STAGE_AB1_DONE" > $RUN/AUTOPILOT_STATE

# ---------------- Phase S ----------------
$PY - <<'EOF' >> runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552/logs/autopilot.log 2>&1
import csv, glob, json, os
RUN = "runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552"
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")
rows = {}
for p in glob.glob(os.path.join(RUN, "shards", "al__*__int4__*.csv")):
    parts = os.path.basename(p)[:-4].split("__")
    tag, ds = parts[1], parts[3]
    if ds not in DS or len(parts) > 4:
        continue
    tot = cyc = 0
    for r in csv.DictReader(open(p)):
        al = json.loads(r["acceptance_list"])
        tot += sum(al); cyc += len(al)
    rows.setdefault(tag, {})[ds] = round(tot / max(cyc, 1), 4)
for tag, d in rows.items():
    if all(k in d for k in DS):
        d["mean4"] = round(sum(d[k] for k in DS) / 4, 4)
out = os.path.join(RUN, "tables", "al_4dataset.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag"] + list(DS) + ["mean4"])
    for tag in sorted(rows):
        d = rows[tag]
        w.writerow([tag] + [d.get(k, "") for k in DS] +
                   [d.get("mean4", "")])
print("[autopilot] al_4dataset.csv:", json.dumps(rows))
EOF
for ds in mtbench gsm8k sharegpt humaneval; do
  $PY scripts/bootstrap_eagle_tau.py --run-dir $RUN --dataset $ds \
    --reps 10000 \
    --pair "P1_ctrl_vs_srtfp16=CTRL_OI@int4:SRT_FP16@int4" \
    --pair "P2_fp16_vs_w8a8=SRT_FP16@int4:SRT_W8A8_RTN@int4" \
    --pair "P4_fp16_vs_w4a4=SRT_FP16@int4:SRT_W4A4_RTN@int4" \
    >> $ST 2>&1
done
log "bootstrap P1/P2/P4 done"
echo "AUTOPILOT_DONE" > $RUN/AUTOPILOT_STATE
log "autopilot complete — Stage-B2 branching + final report need operator"
