#!/usr/bin/env bash
# Auto-chaining orchestrator for the GS R1/R2 factorial study.
# Keeps all 8 GPUs busy across phases with no idle gaps:
#   phase0: wait for R1_T (outputs/rotations/learned_chat_w4a4kv16/R.bin)
#   phase1: corpus 5-way (GPUs 0-4)  ||  gates (GPU 5: model audit + PPL,
#           GPU 6: R2-A/B gates, GPU 7: GS parity gate)
#   phase2: merge corpus (CPU) ; ABORT if any gate failed
#   phase3: LR pilot 6-way (GPUs 0-5)
#   phase4: LR selection -> jobs_train.txt -> 9 training jobs on 8 GPUs
#   phase5: median-seed selection -> jobs_eval.txt -> eval queue on 8 GPUs
# Every phase logs to $RD/logs/orchestrator.log (the driver Monitor
# watches it); any failure stops the chain with PHASE_FAIL.
set -uo pipefail
RD=/home/thahn1230/SEAGLE/runs/eagle1_draft_aware_r1_r2_gs_20260810_065036
cd /home/thahn1230/SEAGLE
source ~/anaconda3/etc/profile.d/conda.sh
conda activate seagle
export HF_HOME=/data/thahn1230/hf_cache
LOG() { echo "[orch $(date +%H:%M:%S)] $*"; }

RBIN=outputs/rotations/learned_chat_w4a4kv16/R.bin
OPTLOG=$RD/logs/optrot_chat_w4a4kv16.log

LOG "phase0: waiting for R1_T ($RBIN)"
until [ -f "$RBIN" ] && grep -q "\[optrot\] chat_w4a4kv16 DONE" "$OPTLOG"; do
  if grep -qE "ERROR|Traceback" "$OPTLOG" && ! pgrep -f optimize_rotation.py > /dev/null; then
    LOG "PHASE_FAIL phase0: rotation training died"; exit 1
  fi
  sleep 60
done
sha256sum "$RBIN" | tee -a "$RD/manifests/r1t_sha256.txt"
LOG "phase0 DONE: R1_T ready"

LOG "phase1: corpus (GPUs 0-4) + gates (GPUs 5-7)"
python scripts/_r1r2gs_gpuq.py --jobs $RD/configs/jobs_corpus.txt \
  --gpus 0,1,2,3,4 --log-dir $RD/logs > $RD/logs/gpuq_corpus.log 2>&1 &
CORPUS_PID=$!
( export CUDA_VISIBLE_DEVICES=5
  python scripts/audit_r2_contract.py --run-dir $RD --level model \
    > $RD/logs/gate_model_audit.log 2>&1
  echo $? > $RD/gradchecks/exit_model_audit
  python scripts/_r1r2gs_ppl_sanity.py --run-dir $RD \
    > $RD/logs/ppl_sanity.log 2>&1
  echo $? > $RD/gradchecks/exit_ppl ) &
G5_PID=$!
( export CUDA_VISIBLE_DEVICES=6
  python scripts/check_r2_gates.py --run-dir $RD \
    > $RD/logs/gate_r2_ab.log 2>&1
  echo $? > $RD/gradchecks/exit_r2_ab ) &
G6_PID=$!
( export CUDA_VISIBLE_DEVICES=7
  python scripts/check_r1r2_gs_parity.py --run-dir $RD \
    > $RD/logs/gate_gs_parity.log 2>&1
  echo $? > $RD/gradchecks/exit_gs_parity ) &
G7_PID=$!
wait $CORPUS_PID; CORPUS_RC=$?
wait $G5_PID $G6_PID $G7_PID
LOG "phase1 gates: audit=$(cat $RD/gradchecks/exit_model_audit) ppl=$(cat $RD/gradchecks/exit_ppl) r2ab=$(cat $RD/gradchecks/exit_r2_ab) gs=$(cat $RD/gradchecks/exit_gs_parity) corpus_rc=$CORPUS_RC"

if [ "$CORPUS_RC" != "0" ]; then LOG "PHASE_FAIL phase1: corpus"; exit 1; fi
for g in exit_model_audit exit_r2_ab exit_gs_parity exit_ppl; do
  if [ "$(cat $RD/gradchecks/$g)" != "0" ]; then
    LOG "PHASE_FAIL phase1: gate $g failed — NOT starting training"; exit 1
  fi
done

LOG "phase2: merge corpus (stratified shuffle)"
python scripts/_r1r2gs_merge_corpus.py $RD \
  > $RD/logs/merge_corpus.log 2>&1 || { LOG "PHASE_FAIL phase2"; exit 1; }

LOG "phase3: LR pilot (6 jobs, GPUs 0-5)"
python scripts/_r1r2gs_gpuq.py --jobs $RD/configs/jobs_pilot.txt \
  --gpus 0,1,2,3,4,5 --log-dir $RD/logs > $RD/logs/gpuq_pilot.log 2>&1 \
  || { LOG "PHASE_FAIL phase3: pilot"; exit 1; }

LOG "phase4: LR selection + full training (9 jobs, 8 GPUs)"
python scripts/_r1r2gs_select_lr.py --run-dir $RD \
  > $RD/logs/lr_selection.log 2>&1 || { LOG "PHASE_FAIL phase4: lrsel"; exit 1; }
cat $RD/logs/lr_selection.log
python scripts/_r1r2gs_gpuq.py --jobs $RD/configs/jobs_train.txt \
  --gpus 0,1,2,3,4,5,6,7 --log-dir $RD/logs > $RD/logs/gpuq_train.log 2>&1 \
  || { LOG "PHASE_FAIL phase4: training"; exit 1; }

LOG "phase5: eval jobs (median seeds) + eval queue (8 GPUs)"
python scripts/_r1r2gs_make_eval_jobs.py --run-dir $RD \
  > $RD/logs/eval_jobs.log 2>&1 || { LOG "PHASE_FAIL phase5: evaljobs"; exit 1; }
cat $RD/logs/eval_jobs.log
python scripts/_r1r2gs_gpuq.py --jobs $RD/configs/jobs_eval.txt \
  --gpus 0,1,2,3,4,5,6,7 --log-dir $RD/logs > $RD/logs/gpuq_eval.log 2>&1 \
  || { LOG "PHASE_FAIL phase5: eval"; exit 1; }

LOG "ALL_PHASES_DONE — ready for stats/RCAL/mechanism/runtime"
