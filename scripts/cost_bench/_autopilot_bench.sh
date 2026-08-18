#!/bin/bash
# SEAGLE training-cost benchmark autopilot — session-independent.
# Launch: setsid nohup bash scripts/cost_bench/_autopilot_bench.sh &
# GPU 7 only (user-selected). Waits for GPU 7 to be free (3 sustained
# checks) before each job; never touches other GPUs. 3 interleaved
# rounds x {R5, R6, QAT, RT-A, RT-B} + 1 QAT val-cadence round, then
# analysis. State: $RUN/BENCH_STATE; log: $RUN/logs/autopilot.log.
set -u
cd /home/thahn1230/SEAGLE
RUN=$(cat runs/COST_BENCH_RUN_DIR)
PY=/home/thahn1230/anaconda3/envs/seagle/bin/python
export HF_HOME=/data/thahn1230/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
GPU=7
AAQ=runs/eagle1_aaq_canonical_20260813_082500
GS=runs/eagle1_draft_aware_r1_r2_gs_20260810_065036
RD5=runs/eagle1_draft_residual_rotation_ep3p_20260804_165317/rotations/RD_HYB_s2.pt
SRT=runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552
ALPHA=32.89964245299412
log(){ echo "$(date -u +%FT%TZ) $*" >> $RUN/logs/autopilot.log; }
log "bench autopilot start pid $$ (GPU $GPU)"

wait_gpu(){  # sustained-free guard (minutes-long allocation race)
  local ok=0
  while [ $ok -lt 3 ]; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    if [ "$m" -lt 500 ]; then ok=$((ok+1)); else ok=0; log "GPU$GPU busy (${m}MiB), waiting"; fi
    sleep 30
  done
}

snap(){ nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -i $GPU >> $RUN/manifests/gpu_during_bench.csv; }

lk_job(){  # name round extra-args...
  local name=$1 round=$2; shift 2
  wait_gpu; snap
  log "start $name r$round"
  local t0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_eagle_lk_rotation.py \
    --corpus $AAQ/manifests/lkcorpus__can_all.json --run-dir $RUN \
    --teacher t4 --kv-bits 16 --K 4 --batch 32 --accum 1 \
    --alpha-init $ALPHA --steps 360 --seed $round "$@" \
    --out $RUN/logs/${name}/bench_r${round}.pt \
    > $RUN/logs/${name}/r${round}.log 2>&1
  local rc=$?
  echo "{\"job\":\"$name\",\"round\":$round,\"rc\":$rc,\"setup_plus_total_s\":$(( $(date +%s) - t0 ))}" >> $RUN/tables/job_walls.jsonl
  log "end $name r$round rc=$rc"
}

rt_job(){  # name round teacher extra...
  local name=$1 round=$2 teacher=$3; shift 3
  wait_gpu; snap
  log "start $name r$round"
  local t0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=$GPU RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    $PY scripts/strict_rt/train_eagle1_strict_rt.py --run-dir $RUN \
    --teacher $teacher --grad-ckpt on --bench 560 --seed $round "$@" \
    > $RUN/logs/RT/${name}_r${round}.log 2>&1
  local rc=$?
  echo "{\"job\":\"$name\",\"round\":$round,\"rc\":$rc,\"setup_plus_total_s\":$(( $(date +%s) - t0 ))}" >> $RUN/tables/job_walls.jsonl
  log "end $name r$round rc=$rc"
}

for r in 1 2 3; do
  lk_job PTQ_R5 $r --objective hybrid --rot residual --r2-mode frozen \
      --lr 3e-4 --eval-every 999999
  lk_job PTQ_R6 $r --objective hybrid --rot shared --r2-mode residual \
      --lr 1e-3 --eval-every 999999
  lk_job QAT $r --objective conv --train-draft-core --core-lr 1e-6 \
      --rot shared --rot-fixed-ckpt $RD5 --r2-mode frozen --lr 3e-4 \
      --eval-every 999999
  rt_job RT_A $r cache --bench-cached-only
  rt_job RT_B $r hybrid
done
# QAT with canonical validation cadence (amortization measurement)
lk_job QAT_valcad 1 --objective conv --train-draft-core --core-lr 1e-6 \
    --rot shared --rot-fixed-ckpt $RD5 --r2-mode frozen --lr 3e-4 \
    --eval-every 100

echo "BENCH_JOBS_DONE" > $RUN/BENCH_STATE
log "all jobs done; running analysis"
$PY scripts/cost_bench/analyze.py --run-dir $RUN \
    > $RUN/logs/analyze.log 2>&1 && echo "BENCH_DONE" > $RUN/BENCH_STATE \
    || echo "ANALYZE_FAILED" > $RUN/BENCH_STATE
log "autopilot complete: $(cat $RUN/BENCH_STATE)"
