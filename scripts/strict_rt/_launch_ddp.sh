#!/bin/bash
# Manual DDP launcher with PER-RANK SINGLE-GPU VISIBILITY (avoids the
# SpinQuant-build cuda:0 leak that torchrun's shared visibility hits).
# Usage: _launch_ddp.sh <world> <teacher> <grad_ckpt> <bench_steps|0> [extra args...]
set -u
cd /home/thahn1230/SEAGLE
RUN=runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552
PY=/home/thahn1230/anaconda3/envs/seagle/bin/python
export HF_HOME=/data/thahn1230/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
W=$1; TEACHER=$2; CKPT=$3; BENCH=$4; shift 4
EXTRA=("$@")
PORT=29671
PIDS=()
for k in $(seq 0 $((W-1))); do
  BF=""
  [ "$BENCH" != "0" ] && BF="--bench $BENCH"
  env CUDA_VISIBLE_DEVICES=$k RANK=$k LOCAL_RANK=0 WORLD_SIZE=$W \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
      $PY scripts/strict_rt/train_eagle1_strict_rt.py --run-dir $RUN \
      --teacher $TEACHER --grad-ckpt $CKPT $BF "${EXTRA[@]}" \
      > $RUN/logs/ddp_r$k.log 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait $p || rc=1; done
exit $rc
