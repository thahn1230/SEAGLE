#!/bin/bash
# Strict SEAGLE-RT §5/§6 benchmark suite. Waits for parity gates to
# report ALL PASS, then benchmarks the three training modes and DDP
# scaling. One bench at a time (each owns all its GPUs).
set -u
cd /home/thahn1230/SEAGLE
RUN=runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552
PY=/home/thahn1230/anaconda3/envs/seagle/bin/python
export HF_HOME=/data/thahn1230/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# wait up to 30 min for gates
for i in $(seq 1 60); do
  grep -q "\[gates\] ALL PASS" $RUN/logs/parity_gates.log 2>/dev/null && break
  grep -qE "FAILURE|Traceback" $RUN/logs/parity_gates.log 2>/dev/null && {
    echo "GATES FAILED — aborting benches"; exit 1; }
  sleep 30
done
grep -q "\[gates\] ALL PASS" $RUN/logs/parity_gates.log || {
  echo "GATES TIMEOUT"; exit 1; }
echo "gates passed; benches start $(date -u +%H:%M:%SZ)"

bench () {  # name world teacher ckpt steps
  local name=$1 w=$2 teacher=$3 ck=$4 steps=$5
  echo "=== bench $name (world=$w teacher=$teacher ckpt=$ck) ==="
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((w-1))) torchrun \
    --nproc_per_node=$w --master_port=29655 \
    scripts/strict_rt/train_eagle1_strict_rt.py --run-dir $RUN \
    --teacher $teacher --grad-ckpt $ck --bench $steps \
    2>&1 | tee $RUN/logs/bench_$name.log | grep -E '"bench"|\[srt\] world|Error|OutOfMemory' || true
}

bench C_hyb_off 8 hybrid off 120
C_OK=$(grep -c '"bench"' $RUN/logs/bench_C_hyb_off.log || true)
bench B_hyb_on 8 hybrid on 120
bench A_online 8 online on 40

# world scaling for the chosen mode (C if it survived, else B)
if [ "${C_OK:-0}" -ge 1 ]; then MODE=off; else MODE=on; fi
for w in 4 2 1; do
  bench scale_w$w $w hybrid $MODE 30
done
echo "benches done $(date -u +%H:%M:%SZ)"
