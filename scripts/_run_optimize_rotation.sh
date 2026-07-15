#!/usr/bin/env bash
# Official SpinQuant rotation optimization (vendored, unmodified).
# Usage: _run_optimize_rotation.sh <model_path> <out_dir> <tag> [w a kv]
# Single permitted GPU: reproduces the 8-GPU recipe (global batch 8 x 100
# steps = 800 wikitext samples) via gradient_accumulation_steps=8.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CVD_OVERRIDE:-6}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
VENV=/data/thahn1230/envs/spinquant/bin
[ -x "$VENV/torchrun" ] && PATH="$VENV:$PATH"
MODEL="$1"; OUTD="$2"; TAG="$3"; W=${4:-4}; A=${5:-4}; KV=${6:-16}
ROOT=/home/thahn1230/eagle_spinquant_w4a4
LOGD="$ROOT/artifacts/spinquant_ppl_reproduction_fix/logs"
mkdir -p "$LOGD" "$OUTD"
cd "$ROOT/third_party/SpinQuant"
MASTER_PORT=$((28500 + RANDOM % 1000))
torchrun --nnodes=1 --nproc_per_node=${NPROC:-1} --master_port=$MASTER_PORT optimize_rotation.py \
  --input_model "$MODEL" \
  --output_rotation_path "$OUTD" \
  --output_dir "$OUTD/out/" \
  --logging_dir "$OUTD/log/" \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${ACCUM:-8} \
  --logging_steps 1 \
  --learning_rate 1.5 \
  --weight_decay 0. \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing True \
  --save_safetensors False \
  --max_steps 100 \
  --w_bits "$W" \
  --a_bits "$A" \
  --k_bits "$KV" \
  --v_bits "$KV" \
  --w_clip \
  --a_asym \
  --k_asym \
  --v_asym \
  --k_groupsize 128 \
  --v_groupsize 128 2>&1 | tee "$LOGD/${TAG}.log"
echo "[optrot] ${TAG} DONE"
