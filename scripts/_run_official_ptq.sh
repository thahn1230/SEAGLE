#!/usr/bin/env bash
# Official SpinQuant ptq.py runner (vendored @8f47aa3, unmodified).
# Usage: _run_official_ptq.sh <model_path_or_id> <out_tag> [extra ptq args...]
# Runs torchrun single-proc on the permitted GPU(s); logs to
# artifacts/spinquant_ppl_reproduction_fix/logs/<out_tag>.log
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CVD_OVERRIDE:-6}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# pinned SpinQuant stack (transformers 4.44.2) if built
VENV=/data/thahn1230/envs/spinquant/bin
[ -x "$VENV/torchrun" ] && PATH="$VENV:$PATH"
MODEL="$1"; TAG="$2"; shift 2
ROOT=/home/thahn1230/eagle_spinquant_w4a4
LOGD="$ROOT/artifacts/spinquant_ppl_reproduction_fix/logs"
mkdir -p "$LOGD"
cd "$ROOT/third_party/SpinQuant"
MASTER_PORT=$((29500 + RANDOM % 1000))
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT ptq.py \
  --input_model "$MODEL" \
  --do_train False \
  --do_eval True \
  --per_device_eval_batch_size 4 \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --save_safetensors False \
  "$@" 2>&1 | tee "$LOGD/${TAG}.log"
echo "[official] ${TAG} DONE"
