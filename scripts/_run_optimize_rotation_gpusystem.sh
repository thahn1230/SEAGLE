#!/usr/bin/env bash
# R1_T rebuild launcher for host gpusystem (fresh server, 2026-08-10).
# Mirrors scripts/_run_optimize_rotation.sh (official SpinQuant recipe:
# global batch 8 x 100 steps, lr 1.5 cosine, seqlen 2048, W4A4KV16-in-loop)
# with corrected ROOT/env for this server: 4 GPUs x batch 1 x accum 2 = 8.
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate seagle
export HF_HOME=/data/thahn1230/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CVD_OVERRIDE:-0,1,2,3}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_DISABLED=true
MODEL="$1"; OUTD="$2"; TAG="$3"; W=${4:-4}; A=${5:-4}; KV=${6:-16}
ROOT=/home/thahn1230/SEAGLE
LOGD="$ROOT/runs/eagle1_draft_aware_r1_r2_gs_20260810_065036/logs"
mkdir -p "$LOGD" "$OUTD"
cd "$ROOT/third_party/SpinQuant"
MASTER_PORT=$((28500 + RANDOM % 1000))
torchrun --nnodes=1 --nproc_per_node=${NPROC:-4} --master_port=$MASTER_PORT optimize_rotation.py \
  --input_model "$MODEL" \
  --output_rotation_path "$OUTD" \
  --output_dir "$OUTD/out/" \
  --logging_dir "$OUTD/log/" \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${ACCUM:-2} \
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
  --v_groupsize 128 2>&1 | tee "$LOGD/optrot_${TAG}.log"
echo "[optrot] ${TAG} DONE"
