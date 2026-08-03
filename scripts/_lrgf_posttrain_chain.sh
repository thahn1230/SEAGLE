#!/bin/bash
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/LRGF_RUN_DIR)
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID
while [ $(ps aux | grep "train_eagle_learned_rotation.py --" | grep -v grep | wc -l) -gt 0 ]; do sleep 60; done
echo "[pt] training done -> heldout evals"
python scripts/eval_eagle_learned_rotation.py --phase heldout --run-dir $RD > $RD/logs/lr_heldout.log 2>&1
echo "[pt] heldout done -> validation RCAL captures"
python scripts/eval_eagle_learned_rotation.py --phase rcal --run-dir $RD > $RD/logs/lr_rcal.log 2>&1
echo "[pt] captures done -> error-flow trace + granularity"
CUDA_VISIBLE_DEVICES=0 python scripts/trace_eagle_quantization_error_flow.py --run-dir $RD > $RD/logs/error_flow.log 2>&1 &
wait
echo LRGF_POSTTRAIN_DONE
