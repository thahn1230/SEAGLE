#!/bin/bash
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/LRGF_RUN_DIR)
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID
export LRGF_GPUS=0,1,2,3,4,5,6
python scripts/eval_eagle_learned_rotation.py --phase heldout --run-dir $RD > $RD/logs/lr_heldout2.log 2>&1
echo "[rc] heldout2 done"
python scripts/eval_eagle_learned_rotation.py --phase rcal --run-dir $RD > $RD/logs/lr_rcal2.log 2>&1
echo "[rc] rcal2 done"
CUDA_VISIBLE_DEVICES=0 python scripts/trace_eagle_quantization_error_flow.py --run-dir $RD > $RD/logs/error_flow.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/build_eagle_rotation_granularity.py --run-dir $RD > $RD/logs/granularity.log 2>&1 &
wait
echo "[rc] trace + granularity done"
python scripts/analyze_eagle_learned_rotation_mechanism.py --run-dir $RD > $RD/logs/lr_mechanism.log 2>&1
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark_eagle_transform_folding.py --run-dir $RD > $RD/logs/folding_bench.log 2>&1
echo LRGF_RECOVERY_DONE
