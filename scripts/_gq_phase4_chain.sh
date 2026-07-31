#!/bin/bash
# GQ study phase 4: waits for phase-3 (matrix + RCAL + bootstrap),
# then cross-dataset finalists -> overhead -> plots -> tests -> final
# output draft. Idempotent.
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/GQ_RUN_DIR)
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[p4] waiting for GQ_PHASE3_DONE"
until grep -q GQ_PHASE3_DONE outputs/gq_phase3_chain.log 2>/dev/null; do
  sleep 60
done
echo "[p4] phase-3 done -> cross-dataset finalists"

python scripts/eval_eagle1_generic_qat_vs_p3exp.py --phase xds \
  --run-dir $RD --gpus 0,1,2,3,4,5,6,7 > outputs/gq_xds.log 2>&1
echo "[p4] xds done -> overhead"

CUDA_VISIBLE_DEVICES=0 python scripts/measure_ep3p_overhead.py \
  --run-dir $RD --target int4 --gpu 0 > outputs/gq_overhead.log 2>&1
echo "[p4] overhead done -> plots"

python scripts/plot_eagle1_rcal.py --run-dir $RD \
  > outputs/gq_plots_rcal.log 2>&1
python scripts/plot_eagle1_p3exp_3d.py --run-dir $RD \
  > outputs/gq_plots_3d.log 2>&1
echo "[p4] plots done -> tests"

mkdir -p $RD/manifests
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_gq_p3exp_contracts.py tests/test_rcal_contracts.py \
  > $RD/manifests/test_log.txt 2>&1
tail -1 $RD/manifests/test_log.txt

python scripts/print_generic_qat_p3exp_rcal_final_output.py $RD \
  > $RD/FINAL_OUTPUT_DRAFT.txt 2>&1
echo GQ_PHASE4_DONE
