#!/bin/bash
# R-EP3-P stage 5 + RCAL: waits for s4 pairs + smoke, then runs the
# acceptance ladder and lockstep captures. Idempotent.
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/REP3P_RUN_DIR)
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID
EV="python scripts/eval_rotated_ep3p_acceptance.py --run-dir $RD --target int4 --draft-cfg d4p3_deploy"
GQRD=runs/eagle1_generic_qat_pathwise_p3exp_rcal_20260731_154652

until [ -f $RD/candidates/s4_int4_pairs.json ]; do sleep 30; done
until ls $RD/candidates/cayley_rec_s2.json >/dev/null 2>&1; do sleep 30; done
echo "[ac] s4 + cayley ready -> stage 5"

python scripts/_rep3p_stage5_driver.py --phase calib9 --run-dir $RD \
  > $RD/logs/stage5_calib9.log 2>&1
echo "[ac] top-9 calib done"
python scripts/_rep3p_stage5_driver.py --phase mtbench --run-dir $RD \
  > $RD/logs/stage5_mtbench.log 2>&1
echo "[ac] mtbench done -> RCAL captures"
python scripts/_rep3p_stage5_driver.py --phase rcal --run-dir $RD \
  > $RD/logs/stage5_rcal.log 2>&1
echo "[ac] captures done -> reuse GQ baselines + metrics + bootstrap"
for TAG in RC_EP3P_T1 RC_EP3G_T1 RC_NPTQ_T1 RC_LP3RD_T1 RC_LP3QAT_T1; do
  [ -f $RD/cycles/cyc__${TAG}__mtbench.jsonl ] || \
    cp $GQRD/cycles/cyc__${TAG}__mtbench.jsonl $RD/cycles/ 2>/dev/null
done
python scripts/compute_eagle_rcal_metrics.py --run-dir $RD \
  > $RD/logs/rcal_metrics.log 2>&1
python scripts/_rep3p_stage5_driver.py --phase bootstrap --run-dir $RD \
  > $RD/logs/stage5_bootstrap.log 2>&1
echo REP3P_ACCEPT_DONE
