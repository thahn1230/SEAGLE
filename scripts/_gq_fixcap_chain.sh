#!/bin/bash
# Recapture RC_LP3RD_T1 + RC_F16D_T1 after capture-config fix (rot
# alpha fallback, restored interface for stock@int4), then refresh
# metrics + bootstrap + plots + final output draft. Waits for phase 4.
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/GQ_RUN_DIR)
LK=runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_AUXG_s0.pt
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID

until grep -q GQ_PHASE4_DONE outputs/gq_phase4_chain.log 2>/dev/null; do
  sleep 60
done
echo "[fixcap] phase-4 done -> recapture broken tags"
rm -f $RD/cycles/cyc__RC_LP3RD_T1__mtbench.jsonl \
      $RD/cycles/cyc__RC_F16D_T1__mtbench.jsonl

CUDA_VISIBLE_DEVICES=0,1 python scripts/capture_eagle_proposal_cycles.py \
  --run-dir $RD --target int4 --draft-cfg rot --ckpt $LK \
  --tag RC_LP3RD_T1 --datasets mtbench --n-prompts 40 \
  --max-new-tokens 128 > outputs/gq_fixcap_lp3rd.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python scripts/capture_eagle_proposal_cycles.py \
  --run-dir $RD --target int4 --draft-cfg stock \
  --tag RC_F16D_T1 --datasets mtbench --n-prompts 40 \
  --max-new-tokens 128 > outputs/gq_fixcap_f16d.log 2>&1 &
wait
echo "[fixcap] recapture done -> metrics/bootstrap/plots refresh"

python scripts/compute_eagle_rcal_metrics.py --run-dir $RD \
  >> outputs/gq_rcal_metrics.log 2>&1
python scripts/bootstrap_eagle_rcal.py --run-dir $RD \
  --tags RC_LP3RD_T1,RC_F16D_T1 --dataset mtbench \
  >> outputs/gq_bootstrap.log 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/replay_eagle_proposals_reference_target.py \
  --run-dir $RD --tag RC_LP3RD_T1 --max-prompts 4 \
  >> outputs/gq_fixcap_lp3rd.log 2>&1
python scripts/plot_eagle1_rcal.py --run-dir $RD \
  > outputs/gq_plots_rcal.log 2>&1
python scripts/plot_eagle1_p3exp_3d.py --run-dir $RD \
  > outputs/gq_plots_3d.log 2>&1
python scripts/print_generic_qat_p3exp_rcal_final_output.py $RD \
  > $RD/FINAL_OUTPUT_DRAFT.txt 2>&1
echo GQ_FIXCAP_DONE
