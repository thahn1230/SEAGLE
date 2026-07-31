#!/bin/bash
# GQ study phase 3: waits for phase-2 (seeds + p3exp search), then
# cksel -> rebalancing/pathdist -> EP3 selection -> method matrix ->
# RCAL captures -> tree-seq controls -> bootstrap. Idempotent.
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/GQ_RUN_DIR)
AN=checkpoints/eagle1_fresh_fp16_anchor/anchor.pt
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[p3] waiting for phase-2 chain (GQ_PHASE2_DONE)"
until grep -q GQ_PHASE2_DONE outputs/gq_phase2_chain.log 2>/dev/null; do
  sleep 60
done
echo "[p3] phase-2 done -> cksel x6"

# 1. per-seed checkpoint selection (calib grid + mtbench best/final)
i=0
for V in T0 T1; do
  TGT=fp16; [ "$V" = T1 ] && TGT=int4
  for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i python scripts/select_best_qat_ckpt.py \
      --tag GQAT_${V}_s$S --target $TGT --alpha 1.0 \
      --draft-cfg naive_w4a4 --run-dir $RD \
      > outputs/gq_cksel_${V}_s$S.log 2>&1 &
    i=$((i+1))
  done
done
# 2. rebalancing + int4 path distributions (CPU-heavy, GPU 6/7)
CUDA_VISIBLE_DEVICES=6 python scripts/analyze_generic_qat_weight_rebalancing.py \
  --run-dir $RD --variant T0 --anchor $AN --lp3-alpha 45.254834 \
  > outputs/gq_rebal_T0.log 2>&1 &
REB0=$!
CUDA_VISIBLE_DEVICES=7 python scripts/analyze_generic_qat_weight_rebalancing.py \
  --run-dir $RD --variant T1 --anchor $AN --lp3-alpha 32.0 \
  > outputs/gq_rebal_T1.log 2>&1 &
REB1=$!
python scripts/collect_eagle_projection_path_distributions.py \
  --target int4 --run-dir $RD > outputs/gq_pathdist_int4.log 2>&1 &
wait
echo "[p3] cksel + rebalancing done -> EP3 selection"

# 3. EP3-G derivation + EP3-P staged selection
python scripts/eval_eagle1_generic_qat_vs_p3exp.py --phase ep3g \
  --run-dir $RD > outputs/gq_ep3g.log 2>&1
python scripts/eval_eagle1_generic_qat_vs_p3exp.py --phase ep3sel \
  --run-dir $RD --gpus 0,1,2,3,4,5,6,7 > outputs/gq_ep3sel.log 2>&1
echo "[p3] EP3 selection done -> method matrix"

# 4. full-method MT-Bench matrix
python scripts/eval_eagle1_generic_qat_vs_p3exp.py --phase matrix \
  --run-dir $RD --gpus 0,1,2,3,4,5,6,7 > outputs/gq_matrix.log 2>&1
echo "[p3] matrix done -> RCAL captures"

# 5. RCAL captures (2 GPUs/job) + metrics + replay equivalence
python scripts/eval_eagle1_generic_qat_vs_p3exp.py --phase rcal \
  --run-dir $RD --gpus 0,1,2,3,4,5,6,7 > outputs/gq_rcal_caps.log 2>&1
echo "[p3] captures done -> tree-seq controls"

# 6. tree-vs-sequential fidelity controls (2 GPUs each, 4 at a time)
j=0
for TAGTGT in RC_F16:fp16 RC_LP3_T1:int4 RC_GQAT_T1:int4 RC_EP3P_T1:int4; do
  TAG=${TAGTGT%%:*}; TGT=${TAGTGT##*:}
  G0=$((j*2)); G1=$((j*2+1))
  CUDA_VISIBLE_DEVICES=$G0,$G1 python scripts/eval_eagle_tree_sequential_fidelity.py \
    --tag $TAG --target $TGT --dataset mtbench --run-dir $RD \
    --max-prompts 20 > outputs/gq_treeseq_$TAG.log 2>&1 &
  j=$((j+1))
done
wait
echo "[p3] tree-seq done -> bootstrap"

# 7. metrics refresh + bootstrap CIs (T1 primary pairs + T0 pairs)
python scripts/compute_eagle_rcal_metrics.py --run-dir $RD \
  > outputs/gq_rcal_metrics.log 2>&1
python scripts/bootstrap_eagle_rcal.py --run-dir $RD \
  --tags RC_F16,RC_NPTQ_T0,RC_LP3_T0,RC_EP3G_T0,RC_EP3P_T0,RC_GQAT_T0,RC_LP3QAT_T0,RC_F16D_T1,RC_NPTQ_T1,RC_LP3_T1,RC_EP3G_T1,RC_EP3P_T1,RC_GQAT_T1,RC_LP3QAT_T1,RC_LP3RD_T1 \
  --pairs RC_LP3_T1:RC_GQAT_T1,RC_EP3P_T1:RC_GQAT_T1,RC_LP3_T1:RC_EP3P_T1,RC_EP3G_T1:RC_EP3P_T1,RC_GQAT_T1:RC_LP3QAT_T1,RC_EP3P_T1:RC_LP3QAT_T1,RC_LP3_T0:RC_GQAT_T0,RC_EP3P_T0:RC_GQAT_T0,RC_LP3_T0:RC_EP3P_T0,RC_EP3G_T0:RC_EP3P_T0,RC_GQAT_T0:RC_LP3QAT_T0,RC_EP3P_T0:RC_LP3QAT_T0 \
  --dataset mtbench > outputs/gq_bootstrap.log 2>&1
echo GQ_PHASE3_DONE
