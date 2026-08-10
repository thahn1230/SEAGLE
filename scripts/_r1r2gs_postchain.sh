#!/usr/bin/env bash
# Post-chain analyses for the GS R1/R2 study (after jobs_eval completes).
# Usage: _r1r2gs_postchain.sh <A1_CKPT> <A2_CKPT> <A3_CKPT>
#   (the MEDIAN-val-seed checkpoints from tables/median_seeds.json)
# Wave 1 (8 GPUs): RCAL same-proposal captures, 4 arms x (deploy,ref)
#   GPU pairs (0,1) A0 / (2,3) A1 / (4,5) A2 / (6,7) A3, mtbench-80, T4.
# Wave 2: rcal metrics + rcal bootstrap (CPU) ; composed-control ckpt ;
#   mechanism analysis (GPU 0) ; runtime bench A0-A3 (GPUs 1-4) ;
#   composed-arm eval (GPUs 5-7 + 0 as they free via gpuq).
set -uo pipefail
RD=/home/thahn1230/SEAGLE/runs/eagle1_draft_aware_r1_r2_gs_20260810_065036
A1CK="$1"; A2CK="$2"; A3CK="$3"
ALPHA=32.89964245299412
cd /home/thahn1230/SEAGLE
source ~/anaconda3/etc/profile.d/conda.sh
conda activate seagle
export HF_HOME=/data/thahn1230/hf_cache
LOG() { echo "[post $(date +%H:%M:%S)] $*"; }

LOG "wave1: RCAL captures (4 arms x 2 GPUs)"
CUDA_VISIBLE_DEVICES=0,1 python scripts/capture_eagle_proposal_cycles.py \
  --target int4 --draft-cfg d4p3 --alpha $ALPHA --tag A0_BASE \
  --datasets mtbench --n-prompts 80 --run-dir $RD \
  --ref-device cuda:1 > $RD/logs/rcal_A0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=2,3 python scripts/capture_eagle_proposal_cycles.py \
  --target int4 --draft-cfg rot --ckpt "$A1CK" --alpha $ALPHA \
  --tag A1_MED --datasets mtbench --n-prompts 80 --run-dir $RD \
  --ref-device cuda:1 > $RD/logs/rcal_A1.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=4,5 python scripts/capture_eagle_proposal_cycles.py \
  --target int4 --draft-cfg rot --ckpt "$A2CK" --alpha $ALPHA \
  --tag A2_MED --datasets mtbench --n-prompts 80 --run-dir $RD \
  --ref-device cuda:1 > $RD/logs/rcal_A2.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=6,7 python scripts/capture_eagle_proposal_cycles.py \
  --target int4 --draft-cfg rot --ckpt "$A3CK" --alpha $ALPHA \
  --tag A3_MED --datasets mtbench --n-prompts 80 --run-dir $RD \
  --ref-device cuda:1 > $RD/logs/rcal_A3.log 2>&1 &
P3=$!
wait $P0 $P1 $P2 $P3
LOG "wave1 done; computing RCAL metrics + bootstrap"
python scripts/compute_eagle_rcal_metrics.py --run-dir $RD \
  > $RD/logs/rcal_metrics.log 2>&1 || LOG "WARN rcal metrics rc=$?"
python scripts/bootstrap_eagle_rcal.py --run-dir $RD \
  --tags A0_BASE,A1_MED,A2_MED,A3_MED \
  --pairs "A0_BASE:A1_MED;A0_BASE:A2_MED;A1_MED:A3_MED" \
  --dataset mtbench --reps 3000 \
  > $RD/logs/rcal_bootstrap.log 2>&1 || LOG "WARN rcal bootstrap rc=$?"

LOG "wave2: composed ckpt + mechanism + runtime + composed eval"
python scripts/make_r2_control_ckpts.py --mode composed \
  --a1-ckpt "$A1CK" --a2-ckpt "$A2CK" --out-dir $RD/rotations \
  > $RD/logs/composed.log 2>&1 || LOG "WARN composed rc=$?"
COMP=$RD/rotations/COMPOSED_A1R1_A2R2.pt

( export CUDA_VISIBLE_DEVICES=0
  python scripts/analyze_r2_mechanism.py --run-dir $RD \
    --ckpts "A1=$A1CK,A2=$A2CK,A3=$A3CK,A4COMP=$COMP" \
    > $RD/logs/mechanism.log 2>&1 || echo "MECH_FAIL" ) &
M0=$!
( export CUDA_VISIBLE_DEVICES=1
  python scripts/bench_pmg_runtime.py --target int4 --draft-cfg d4p3 \
    --alpha $ALPHA --tag A0_BASE --run-dir $RD \
    > $RD/logs/runtime_A0.log 2>&1 ) &
B0=$!
( export CUDA_VISIBLE_DEVICES=2
  python scripts/bench_pmg_runtime.py --target int4 --draft-cfg rot_ep3p \
    --ckpt "$A1CK" --alpha $ALPHA --tag A1_MED --run-dir $RD \
    > $RD/logs/runtime_A1.log 2>&1 ) &
B1=$!
( export CUDA_VISIBLE_DEVICES=3
  python scripts/bench_pmg_runtime.py --target int4 --draft-cfg rot_ep3p \
    --ckpt "$A2CK" --alpha $ALPHA --tag A2_MED --run-dir $RD \
    > $RD/logs/runtime_A2.log 2>&1 ) &
B2=$!
( export CUDA_VISIBLE_DEVICES=4
  python scripts/bench_pmg_runtime.py --target int4 --draft-cfg rot_ep3p \
    --ckpt "$A3CK" --alpha $ALPHA --tag A3_MED --run-dir $RD \
    > $RD/logs/runtime_A3.log 2>&1 ) &
B3=$!
# composed-arm eval on the remaining GPUs
cat > $RD/configs/jobs_composed_eval.txt <<EOF
python scripts/eval_eagle_acceptance_length.py --target int4 --draft-cfg rot --ckpt $COMP --alpha $ALPHA --tag A4_COMPOSED --datasets mtbench:80 --max-new-tokens 128 --run-dir $RD
python scripts/eval_eagle_acceptance_length.py --target int4 --draft-cfg rot --ckpt $COMP --alpha $ALPHA --tag A4_COMPOSED --datasets gsm8k:200 --max-new-tokens 128 --run-dir $RD
python scripts/eval_eagle_acceptance_length.py --target int4 --draft-cfg rot --ckpt $COMP --alpha $ALPHA --tag A4_COMPOSED --datasets sharegpt:80 --max-new-tokens 128 --run-dir $RD
python scripts/eval_eagle_acceptance_length.py --target int4 --draft-cfg rot --ckpt $COMP --alpha $ALPHA --tag A4_COMPOSED --datasets humaneval:164 --max-new-tokens 128 --run-dir $RD
EOF
python scripts/_r1r2gs_gpuq.py --jobs $RD/configs/jobs_composed_eval.txt \
  --gpus 5,6,7 --log-dir $RD/logs > $RD/logs/gpuq_composed.log 2>&1 &
C0=$!
wait $M0 $B0 $B1 $B2 $B3 $C0
LOG "POSTCHAIN_DONE"
