#!/bin/bash
# R_D EP3-P training launcher: waits for the 5 domain corpus manifests,
# merges them, then runs the 8 training arms over GPUs 1-7 (GPU 0
# excluded per operator policy). Detach with setsid nohup.
set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/RDROT_RUN_DIR)
AF=$(python3 -c "print(4096**0.40)")
AR=$(python3 -c "print(4096**0.45)")

for d in wiki c4 sharegpt gsm8k code; do
  while [ ! -f "$RD/manifests/lkcorpus__rd_t4_greedy_${d}.json" ]; do
    sleep 60
  done
done
python scripts/_rd_merge_corpus.py "$RD"
M="$RD/manifests/lkcorpus__rd_t4_all.json"

COMMON="--run-dir $RD --corpus $M --batch 32 --accum 1 --steps 3000
 --eval-every 500 --teacher t4 --rot residual --kv-bits 16
 --alpha-init $AF --alpha-rec-init $AR"

run_arm () {  # $1 gpu, $2 name, $3 extra args
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 \
    python scripts/train_eagle_lk_rotation.py $COMMON $3 \
    --out "$RD/rotations/$2.pt" \
    > "$RD/logs/train_$2.log" 2>&1
  echo "[launcher] arm $2 rc=$?" >> "$RD/logs/launcher.log"
}

run_arm 1 RD_HYB_s0  "--objective hybrid --seed 0" &
run_arm 2 RD_HYB_s1  "--objective hybrid --seed 1" &
run_arm 3 RD_HYB_s2  "--objective hybrid --seed 2" &
run_arm 4 RD_ACCS_s0 "--objective accsurv --seed 0" &
run_arm 5 RD_ACCS_s1 "--objective accsurv --seed 1" &
run_arm 6 RD_ACCS_s2 "--objective accsurv --seed 2" &
( run_arm 7 RD_AUXG_s0 "--objective hybrid --aux-greedy 0.3 --seed 0"
  run_arm 7 RD_EXPT_s0 "--objective exptau --seed 0" ) &
wait
echo "[launcher] ALL ARMS DONE" >> "$RD/logs/launcher.log"
