#!/bin/bash
# LRGF Question B: waits for the component audit, then teacher cache,
# smoke, and the learned-rotation training grid. Idempotent.
set -u
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/LRGF_RUN_DIR)
export TMPDIR=/data/thahn1230/tmp CUDA_DEVICE_ORDER=PCI_BUS_ID
TR="python scripts/train_eagle_learned_rotation.py --run-dir $RD"

until grep -q CMP_AUDIT_DONE2 $RD/logs/component_audit.log 2>/dev/null; do
  sleep 60
done
echo "[rotchain] audit done -> teacher cache"
[ -f $RD/tensors/teacher_cache.pt ] || \
  CUDA_VISIBLE_DEVICES=1 $TR --mode precompute-teachers --n-rows 256 \
    > $RD/logs/teacher_cache.log 2>&1
echo "[rotchain] cache ready -> smoke"
CUDA_VISIBLE_DEVICES=1 $TR --objective eagle --param givens \
  --steps 10 --tag SMOKE --ckpt-every 0 \
  > $RD/logs/rot_smoke.log 2>&1 || { echo ROT_SMOKE_FAIL; exit 1; }
echo "[rotchain] smoke ok -> training grid"

# grid: objectives x parameterizations (seed 0), pathwise + fixed-init
# variants for the flagship objective; 3 seeds for finalists later
i=0
launch() {
  G=$((i % 8)); i=$((i+1))
  setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G \
    nohup $TR $@ > $RD/logs/rot_$(echo "$@" | tr -s ' /' '_').log 2>&1 &
}
wait_all() { while pgrep -f "train_eagle_learned_rotation.*--objective" >/dev/null; do sleep 60; done; }

for OBJ in nmse eagle acc rc hybrid; do
  launch --objective $OBJ --param cayley --block 32 --steps 400 --seed 0
done
launch --objective eagle --param givens --steps 400 --seed 0
launch --objective eagle --param householder --hh-k 16 --steps 400 --seed 0
launch --objective eagle --param cayley --block 32 --steps 400 --seed 0 --fixed-init
wait_all
echo "[rotchain] wave1 done"
for S in 1 2; do
  launch --objective eagle --param cayley --block 32 --steps 400 --seed $S
  launch --objective rc --param cayley --block 32 --steps 400 --seed $S
done
launch --objective eagle --param cayley --block 32 --steps 400 --seed 0 --pathwise
launch --objective rc --param cayley --block 32 --steps 400 --seed 0 --pathwise
wait_all
echo LRGF_ROT_TRAIN_DONE
