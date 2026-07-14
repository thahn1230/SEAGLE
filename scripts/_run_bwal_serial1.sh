set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
RUN=$(cat /tmp/claude-1010/-home-thahn1230/613ac1c1-e9b0-4374-bf6e-a14c6b0d2cf8/scratchpad/bwal_val_rundir.txt)
for g in stock t8 t4; do
  python scripts/run_target_draft_3x3_matrix.py --run-dir $RUN --group $g --device cuda:0 --num-prompts 2 --max-new-tokens 32
done
python scripts/run_acceptance_equivalence_forensics.py --device cuda:0 --num-prompts 20 --max-new-tokens 64
