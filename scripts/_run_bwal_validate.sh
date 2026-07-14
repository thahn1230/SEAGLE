set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in stock t8 t4; do
  python scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_validate_20260715_0100 --group $g --device cuda:0 --num-prompts 2 --max-new-tokens 32
done
