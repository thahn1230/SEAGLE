set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in t8 t4; do
  python scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_pilot_20260715_0131 --group $g --device cuda:0 --num-prompts 20 --max-new-tokens 64
done
python scripts/run_fixed_tree_target_grader.py --device cuda:0 --num-prompts 12
