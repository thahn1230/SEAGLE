set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_corrected_20260716_1434 --group t8 --device cuda:0 --num-prompts 80 --max-new-tokens 128 --rotation-kind learned_chat_w4a4kv16
echo "[corr] group t8 EXIT $?"
