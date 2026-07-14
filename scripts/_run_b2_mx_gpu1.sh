set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in rot quant_w4a16; do
  python scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group $g --device cuda:1 --num-prompts 20 --max-new-tokens 64
done
