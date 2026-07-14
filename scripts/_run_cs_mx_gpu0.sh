set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in stock quant; do
  python scripts/run_concat_selective_acceptance_matrix.py --run-dir runs/cs_matrix_20260714_2111 --group $g --device cuda:0 --num-prompts 20 --max-new-tokens 64
done
