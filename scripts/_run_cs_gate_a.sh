set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/validate_concat_selective_fp.py --device cuda:0 --num-prompts 8 --max-new-tokens 48
