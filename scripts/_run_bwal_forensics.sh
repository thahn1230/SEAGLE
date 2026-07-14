set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/run_acceptance_equivalence_forensics.py --device cuda:1 --num-prompts 20 --max-new-tokens 64
