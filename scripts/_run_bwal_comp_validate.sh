set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in draft branch thead tembed tbody; do
  python scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_validate_20260715_0521 --group $g --device cuda:0 --num-prompts 2 --max-new-tokens 32
done
echo "[compval] ALL GROUPS DONE"
