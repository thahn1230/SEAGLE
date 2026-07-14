set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
cd /home/thahn1230/eagle_spinquant_w4a4
for g in draft branch thead tembed tbody; do
  python scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group $g --device cuda:0 --num-prompts 20 --max-new-tokens 64
done
echo "[compfinal] ALL GROUPS DONE"
