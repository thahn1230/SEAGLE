set -euo pipefail
export CUDA_VISIBLE_DEVICES=7
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/plot_eagle_draft_weight_activation_3d.py \
    --run-dir runs/eagle_draft_weight_activation_3d_20260711_1127 \
    --num-prompts 8 --max-new-tokens 48 --capture-cycle 1
