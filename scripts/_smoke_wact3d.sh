set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/plot_eagle_draft_weight_activation_3d.py \
    --run-dir runs/eagle_draft_weight_activation_3d_smoke_20260711_1108 \
    --num-prompts 2 --max-new-tokens 16 --capture-cycle 1
