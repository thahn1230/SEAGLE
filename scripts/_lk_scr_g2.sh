set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "_lk_pilot_[g]2.sh" >/dev/null; do sleep 60; done
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_F_EXPTAU.pt --tag F_EXPTAU
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_G_ALPHA.pt --tag G_ALPHA
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_TRUSTH03.pt --tag ABL_TRUSTH03
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_TRUSTMED.pt --tag ABL_TRUSTMED
echo "[scr_g2] DONE"
