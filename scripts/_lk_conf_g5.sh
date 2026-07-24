set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_LR1E4.pt --tag ABL_LR1E4 --target t4kv4 --datasets c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag SHARED_RT --target t4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag SHARED_RT --target t8 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[conf_g5] DONE"
