set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID_C --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --datasets humaneval --n-prompts 164
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_LR1E4.pt --tag ABL_LR1E4 --target t4kv4 --datasets c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[confb_g5] DONE"
