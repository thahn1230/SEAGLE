set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_KL_s0.pt --tag F_KL_s0 --target t4kv4 --datasets sharegpt --n-prompts 80 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_KL_s0.pt --tag F_KL_s0 --target t4kv4 --datasets humaneval --n-prompts 164 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[fconf5_g0] DONE"
