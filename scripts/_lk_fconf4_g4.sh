set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_KL_s0.pt --tag F_KL_s0 --target t4kv4 --datasets mtbench --n-prompts 80 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_KL_s0.pt --tag F_KL_s0 --target t4kv4 --datasets c4 --n-prompts 200 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_KL_s0.pt --tag F_KL_s0 --target t4kv4 --datasets gsm8k --n-prompts 200 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[fconf4_g4] DONE"
