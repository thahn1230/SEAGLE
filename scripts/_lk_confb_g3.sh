set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID_C --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --datasets sharegpt --n-prompts 80
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID_C --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --datasets mtbench --n-prompts 80
echo "[confb_g3] DONE"
