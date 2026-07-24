set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID_T --target t4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID_T --target t8 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[transfer_g1] DONE"
