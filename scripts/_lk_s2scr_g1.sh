set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_EXPTAU_s0.pt --tag S2_LK2_EXPTAU_s0
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s1.pt --tag S2_LK2_HYBRID_s1
echo [s2scr_g1] DONE
