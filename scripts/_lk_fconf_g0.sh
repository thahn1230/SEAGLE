set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s2.pt --tag F_HYBRID_s2 --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --datasets c4 --n-prompts 200
echo [fconf_g0] DONE
