set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "_lk_pilot_[g]4.sh" >/dev/null; do sleep 60; done
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_ETA07.pt --tag ABL_ETA07
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_ETA1.pt --tag ABL_ETA1
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_ETA10.pt --tag ABL_ETA10
python scripts/eval_eagle_lk_chain.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --mode greedy --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID
python scripts/eval_eagle_lk_chain.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --mode t1 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_D_HYBRID.pt --tag D_HYBRID
echo "[scr_g4] DONE"
