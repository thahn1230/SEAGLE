set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "_lk_pilot_[g]3.sh" >/dev/null; do sleep 60; done
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_TV.pt --tag ABL_TV
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_FIXED05.pt --tag ABL_FIXED05
python scripts/eval_eagle_lk_tree.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK_ABL_TOPK64.pt --tag ABL_TOPK64
python scripts/eval_eagle_lk_chain.py --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 20 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --mode t1 --lk-ckpt shared --tag SHARED_RT
echo "[scr_g3] DONE"
