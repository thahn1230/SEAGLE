set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_chain.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s2.pt --tag F_HYBRID_s2 --mode greedy --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_chain.py --lk-ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s2.pt --tag F_HYBRID_s2 --mode t1 --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_chain.py --lk-ckpt shared --tag SHARED_RT40 --mode greedy --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_chain.py --lk-ckpt shared --tag SHARED_RT40 --mode t1 --target t4kv4 --datasets mtbench,c4,gsm8k --n-prompts 40 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo "[xtra_g3] DONE"
