set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_chain.py --lk-ckpt shared --tag SHARED_RT --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --mode greedy --datasets mtbench,c4,gsm8k --n-prompts 40
python scripts/eval_eagle_lk_chain.py --lk-ckpt shared --tag SHARED_RT --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --mode t1 --datasets mtbench,c4,gsm8k --n-prompts 40
echo "[conf_g4] DONE"
