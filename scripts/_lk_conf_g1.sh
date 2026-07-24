set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag SHARED_RT --target t4kv4 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --datasets c4 --n-prompts 200
echo "[conf_g1] DONE"
