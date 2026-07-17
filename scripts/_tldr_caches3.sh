set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
for T in t4 t8 t4kv4 fp16; do
  python scripts/build_rotation_training_cache.py --target $T --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
done
python scripts/build_rotation_training_cache.py --target t4 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0 --domains wiki --n-per-domain 120
echo "[caches] ALL DONE"
