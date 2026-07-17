set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag spec_T4 --target t8 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_MIXED.pt --tag mixed --target t8 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4KV4.pt --tag spec_T4KV4 --target t8 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag spec_T4 --target t4kv4 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_MIXED.pt --tag mixed --target t4kv4 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T8.pt --tag spec_T8 --target t4kv4 --n-prompts 20 --run-dir  --device cuda:0
echo "[screen5] DONE"
