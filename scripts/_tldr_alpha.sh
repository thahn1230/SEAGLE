set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag specT4_a16 --target t4 --datasets mtbench --alpha-override 16.0 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag specT4_a22.6 --target t4 --datasets mtbench --alpha-override 22.6274 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag specT4_a45.25 --target t4 --datasets mtbench --alpha-override 45.2548 --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag specT4_a64 --target t4 --datasets mtbench --alpha-override 64.0 --n-prompts 20 --run-dir  --device cuda:0
echo "[alpha] DONE"
