set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation shared --tag gateB_shared_d16 --draft d16 --target t4 --datasets mtbench --n-prompts 20 --run-dir  --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation /DROT_EAGLE_T4.pt --tag gateB_specT4_d16 --draft d16 --target t4 --datasets mtbench --n-prompts 20 --run-dir  --device cuda:0
echo "[gateB] DONE"
