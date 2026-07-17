set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag spec_T4 --target t8 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag spec_T4 --target t4kv4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
echo "[tm_a] DONE"
