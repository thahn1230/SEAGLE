set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation shared --tag shared_RT --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/abl__t4__selfrecon.pt --tag selfrecon --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/abl__t4__deployKL.pt --tag deployKL --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
echo "[screen1] DONE"
