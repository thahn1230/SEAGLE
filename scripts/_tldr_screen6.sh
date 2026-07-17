set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while kill -0 2526307 2>/dev/null; do sleep 60; done
python scripts/eval_eagle_draft_rotation.py --rotation identity --tag drot_none --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag specT4_a16 --target t4 --datasets mtbench --alpha-override 16.0 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag specT4_a22.6 --target t4 --datasets mtbench --alpha-override 22.6274 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag specT4_a45.25 --target t4 --datasets mtbench --alpha-override 45.2548 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag specT4_a64 --target t4 --datasets mtbench --alpha-override 64.0 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation shared --tag gateB_shared_d16 --draft d16 --target t4 --datasets mtbench --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_EAGLE_T4.pt --tag gateB_specT4_d16 --draft d16 --target t4 --datasets mtbench --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
echo "[screen6] DONE"
