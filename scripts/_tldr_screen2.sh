set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/abl__t4__deployKL_rank_feature_self.pt --tag full_eagle --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/abl__t4__full_dual.pt --tag full_dual --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0
python scripts/eval_eagle_draft_rotation.py --rotation runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/DROT_T4_randominit.pt --tag randominit --target t4 --n-prompts 20 --run-dir runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003 --device cuda:0 || true
echo "[screen2] DONE"
