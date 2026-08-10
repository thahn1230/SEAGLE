#!/usr/bin/env bash
# GS R1/R2 factorial study — pre-registered statistics battery.
# Usage: _r1r2gs_stats.sh <run_dir> <A1_TAG> <A2_TAG> <A3_TAG>
#   where the tags are the MEDIAN-val-seed eval tags (e.g. A1_s1002).
# Families (Holm within each): F1 = A1 vs A0, F2 = A2 vs A0,
# F3 = A3 vs A1, each across mtbench/gsm8k/sharegpt/humaneval.
# Exploratory (separate, labeled): A3 vs A2, A3 vs A0, composed, randR2.
set -euo pipefail
RD="$1"; A1="$2"; A2="$3"; A3="$4"
cd "$(dirname "$0")/.."
for ds in mtbench gsm8k sharegpt humaneval; do
  python scripts/bootstrap_eagle_tau.py --run-dir "$RD" --dataset "$ds" \
    --reps 3000 \
    --pair "F1_r1_effect=A0_BASE@int4:${A1}@int4" \
    --pair "F2_r2_effect=A0_BASE@int4:${A2}@int4" \
    --pair "F3_r2_after_r1=${A1}@int4:${A3}@int4" \
    --pair "X_r1_after_r2=${A2}@int4:${A3}@int4" \
    --pair "X_joint_vs_base=A0_BASE@int4:${A3}@int4"
done
python scripts/holm_adjust_bootstrap.py --run-dir "$RD" --reps 3000 \
  --family "F1_r1_effect=mtbench:F1_r1_effect,gsm8k:F1_r1_effect,sharegpt:F1_r1_effect,humaneval:F1_r1_effect" \
  --family "F2_r2_effect=mtbench:F2_r2_effect,gsm8k:F2_r2_effect,sharegpt:F2_r2_effect,humaneval:F2_r2_effect" \
  --family "F3_r2_after_r1=mtbench:F3_r2_after_r1,gsm8k:F3_r2_after_r1,sharegpt:F3_r2_after_r1,humaneval:F3_r2_after_r1"
echo "[stats] done -> $RD/stats/"
