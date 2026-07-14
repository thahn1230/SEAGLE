#!/usr/bin/env bash
set -euo pipefail
# GPU 4-7 queue launcher. POLICY: physical GPUs 4,5,6,7 ONLY; one physical GPU
# per model job (CUDA_VISIBLE_DEVICES=<one id> per process); GPUs 5-7 only for
# independent parallel jobs after the GPU-4 smoke passes. Never GPUs 0-3.
#
# Usage:
#   bash scripts/launch_gpu4_7_queue.sh [--dry-run] vicuna-smoke <run_id>
#   bash scripts/launch_gpu4_7_queue.sh [--dry-run] vicuna-main  <run_id>
#
# vicuna-smoke: sequential n=4 gates on GPU 4 (foreground).
# vicuna-main : writes runs/<run_id>/grid/gpu{4..7}.sh (one independent queue
#               per physical GPU) and launches each under nohup.

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
MODE="${1:?mode required: vicuna-smoke | vicuna-main}"
RUN_ID="${2:?run_id required}"
CFG="configs/vicuna_experiment.yaml"
RUNNER="scripts/run_eagle_rotation_study.py"
RUN_DIR="runs/${RUN_ID}"

run() {  # run <physical_gpu> <cmd...>
  local gpu="$1"; shift
  case "$gpu" in 4|5|6|7) ;; *) echo "refusing GPU $gpu (policy 4-7)"; exit 1;; esac
  echo "CUDA_VISIBLE_DEVICES=$gpu $*"
  if [[ "$DRY" == 0 ]]; then CUDA_VISIBLE_DEVICES="$gpu" "$@"; fi
}

write_env_snapshot() {
  mkdir -p "$RUN_DIR"
  { echo "CUDA_VISIBLE_DEVICES policy: one physical GPU in {4,5,6,7} per job"
    date
    nvidia-smi
    python - <<'PYEOF'
import json, sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from eagle_spinquant import logging_utils
print(json.dumps(logging_utils.env_summary(), indent=2))
PYEOF
  } > "$RUN_DIR/environment.txt"
}

case "$MODE" in
vicuna-smoke)
  # Phase 2 gates, GPU 4 only, n=4, quant OFF.
  run 4 python "$RUNNER" --run-id "$RUN_ID" --config "$CFG" \
      --csv acceptance_main --variants stock,vanilla --quant none \
      --rotation none --num-prompts 4 --vanilla-num-prompts 4 \
      --max-new-tokens 64 --tag smoke_fp16
  run 4 python "$RUNNER" --run-id "$RUN_ID" --config "$CFG" \
      --csv acceptance_main --variants naive,A,B2,B --quant none \
      --rotation full --rotation-type random_hadamard --seed 0 \
      --num-prompts 4 --max-new-tokens 64 --tag smoke_rotonly
  ;;

vicuna-main)
  # Phase 3: four independent per-GPU queues. Every entry is a single-GPU job.
  if [[ "$DRY" == 0 ]]; then write_env_snapshot; fi
  mkdir -p "$RUN_DIR/grid"

  gen() {  # gen <gpu> <<'EOF' ... (list of runner arg-strings, one per line)
    local gpu="$1"
    local f="$RUN_DIR/grid/gpu${gpu}.sh"
    { echo '#!/usr/bin/env bash'
      echo 'set -euo pipefail'
      echo "export CUDA_VISIBLE_DEVICES=${gpu}"
      echo "cd '$PROJECT_ROOT'"
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        echo "python $line"
      done
    } > "$f"
    chmod +x "$f"
    echo "wrote $f"
    if [[ "$DRY" == 0 ]]; then
      nohup bash "$f" > "$RUN_DIR/grid/gpu${gpu}.log" 2>&1 &
      echo "launched gpu${gpu}.sh (pid $!)"
    else
      sed 's/^/  [dry] /' "$f"
    fi
  }

  gen 4 <<EOF
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants stock,vanilla --quant none --rotation none --num-prompts 80 --vanilla-num-prompts 40 --max-new-tokens 160 --tag main_fp16
$RUNNER --run-id $RUN_ID --config $CFG --csv gamma_ablation --variants naive,A,A_nogamma --quant none --rotation full --num-prompts 80 --max-new-tokens 160 --tag gamma
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2 --quant none --rotation full --seed 1 --num-prompts 40 --max-new-tokens 160 --tag rotseed1
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2 --quant none --rotation full --seed 2 --num-prompts 40 --max-new-tokens 160 --tag rotseed2
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A --quant none --rotation full --prompt-set wikitext --num-prompts 20 --max-new-tokens 160 --tag wikitext
EOF

  gen 5 <<EOF
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2,B,vanilla --quant none --rotation full --num-prompts 80 --vanilla-num-prompts 40 --max-new-tokens 160 --tag main_rotonly
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2,B,vanilla --quant w4a4 --rotation full --num-prompts 80 --vanilla-num-prompts 40 --max-new-tokens 160 --tag main_w4a4
scripts/eval_ppl_study.py --run-id $RUN_ID --config $CFG --settings fp16,rotonly,w4a4,w4a4kv4
EOF

  gen 6 <<EOF
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2,B,vanilla --quant w4a4kv4 --rotation full --num-prompts 80 --vanilla-num-prompts 40 --max-new-tokens 160 --tag main_w4a4kv4
$RUNNER --run-id $RUN_ID --config $CFG --csv acceptance_main --variants naive,A,B2,B,vanilla --quant w4a16 --rotation full --num-prompts 80 --vanilla-num-prompts 40 --max-new-tokens 160 --tag main_w4a16
scripts/validate_rotation_interface.py --run-id $RUN_ID --config $CFG
EOF

  gen 7 <<EOF
$RUNNER --run-id $RUN_ID --config $CFG --csv depth_sweep --variants stock --quant none --rotation none --tree-depths 2,3,4,5 --num-prompts 40 --max-new-tokens 128 --tag depth_stock
$RUNNER --run-id $RUN_ID --config $CFG --csv depth_sweep --variants A,B2,B --quant none --rotation full --tree-depths 2,3,4,5 --num-prompts 40 --max-new-tokens 128 --tag depth_rot
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant none --rotation r1 --num-prompts 40 --max-new-tokens 128 --tag comp_r1_off
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant none --rotation r1r2 --num-prompts 40 --max-new-tokens 128 --tag comp_r1r2_off
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant none --rotation r1r2r3r4 --num-prompts 40 --max-new-tokens 128 --tag comp_r1234_off
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants stock --quant w4a4 --rotation none --num-prompts 40 --max-new-tokens 128 --tag comp_none_w4a4
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant w4a4 --rotation r1 --num-prompts 40 --max-new-tokens 128 --tag comp_r1_w4a4
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant w4a4 --rotation r1r2 --num-prompts 40 --max-new-tokens 128 --tag comp_r1r2_w4a4
$RUNNER --run-id $RUN_ID --config $CFG --csv rotation_component_ablation --variants A --quant w4a4 --rotation r1r2r3r4 --num-prompts 40 --max-new-tokens 128 --tag comp_r1234_w4a4
EOF
  ;;

*)
  echo "unknown mode $MODE"; exit 1;;
esac
