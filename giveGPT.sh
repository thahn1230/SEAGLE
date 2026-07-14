cd ~/eagle_spinquant_w4a4

# 가장 최근 weight+activation 3D run 자동 선택
RUN_DIR="$(ls -td runs/eagle_draft_weight_activation_3d_* 2>/dev/null | head -n 1)"

if [ -z "${RUN_DIR:-}" ]; then
  echo "ERROR: no runs/eagle_draft_weight_activation_3d_* directory found"
  exit 1
fi

echo "Using RUN_DIR=$RUN_DIR"

BUNDLE="eagle_draft_weight_activation_review_bundle_$(date +%Y%m%d_%H%M%S).tar.gz"
STAGE="/tmp/${BUNDLE%.tar.gz}"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# 1) 이번 run 전체 복사: figures + stats + summary 포함
mkdir -p "$STAGE/runs"
cp -a "$RUN_DIR" "$STAGE/runs/"

# 2) 관련 docs
mkdir -p "$STAGE/docs"
for f in \
  docs/eagle_draft_weight_activation_3d_summary.md \
  docs/eagle_draft_activation_3d_summary.md \
  docs/w4a4_draft_distribution_analysis_summary.md \
  docs/fake_w8a8_draft_acceptance_summary.md \
  docs/fake_w4a4_draft_acceptance_summary.md \
  docs/spinquant_draft_pure_r1_summary.md \
  docs/spinquant_draft_pure_r1_audit.md \
  docs/fc_projection_fold_audit.md \
  docs/lm_head_basis_audit.md
do
  [ -f "$f" ] && cp -a "$f" "$STAGE/docs/"
done

# 3) 관련 scripts
mkdir -p "$STAGE/scripts"
for f in \
  scripts/plot_eagle_draft_weight_activation_3d.py \
  scripts/plot_eagle_draft_activation_3d.py \
  scripts/analyze_w4a4_draft_distributions.py \
  scripts/run_fake_w8a8_draft_acceptance.py \
  scripts/run_fake_w4a4_draft_acceptance.py \
  scripts/validate_spinquant_draft_pure_r1.py
do
  [ -f "$f" ] && cp -a "$f" "$STAGE/scripts/"
done

# 4) 관련 src
mkdir -p "$STAGE/src/eagle_spinquant"
for f in \
  src/eagle_spinquant/fake_w4a4_draft.py \
  src/eagle_spinquant/fake_w8a8_draft.py \
  src/eagle_spinquant/spinquant_draft.py \
  src/eagle_spinquant/pure_r1_eagle.py \
  src/eagle_spinquant/w4a4_impl_fix.py \
  src/eagle_spinquant/tail_unfused.py \
  src/eagle_spinquant/rotation_aware.py \
  src/eagle_spinquant/study.py
do
  [ -f "$f" ] && cp -a "$f" "$STAGE/src/eagle_spinquant/"
done

# 5) 비교용 이전 activation-only run
PREV_ACT_RUN="$(ls -td runs/eagle_draft_activation_3d_* 2>/dev/null | head -n 1 || true)"
if [ -n "${PREV_ACT_RUN:-}" ]; then
  cp -a "$PREV_ACT_RUN" "$STAGE/runs/"
fi

# 6) 비교용 acceptance / distribution run
for d in \
  runs/fake_w8a8_draft_20260708_0353 \
  runs/fake_w4a4_draft_20260708_0224 \
  runs/w4a4_distribution_analysis_20260708_0401_n20 \
  runs/spinquant_draft_pure_r1_20260708_0119
do
  [ -d "$d" ] && cp -a "$d" "$STAGE/runs/"
done

# 7) metadata
mkdir -p "$STAGE/meta"
git status --short > "$STAGE/meta/git_status_short.txt" 2>&1 || true
git rev-parse HEAD > "$STAGE/meta/git_head.txt" 2>&1 || true
git diff -- docs scripts src/eagle_spinquant > "$STAGE/meta/git_diff_relevant.patch" 2>&1 || true
find "$STAGE" -maxdepth 10 -type f | sort > "$STAGE/meta/bundle_file_list.txt"

# 8) 대형 checkpoint 제거
find "$STAGE" -type f \( \
  -name "*.safetensors" -o \
  -name "*.bin" -o \
  -name "*.pt" -o \
  -name "*.pth" -o \
  -name "*.gguf" \
\) -delete

# 9) 압축
tar -C /tmp -czf "$BUNDLE" "$(basename "$STAGE")"

echo "Created: $BUNDLE"
ls -lh "$BUNDLE"