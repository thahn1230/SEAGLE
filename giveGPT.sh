cd "$(git rev-parse --show-toplevel)" || exit 1

RUN="runs/eagle1_target_draft_precision_method_grid_20260805_184316"
COMMIT="0f6966a"
OUT="eagle1_pmg_review_20260807.tar.gz"
STAGE="$(mktemp -d)"

mkdir -p \
  "$STAGE/repo_metadata" \
  "$STAGE/reports" \
  "$STAGE/run"

echo "=== Building PMG compact review bundle ==="
echo "RUN    = $RUN"
echo "COMMIT = $COMMIT"
echo "OUT    = $OUT"
echo

# ============================================================
# 1. Git provenance
# ============================================================

git rev-parse HEAD \
  > "$STAGE/repo_metadata/HEAD.txt"

git branch --show-current \
  > "$STAGE/repo_metadata/BRANCH.txt"

git status --short \
  > "$STAGE/repo_metadata/GIT_STATUS.txt"

git log --oneline --decorate -30 \
  > "$STAGE/repo_metadata/GIT_LOG_30.txt"

git show \
  --format=fuller \
  --stat \
  --summary \
  "$COMMIT" \
  > "$STAGE/repo_metadata/COMMIT_SUMMARY.txt"

git diff-tree \
  --no-commit-id \
  --name-status \
  -r "$COMMIT" \
  > "$STAGE/repo_metadata/COMMIT_FILES.txt"

git diff "${COMMIT}^" "$COMMIT" -- \
  > "$STAGE/repo_metadata/COMMIT.patch"


# ============================================================
# 2. 해당 commit에서 변경된 코드 파일 실제 복사
# ============================================================

git diff-tree \
  --no-commit-id \
  --name-only \
  -r "$COMMIT" |
while IFS= read -r f; do
    [ -f "$f" ] || continue
    mkdir -p "$STAGE/source/$(dirname "$f")"
    cp "$f" "$STAGE/source/$f"
done


# ============================================================
# 3. 최종 보고서
# ============================================================

find docs -maxdepth 4 -type f \( \
    -iname '*TARGET*DRAFT*PRECISION*METHOD*GRID*.md' -o \
    -iname '*precision*method*grid*.md' -o \
    -iname '*PMG*.md' \
\) -print0 2>/dev/null |
while IFS= read -r -d '' f; do
    mkdir -p "$STAGE/reports/$(dirname "$f")"
    cp "$f" "$STAGE/reports/$f"
done


# ============================================================
# 4. Run directory의 모든 "텍스트 기반" 연구 결과
#
# 포함:
# - CSV / TSV
# - JSON / JSONL
# - YAML
# - Markdown
# - TXT / LOG
# - checksums
#
# 제외:
# - 모델 checkpoint
# - rotation matrix 자체
# - numpy binary
# - profiler binary
# ============================================================

find "$RUN" -type f \( \
    -name '*.csv' -o \
    -name '*.tsv' -o \
    -name '*.json' -o \
    -name '*.jsonl' -o \
    -name '*.yaml' -o \
    -name '*.yml' -o \
    -name '*.toml' -o \
    -name '*.md' -o \
    -name '*.txt' -o \
    -name '*.log' -o \
    -name '*.sha256' \
\) \
! -iname '*nsys*' \
! -iname '*trace*' \
! -iname '*chrome*' \
-print0 |
while IFS= read -r -d '' f; do
    rel="${f#$RUN/}"
    mkdir -p "$STAGE/run/$(dirname "$rel")"
    cp "$f" "$STAGE/run/$rel"
done


# ============================================================
# 5. 특히 중요한 PMG 테이블/selection을 빠짐없이 재확인
# ============================================================

for pattern in \
    'qat_selection*' \
    '*runtime*' \
    '*folding*' \
    '*calibration*' \
    '*target_quality*' \
    '*holm*' \
    '*bootstrap*' \
    '*precision*grid*' \
    '*method*grid*' \
    '*rcal*' \
    '*cycles*' \
    '*final*summary*' \
    'FINAL_OUTPUT*'
do
    find "$RUN" -type f -iname "$pattern" -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
        rel="${f#$RUN/}"
        mkdir -p "$STAGE/run/$(dirname "$rel")"
        cp -n "$f" "$STAGE/run/$rel" 2>/dev/null || true
    done
done


# ============================================================
# 6. Selected checkpoint / rotation 파일의 "경로 + 크기 + checksum"
#    실제 .pt 파일은 복사하지 않음
# ============================================================

{
    echo -e "size_bytes\tsha256\tpath"

    find "$RUN" -type f \( \
        -name '*.pt' -o \
        -name '*.pth' -o \
        -name '*.ckpt' -o \
        -name '*.safetensors' -o \
        -name '*.bin' -o \
        -name '*.npy' -o \
        -name '*.npz' \
    \) -print0 |
    while IFS= read -r -d '' f; do
        size="$(stat -c '%s' "$f")"
        sha="$(sha256sum "$f" | awk '{print $1}')"
        printf "%s\t%s\t%s\n" "$size" "$sha" "$f"
    done
} > "$STAGE/repo_metadata/LARGE_ARTIFACT_MANIFEST.tsv"


# ============================================================
# 7. Selected R_D 3종의 provenance만 별도 추출
# ============================================================

{
    echo "Expected selected R_D artifacts:"
    echo "  RD_T16_HYB_s2v2"
    echo "  RD_T8_HYB_s2"
    echo "  RD_HYB_s2"
    echo
} > "$STAGE/repo_metadata/RD_SELECTED.txt"

find "$RUN" -type f \( \
    -iname '*RD_T16_HYB_s2v2*' -o \
    -iname '*RD_T8_HYB_s2*' -o \
    -iname '*RD_HYB_s2*' \
\) -print0 |
while IFS= read -r -d '' f; do

    # 작은 metadata 파일만 실제 bundle에 포함
    size="$(stat -c '%s' "$f")"

    if [ "$size" -lt $((5*1024*1024)) ]; then
        rel="${f#$RUN/}"
        mkdir -p "$STAGE/run/$(dirname "$rel")"
        cp -n "$f" "$STAGE/run/$rel" 2>/dev/null || true
    fi

    sha256sum "$f" \
      >> "$STAGE/repo_metadata/RD_SELECTED_CHECKSUMS.sha256"
done


# ============================================================
# 8. 기존 최종 168MB bundle provenance
# ============================================================

for f in \
    eagle1_target_draft_precision_method_grid_20260805.tar.gz \
    eagle1_target_draft_precision_method_grid_20260805.tar.gz.sha256
do
    if [ -e "$f" ]; then
        ls -lh "$f" \
          >> "$STAGE/repo_metadata/ORIGINAL_BUNDLE_INFO.txt"

        if [ -f "$f" ]; then
            sha256sum "$f" \
              >> "$STAGE/repo_metadata/ORIGINAL_BUNDLE_CHECKSUM.txt"
        fi
    fi
done


# ============================================================
# 9. Run tree 및 파일 크기 정보
# ============================================================

find "$RUN" -type f \
  -printf '%s\t%p\n' |
sort -nr \
  > "$STAGE/repo_metadata/RUN_FILES_BY_SIZE.tsv"

find "$RUN" -maxdepth 4 -print \
  > "$STAGE/repo_metadata/RUN_TREE.txt"

du -sh "$RUN" \
  > "$STAGE/repo_metadata/RUN_DISK_USAGE.txt"


# ============================================================
# 10. Compact bundle 내부 manifest
# ============================================================

(
    cd "$STAGE" || exit 1

    find . -type f \
      -printf '%s\t%p\n' |
    sort -n \
      > repo_metadata/COMPACT_FILE_MANIFEST.tsv

    find . -type f \
      ! -path './repo_metadata/INTERNAL_CHECKSUMS.sha256' \
      -print0 |
    sort -z |
    xargs -0 sha256sum \
      > repo_metadata/INTERNAL_CHECKSUMS.sha256
)


# ============================================================
# 11. 압축
# ============================================================

tar -C "$STAGE" -czf "$OUT" .

sha256sum "$OUT" | tee "${OUT}.sha256"

echo
echo "=== Result ==="
ls -lh "$OUT" "${OUT}.sha256"

echo
echo "Top-level contents:"
tar -tzf "$OUT" | head -80

rm -rf "$STAGE"