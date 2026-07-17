#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash pack_lras_review_bundle.sh
#
# Optional:
#   INCLUDE_ROTATION=1 bash pack_lras_review_bundle.sh
#
# The raw rotation checkpoint is excluded by default.
# Set INCLUDE_ROTATION=1 when it must be included.

ROOT="${1:-$(git rev-parse --show-toplevel)}"
ROOT="$(cd "${ROOT}" && pwd)"

RUN_REL="runs/eagle_learned_rotation_al_sensitivity_20260716_192113"
REPORT_REL="docs/EAGLE_LEARNED_ROTATION_AL_AND_COMPONENT_SENSITIVITY_REPORT.md"
ROTATION_REL="outputs/rotations/learned_chat_w4a4kv16/R.bin"

EXPECTED_BRANCH="exp/eagle1-learned-rotation-al-sensitivity"
EXPECTED_COMMIT="fc02575"
EXPECTED_ROTATION_SHA256="4b7e91d2a7531bb8ccde55d35299d3bde38b0f849ebcca951c9320e42a48fa6e"

TARGET_MODEL="meta-llama/Llama-2-7b-chat-hf"
TARGET_REVISION="f5db02db724555f92da89c216ac04704f23d4590"
DRAFT_MODEL="yuhuili/EAGLE-llama2-chat-7B"
DRAFT_REVISION="44e37ec383348306fe9b1dfe7c79e145c96db3d0"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/eagle_lras_interpretation_bundle_${STAMP}.tar.gz"
LATEST="${ROOT}/eagle_lras_interpretation_bundle_latest.tar.gz"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p \
    "${STAGE}/experiment_files" \
    "${STAGE}/source_at_commit" \
    "${STAGE}/git_metadata" \
    "${STAGE}/environment" \
    "${STAGE}/rotation_metadata"

cd "${ROOT}"

if ! git cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null; then
    echo "ERROR: commit ${EXPECTED_COMMIT} is not available in this repository." >&2
    exit 1
fi

FULL_COMMIT="$(git rev-parse "${EXPECTED_COMMIT}^{commit}")"
CURRENT_BRANCH="$(git branch --show-current || true)"
RUN_ABS="${ROOT}/${RUN_REL}"
REPORT_ABS="${ROOT}/${REPORT_REL}"
ROTATION_ABS="${ROOT}/${ROTATION_REL}"

if [[ ! -d "${RUN_ABS}" ]]; then
    echo "ERROR: run directory not found: ${RUN_ABS}" >&2
    exit 1
fi

copy_file_preserving_path() {
    local src="$1"
    local destination_root="$2"
    local rel

    [[ -f "${src}" ]] || return 0

    if [[ "${src}" == "${ROOT}/"* ]]; then
        rel="${src#${ROOT}/}"
    else
        rel="$(basename "${src}")"
    fi

    mkdir -p "${destination_root}/$(dirname "${rel}")"
    cp -a "${src}" "${destination_root}/${rel}"
}

cat > "${STAGE}/BUNDLE_OVERVIEW.txt" <<EOF
EAGLE Learned-Rotation AL Sensitivity Study
===========================================

Run directory:
  ${RUN_REL}

Expected branch:
  ${EXPECTED_BRANCH}

Study commit:
  ${FULL_COMMIT}

Current worktree branch when packed:
  ${CURRENT_BRANCH}

Target:
  ${TARGET_MODEL}
  revision ${TARGET_REVISION}

Draft:
  ${DRAFT_MODEL}
  revision ${DRAFT_REVISION}

Learned rotation:
  ${ROTATION_REL}
  expected SHA256 ${EXPECTED_ROTATION_SHA256}

Evaluation:
  MT-Bench 80 prompts
  fixed prompt order
  seed 0
  greedy decoding
  128 new tokens

Acceptance metric:
  tau = accepted draft tokens + 1 verification bonus
  AL = mean(tau)

Primary results to inspect:
  1. Target/Draft FP16-W8A8-W4A4 3x3 AL matrix
  2. Draft component-wise W4A4 sensitivity
  3. Projection first-only versus recurrent-only sensitivity
  4. Shared-scale versus separate-scale projection
  5. Embedding-scaled shared-scale projection
  6. Alpha calibration
  7. Top-1 agreement and reconstruction NMSE
  8. Execution-path exact-match diagnostics
  9. Stop-gate and regression-test outputs

Raw model checkpoints are intentionally excluded.
Raw rotation checkpoint included: ${INCLUDE_ROTATION:-0}
EOF

# ---------------------------------------------------------------------------
# 1. Final report
# ---------------------------------------------------------------------------

copy_file_preserving_path "${REPORT_ABS}" "${STAGE}/experiment_files"

# Include other closely related reports when present.
for f in \
    "${ROOT}/docs/SPINQUANT_PPL_REPRODUCTION_AND_FIX_REPORT.md" \
    "${ROOT}/docs/HANDOFF_TO_EXECUTOR.md"
do
    copy_file_preserving_path "${f}" "${STAGE}/experiment_files"
done

# ---------------------------------------------------------------------------
# 2. Run outputs
#
# Include text results, tables, logs, configurations and plots.
# Large model/checkpoint files and existing archives are excluded.
# ---------------------------------------------------------------------------

while IFS= read -r -d '' f; do
    copy_file_preserving_path "${f}" "${STAGE}/experiment_files"
done < <(
    find "${RUN_ABS}" -type f -size -64M \
        \( \
            -iname '*.json'  -o \
            -iname '*.jsonl' -o \
            -iname '*.csv'   -o \
            -iname '*.tsv'   -o \
            -iname '*.yaml'  -o \
            -iname '*.yml'   -o \
            -iname '*.toml'  -o \
            -iname '*.md'    -o \
            -iname '*.txt'   -o \
            -iname '*.log'   -o \
            -iname '*.out'   -o \
            -iname '*.err'   -o \
            -iname '*.png'   -o \
            -iname '*.svg'   -o \
            -iname '*.pdf' \
        \) \
        -print0
)

# Include small numeric artifacts only when their names indicate that they
# contain analysis outputs rather than model weights.
while IFS= read -r -d '' f; do
    base="$(basename "${f}")"

    if [[ "${base}" =~ (accept|matrix|metric|summary|component|projection|embedding|alpha|calibr|nmse|top1|exact|gate|test|scale|sensitivity) ]]; then
        copy_file_preserving_path "${f}" "${STAGE}/experiment_files"
    fi
done < <(
    find "${RUN_ABS}" -type f -size -32M \
        \( \
            -iname '*.npy' -o \
            -iname '*.npz' -o \
            -iname '*.pt'  -o \
            -iname '*.pth' -o \
            -iname '*.pkl' \
        \) \
        -print0
)

# ---------------------------------------------------------------------------
# 3. Git provenance and code changes
# ---------------------------------------------------------------------------

{
    echo "expected_branch=${EXPECTED_BRANCH}"
    echo "current_branch=${CURRENT_BRANCH}"
    echo "expected_short_commit=${EXPECTED_COMMIT}"
    echo "resolved_commit=${FULL_COMMIT}"
    echo
    git status --short --branch
} > "${STAGE}/git_metadata/status.txt"

git remote -v \
    > "${STAGE}/git_metadata/remotes.txt" 2>&1 || true

git submodule status --recursive \
    > "${STAGE}/git_metadata/submodules.txt" 2>&1 || true

git show \
    --no-ext-diff \
    --format=fuller \
    --summary \
    --stat \
    "${FULL_COMMIT}" \
    > "${STAGE}/git_metadata/commit_show.txt"

BASE_COMMIT=""

for ref in origin/main origin/master main master; do
    if git rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
        BASE_COMMIT="$(git merge-base "${FULL_COMMIT}" "${ref}")"
        echo "base_reference=${ref}" \
            > "${STAGE}/git_metadata/base_commit.txt"
        break
    fi
done

if [[ -z "${BASE_COMMIT}" ]]; then
    BASE_COMMIT="$(git rev-parse "${FULL_COMMIT}^")"
    echo "base_reference=first_parent" \
        > "${STAGE}/git_metadata/base_commit.txt"
fi

echo "base_commit=${BASE_COMMIT}" \
    >> "${STAGE}/git_metadata/base_commit.txt"

git diff \
    --stat \
    "${BASE_COMMIT}" "${FULL_COMMIT}" \
    > "${STAGE}/git_metadata/branch_diff_stat.txt"

git diff \
    --name-status \
    "${BASE_COMMIT}" "${FULL_COMMIT}" \
    > "${STAGE}/git_metadata/changed_files.txt"

git diff \
    --no-ext-diff \
    --unified=40 \
    "${BASE_COMMIT}" "${FULL_COMMIT}" \
    -- \
    '*.py' '*.sh' '*.md' '*.json' '*.yaml' '*.yml' '*.toml' \
    '*.cpp' '*.cc' '*.c' '*.cu' '*.h' '*.hpp' \
    > "${STAGE}/git_metadata/source_changes.patch"

# Copy relevant changed source files exactly as stored in the study commit.
while IFS= read -r file; do
    [[ -n "${file}" ]] || continue

    case "${file}" in
        *.py|*.sh|*.md|*.json|*.yaml|*.yml|*.toml|*.txt|\
        *.cpp|*.cc|*.c|*.cu|*.h|*.hpp)
            ;;
        *)
            continue
            ;;
    esac

    if git cat-file -e "${FULL_COMMIT}:${file}" 2>/dev/null; then
        size="$(git cat-file -s "${FULL_COMMIT}:${file}")"

        if (( size <= 8 * 1024 * 1024 )); then
            mkdir -p "${STAGE}/source_at_commit/$(dirname "${file}")"
            git show "${FULL_COMMIT}:${file}" \
                > "${STAGE}/source_at_commit/${file}"
        fi
    fi
done < <(
    git diff --diff-filter=ACMR --name-only \
        "${BASE_COMMIT}" "${FULL_COMMIT}"
)

# Include common experiment entry points/configurations even if unchanged.
for rel in \
    pyproject.toml \
    requirements.txt \
    environment.yml \
    setup.py \
    setup.cfg
do
    if git cat-file -e "${FULL_COMMIT}:${rel}" 2>/dev/null; then
        mkdir -p "${STAGE}/source_at_commit/$(dirname "${rel}")"
        git show "${FULL_COMMIT}:${rel}" \
            > "${STAGE}/source_at_commit/${rel}"
    fi
done

# ---------------------------------------------------------------------------
# 4. Rotation provenance
# ---------------------------------------------------------------------------

if [[ -f "${ROTATION_ABS}" ]]; then
    ACTUAL_ROTATION_SHA256="$(
        sha256sum "${ROTATION_ABS}" | awk '{print $1}'
    )"

    {
        echo "path=${ROTATION_REL}"
        echo "expected_sha256=${EXPECTED_ROTATION_SHA256}"
        echo "actual_sha256=${ACTUAL_ROTATION_SHA256}"

        if [[ "${ACTUAL_ROTATION_SHA256}" == "${EXPECTED_ROTATION_SHA256}" ]]; then
            echo "sha256_check=PASS"
        else
            echo "sha256_check=FAIL"
        fi

        echo
        stat "${ROTATION_ABS}"
    } > "${STAGE}/rotation_metadata/rotation_checkpoint.txt"

    if [[ "${INCLUDE_ROTATION:-0}" == "1" ]]; then
        copy_file_preserving_path \
            "${ROTATION_ABS}" \
            "${STAGE}/experiment_files"
    fi
else
    {
        echo "path=${ROTATION_REL}"
        echo "expected_sha256=${EXPECTED_ROTATION_SHA256}"
        echo "status=MISSING"
    } > "${STAGE}/rotation_metadata/rotation_checkpoint.txt"
fi

# ---------------------------------------------------------------------------
# 5. Current environment snapshot
# ---------------------------------------------------------------------------

{
    date --iso-8601=seconds
    uname -a
} > "${STAGE}/environment/system.txt" 2>&1 || true

{
    python --version
    python - <<'PY'
import platform
import sys

print("executable:", sys.executable)
print("python:", sys.version)
print("platform:", platform.platform())

try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch_error:", repr(exc))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as exc:
    print("transformers_error:", repr(exc))
PY
} > "${STAGE}/environment/python_runtime.txt" 2>&1 || true

python -m pip freeze \
    > "${STAGE}/environment/pip_freeze.txt" 2>&1 || true

nvidia-smi -L \
    > "${STAGE}/environment/nvidia_smi_list.txt" 2>&1 || true

nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader \
    > "${STAGE}/environment/nvidia_gpu_inventory.csv" 2>&1 || true

# ---------------------------------------------------------------------------
# 6. Bundle manifest and compression
# ---------------------------------------------------------------------------

(
    cd "${STAGE}"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        > SHA256SUMS
)

tar -C "${STAGE}" -czf "${OUT}" .

ln -sfn "$(basename "${OUT}")" "${LATEST}"

echo
echo "Created:"
echo "  ${OUT}"
echo
echo "Latest symlink:"
echo "  ${LATEST}"
echo
echo "Size:"
du -h "${OUT}"