#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Locate repository
# ----------------------------------------------------------------------

if git rev-parse --show-toplevel >/dev/null 2>&1; then
    ROOT="$(git rev-parse --show-toplevel)"
elif [[ -d /home/thahn1230/eagle_spinquant_w4a4 ]]; then
    ROOT="/home/thahn1230/eagle_spinquant_w4a4"
elif [[ -d /data/thahn1230/eagle_spinquant_w4a4 ]]; then
    ROOT="/data/thahn1230/eagle_spinquant_w4a4"
else
    echo "ERROR: eagle_spinquant_w4a4 repository not found." >&2
    exit 1
fi

cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/data/thahn1230"
mkdir -p "$OUT_DIR"

OUT="$OUT_DIR/eagle_spinquant_bitwidth_al_review_bundle_${TS}.tar.gz"
MANIFEST="$(mktemp)"
META_DIR="artifacts/bitwidth_al_component_causality/review_meta_${TS}"

mkdir -p "$META_DIR"

cleanup() {
    rm -f "$MANIFEST"
    rm -rf "$ROOT/$META_DIR"
}
trap cleanup EXIT

add_file() {
    local path="$1"
    if [[ -f "$path" || -L "$path" ]]; then
        printf '%s\n' "$path" >> "$MANIFEST"
    fi
}

add_tree() {
    local path="$1"

    if [[ ! -d "$path" ]]; then
        return
    fi

    find "$path" \
        \( -type f -o -type l \) \
        ! -path '*/__pycache__/*' \
        ! -path '*/.pytest_cache/*' \
        ! -path '*/.git/*' \
        ! -path '*/huggingface/*' \
        ! -path '*/checkpoints/*' \
        ! -path '*/model_cache/*' \
        ! -path '*/outputs/models/*' \
        ! -name '*.tar.gz' \
        ! -name '*.zip' \
        ! -name '*.safetensors' \
        ! -name '*.gguf' \
        ! -name '*.bin' \
        -size -200M \
        -print >> "$MANIFEST"
}

capture_git_repo() {
    local repo="$1"
    local label="$2"

    if [[ ! -d "$repo" ]]; then
        return
    fi

    if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return
    fi

    {
        echo "Repository: $repo"
        echo
        echo "HEAD:"
        git -C "$repo" rev-parse HEAD
        echo
        echo "Branch:"
        git -C "$repo" branch --show-current
        echo
        echo "Status:"
        git -C "$repo" status --short
        echo
        echo "Remotes:"
        git -C "$repo" remote -v || true
        echo
        echo "Recent commits:"
        git -C "$repo" log --oneline --decorate -30
    } > "$META_DIR/${label}_git_info.txt"

    git -C "$repo" diff --binary \
        > "$META_DIR/${label}_working_tree.patch" || true

    git -C "$repo" diff --binary --cached \
        > "$META_DIR/${label}_staged.patch" || true

    git -C "$repo" ls-files --others --exclude-standard \
        > "$META_DIR/${label}_untracked_files.txt" || true
}

# ----------------------------------------------------------------------
# Environment metadata
# ----------------------------------------------------------------------

{
    echo "Created: $(date -Is)"
    echo "Hostname: $(hostname)"
    echo "Repository root: $ROOT"
    echo "PWD: $(pwd)"
    echo
    echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-<unset>}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo
    echo "Disk usage:"
    df -h "$ROOT" /data 2>&1 || true
    echo
    echo "Python:"
    python --version 2>&1 || true
    echo
    echo "PyTorch:"
    python - <<'PY' 2>&1 || true
import os
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as exc:
    print("PyTorch probe failed:", repr(exc))
PY
    echo
    echo "NVCC:"
    nvcc --version 2>&1 || true
    echo
    echo "nvidia-smi:"
    nvidia-smi 2>&1 || true
} > "$META_DIR/environment_snapshot.txt"

python -m pip freeze \
    > "$META_DIR/pip_freeze.txt" 2>&1 || true

find . -maxdepth 5 \
    \( -type f -o -type l \) \
    ! -path './.git/*' \
    ! -path '*/__pycache__/*' \
    ! -name '*.tar.gz' \
    | sort \
    > "$META_DIR/project_tree.txt"

capture_git_repo "." "main"

for repo in \
    "repos/EAGLE" \
    "EAGLE" \
    "repos/SpinQuant" \
    "SpinQuant" \
    "/data/thahn1230/quarot"
do
    if [[ -d "$repo" ]]; then
        label="$(echo "$repo" | sed 's#^/##; s#[/ ]#_#g')"
        capture_git_repo "$repo" "$label"
    fi
done

# Diff from completed concat-selective baseline.
if git cat-file -e 8a4711c^{commit} 2>/dev/null; then
    git diff --binary 8a4711c..HEAD \
        > "$META_DIR/concat_selective_to_bitwidth.patch"

    git diff --stat 8a4711c..HEAD \
        > "$META_DIR/concat_selective_to_bitwidth_diffstat.txt"

    git log --oneline --decorate 8a4711c..HEAD \
        > "$META_DIR/concat_selective_to_bitwidth_commits.txt"
fi

# ----------------------------------------------------------------------
# Main study evidence
# ----------------------------------------------------------------------

add_tree "artifacts/bitwidth_al_component_causality"

# ----------------------------------------------------------------------
# Prior-study evidence needed for comparison
# ----------------------------------------------------------------------

for prior in \
    "artifacts/concat_selective_rotation_study/summary_tables" \
    "artifacts/concat_selective_rotation_study/fp16_equivalence" \
    "artifacts/concat_selective_rotation_study/verifier_consistency" \
    "artifacts/concat_selective_rotation_study/raw_acceptance" \
    "artifacts/b2_split_study/summary_tables" \
    "artifacts/b2_split_study/fp16_equivalence"
do
    add_tree "$prior"
done

# ----------------------------------------------------------------------
# Reports and audits
# ----------------------------------------------------------------------

for file in \
    "docs/BITWIDTH_AL_COMPONENT_CAUSALITY_FINAL_REPORT.md" \
    "docs/INITIAL_BITWIDTH_AL_COMPONENT_CAUSALITY_AUDIT.md" \
    "docs/TARGET_DRAFT_PARAMETER_OWNERSHIP_AUDIT.md" \
    "docs/CONCAT_SELECTIVE_FINAL_REPORT.md" \
    "docs/CONCAT_SELECTIVE_ARCHITECTURE.md" \
    "docs/CONCAT_SELECTIVE_AR_BASIS_CONTRACT.md" \
    "docs/B2_FINAL_REPORT.md" \
    "docs/B2_SPLIT_BASIS_CONTRACT.md"
do
    add_file "$file"
done

if [[ -d docs ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find docs -maxdepth 2 -type f \
            \( \
                -iname '*bitwidth*' -o \
                -iname '*component*causal*' -o \
                -iname '*target*grader*' -o \
                -iname '*parameter*ownership*' -o \
                -iname '*equivalence*' -o \
                -iname '*verifier*' -o \
                -iname '*projection*sensitiv*' -o \
                -iname '*embedding*' -o \
                -iname '*lm*head*' -o \
                -iname '*final*report*' \
            \) | sort
    )
fi

# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------

for file in \
    "configs/bitwidth_al_component_causality.yaml" \
    "configs/target_grader_thresholds.yaml"
do
    add_file "$file"
done

if [[ -d configs ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find configs -maxdepth 2 -type f \
            \( \
                -iname '*bitwidth*' -o \
                -iname '*component*' -o \
                -iname '*target*grader*' -o \
                -iname '*quant*' -o \
                -iname '*acceptance*' \
            \) | sort
    )
fi

# ----------------------------------------------------------------------
# Relevant implementation
# ----------------------------------------------------------------------

if [[ -d src/eagle_spinquant ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find src/eagle_spinquant -type f \
            ! -path '*/__pycache__/*' \
            \( \
                -iname '*precision*' -o \
                -iname '*quant*' -o \
                -iname '*component*' -o \
                -iname '*embedding*' -o \
                -iname '*lm_head*' -o \
                -iname '*projection*' -o \
                -iname '*concat*' -o \
                -iname '*distribution*' -o \
                -iname '*equivalence*' -o \
                -iname '*verifier*' -o \
                -iname '*rotation*' -o \
                -iname '*spinquant*' \
            \) | sort
    )
fi

if [[ -d src/analysis ]]; then
    add_tree "src/analysis"
fi

# ----------------------------------------------------------------------
# Experiment scripts
# ----------------------------------------------------------------------

if [[ -d scripts ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find scripts -maxdepth 2 -type f \
            \( \
                -iname '*3x3*' -o \
                -iname '*bitwidth*' -o \
                -iname '*component*' -o \
                -iname '*embedding*' -o \
                -iname '*lm_head*' -o \
                -iname '*grader*' -o \
                -iname '*projection*' -o \
                -iname '*distribution*' -o \
                -iname '*equivalence*' -o \
                -iname '*verifier*' -o \
                -iname '*acceptance*' -o \
                -iname '*analy*' -o \
                -iname '*plot*' -o \
                -iname '*validate*' \
            \) | sort
    )
fi

# ----------------------------------------------------------------------
# Tests and reproducibility files
# ----------------------------------------------------------------------

add_tree "tests"

for file in \
    "commands.sh" \
    "README.md" \
    "requirements.txt" \
    "environment.yml" \
    "pyproject.toml" \
    "setup.py" \
    "setup.cfg"
do
    add_file "$file"
done

# Add generated metadata.
add_tree "$META_DIR"

# ----------------------------------------------------------------------
# Manifest and archive
# ----------------------------------------------------------------------

sort -u "$MANIFEST" -o "$MANIFEST"

{
    echo "Review bundle: $OUT"
    echo "Created: $(date -Is)"
    echo "Repository root: $ROOT"
    echo "Git HEAD: $(git rev-parse HEAD)"
    echo "Git branch: $(git branch --show-current)"
    echo "File count: $(wc -l < "$MANIFEST")"
    echo
    cat "$MANIFEST"
} > "$META_DIR/archive_manifest.txt"

add_file "$META_DIR/archive_manifest.txt"
sort -u "$MANIFEST" -o "$MANIFEST"

tar -czf "$OUT" -T "$MANIFEST"

# ----------------------------------------------------------------------
# Integrity and content checks
# ----------------------------------------------------------------------

gzip -t "$OUT"
tar -tzf "$OUT" > "/tmp/bitwidth_al_review_contents_${TS}.txt"

REQUIRED_PATTERNS=(
    "BITWIDTH_AL_COMPONENT_CAUSALITY_FINAL_REPORT.md"
    "target_draft_3x3"
    "component"
    "fixed_tree"
    "verifier"
    "equivalence"
)

echo
echo "Content checks:"
for pattern in "${REQUIRED_PATTERNS[@]}"; do
    if grep -qi "$pattern" "/tmp/bitwidth_al_review_contents_${TS}.txt"; then
        echo "PASS: $pattern"
    else
        echo "WARNING: pattern not found: $pattern"
    fi
done

echo
echo "Bundle:"
realpath "$OUT"

echo
echo "Size:"
du -h "$OUT"

echo
echo "File count:"
tar -tzf "$OUT" | wc -l

echo
echo "Integrity:"
echo "gzip: PASS"
echo "tar:  PASS"

echo
echo "Upload this file:"
realpath "$OUT"
