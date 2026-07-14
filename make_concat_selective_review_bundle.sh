#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/thahn1230/eagle_spinquant_w4a4"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/eagle_spinquant_concat_selective_review_bundle_${TS}.tar.gz"
MANIFEST="$(mktemp)"
META_DIR="artifacts/concat_selective_rotation_study/review_meta_${TS}"

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
    if [[ -d "$path" ]]; then
        find "$path" \
            \( -type f -o -type l \) \
            ! -name '*.tar.gz' \
            ! -name '*.zip' \
            ! -name '*.pt' \
            ! -name '*.pth' \
            ! -name '*.bin' \
            ! -name '*.safetensors' \
            ! -name '*.gguf' \
            ! -path '*/__pycache__/*' \
            ! -path '*/.pytest_cache/*' \
            ! -path '*/.git/*' \
            ! -path '*/checkpoints/*' \
            ! -path '*/cache/*' \
            -print >> "$MANIFEST"
    fi
}

capture_git_repo() {
    local repo="$1"
    local label="$2"

    if [[ -d "$repo" ]] && git -C "$repo" rev-parse --is-inside-work-tree \
        >/dev/null 2>&1; then

        {
            echo "Repository: $repo"
            echo
            echo "HEAD:"
            git -C "$repo" rev-parse HEAD
            echo
            echo "Branch:"
            git -C "$repo" branch --show-current
            echo
            echo "Remotes:"
            git -C "$repo" remote -v || true
            echo
            echo "Status:"
            git -C "$repo" status --short
            echo
            echo "Recent commits:"
            git -C "$repo" log --oneline --decorate -20
        } > "$META_DIR/${label}_git_info.txt"

        git -C "$repo" diff --binary \
            > "$META_DIR/${label}_working_tree.patch" || true

        git -C "$repo" diff --binary --cached \
            > "$META_DIR/${label}_staged.patch" || true

        git -C "$repo" ls-files --others --exclude-standard \
            > "$META_DIR/${label}_untracked_files.txt" || true

        while IFS= read -r file; do
            [[ -z "$file" ]] && continue
            if [[ -f "$repo/$file" ]]; then
                printf '%s\n' "$repo/$file" >> "$MANIFEST"
            fi
        done < <(
            {
                git -C "$repo" diff --name-only HEAD || true
                git -C "$repo" ls-files --others --exclude-standard || true
            } | sort -u
        )
    fi
}

# -------------------------------------------------------------------
# Environment and reproducibility metadata
# -------------------------------------------------------------------

{
    echo "Created: $(date -Is)"
    echo "Hostname: $(hostname)"
    echo "Working directory: $ROOT"
    echo
    echo "CUDA environment:"
    echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-<unset>}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo
    echo "Python:"
    python --version 2>&1 || true
    echo
    echo "NVCC:"
    nvcc --version 2>&1 || true
    echo
    echo "NVIDIA SMI:"
    nvidia-smi 2>&1 || true
} > "$META_DIR/environment_snapshot.txt"

python -m pip freeze \
    > "$META_DIR/pip_freeze.txt" 2>&1 || true

find . -maxdepth 4 \
    \( -type f -o -type l \) \
    ! -path './.git/*' \
    ! -name '*.tar.gz' \
    | sort \
    > "$META_DIR/project_tree.txt"

capture_git_repo "." "main"

# Nested repositories, if present.
for candidate in \
    "repos/EAGLE" \
    "EAGLE" \
    "repos/SpinQuant" \
    "SpinQuant" \
    "/data/thahn1230/quarot"
do
    if [[ -d "$candidate" ]]; then
        label="$(echo "$candidate" | sed 's#^/##; s#[/ ]#_#g')"
        capture_git_repo "$candidate" "$label"
    fi
done

# Diff from the previous B2 implementation branch point when available.
if git cat-file -e 9c59d94^{commit} 2>/dev/null; then
    git diff --binary 9c59d94..HEAD \
        > "$META_DIR/b2_to_concat_selective.patch"

    git log --oneline --decorate 9c59d94..HEAD \
        > "$META_DIR/b2_to_concat_selective_commits.txt"

    git diff --stat 9c59d94..HEAD \
        > "$META_DIR/b2_to_concat_selective_diffstat.txt"
fi

# -------------------------------------------------------------------
# Primary concat-selective evidence
# -------------------------------------------------------------------

add_tree "artifacts/concat_selective_rotation_study"

# Include previous B2 evidence for direct comparison.
if [[ -d "artifacts/b2_split_study" ]]; then
    add_tree "artifacts/b2_split_study"
fi

# -------------------------------------------------------------------
# Reports and architecture documentation
# -------------------------------------------------------------------

if [[ -d "docs" ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find docs -maxdepth 2 -type f \
            \( \
                -iname '*concat*' -o \
                -iname '*selective*' -o \
                -iname '*b2*' -o \
                -iname '*rotation*' -o \
                -iname '*spinquant*' -o \
                -iname '*acceptance*' -o \
                -iname '*architecture*' -o \
                -iname '*basis*' -o \
                -iname '*embedding*' -o \
                -iname '*verifier*' -o \
                -iname '*final*report*' -o \
                -iname '*experiment*protocol*' -o \
                -iname '*audit*' \
            \) | sort
    )
fi

# -------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------

if [[ -d "configs" ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find configs -maxdepth 2 -type f \
            \( \
                -iname '*concat*' -o \
                -iname '*selective*' -o \
                -iname '*b2*' -o \
                -iname '*quant*' -o \
                -iname '*acceptance*' -o \
                -iname '*rotation*' -o \
                -iname '*eval*' \
            \) | sort
    )
fi

# -------------------------------------------------------------------
# Implementation and analysis code
# -------------------------------------------------------------------

add_tree "src/eagle_spinquant"

if [[ -d "src/analysis" ]]; then
    add_tree "src/analysis"
fi

if [[ -d "scripts" ]]; then
    while IFS= read -r file; do
        add_file "$file"
    done < <(
        find scripts -maxdepth 2 -type f \
            \( \
                -iname '*concat*' -o \
                -iname '*selective*' -o \
                -iname '*b2*' -o \
                -iname '*acceptance*' -o \
                -iname '*quant*' -o \
                -iname '*rotation*' -o \
                -iname '*verifier*' -o \
                -iname '*counterfactual*' -o \
                -iname '*analy*' -o \
                -iname '*plot*' -o \
                -iname '*report*' -o \
                -iname '*validate*' \
            \) | sort
    )
fi

# Include all tests because algebra, dispatch and negative controls matter.
add_tree "tests"

# Include top-level reproducibility files.
for file in \
    README.md \
    pyproject.toml \
    setup.py \
    setup.cfg \
    requirements.txt \
    environment.yml \
    commands.sh
do
    add_file "$file"
done

# Add generated metadata.
add_tree "$META_DIR"

# -------------------------------------------------------------------
# Build a deduplicated manifest and archive
# -------------------------------------------------------------------

sort -u "$MANIFEST" -o "$MANIFEST"

{
    echo "Archive: $OUT"
    echo "Created: $(date -Is)"
    echo "File count: $(wc -l < "$MANIFEST")"
    echo
    cat "$MANIFEST"
} > "$META_DIR/archive_manifest.txt"

# Re-add manifest after it was created.
add_tree "$META_DIR"
sort -u "$MANIFEST" -o "$MANIFEST"

tar -czf "$OUT" -T "$MANIFEST"

# -------------------------------------------------------------------
# Integrity checks
# -------------------------------------------------------------------

gzip -t "$OUT"
tar -tzf "$OUT" > "/tmp/concat_selective_bundle_contents_${TS}.txt"

echo
echo "Bundle:"
realpath "$OUT"

echo
echo "Size:"
du -h "$OUT"

echo
echo "Files:"
tar -tzf "$OUT" | wc -l

echo
echo "Integrity:"
echo "gzip: PASS"
echo "tar:  PASS"

echo
echo "Upload this file:"
realpath "$OUT"
