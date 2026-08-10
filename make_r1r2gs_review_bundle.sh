#!/usr/bin/env bash
# Compact review bundle for the GS R1/R2 draft-aware study.
# Includes source diffs, configs, reports, tables, stats, logs, manifests,
# figures, and a checkpoint MANIFEST (path + size + sha256) — never the
# multi-GB checkpoints or model caches themselves.
set -euo pipefail
cd /home/thahn1230/SEAGLE
RD=runs/eagle1_draft_aware_r1_r2_gs_20260810_065036
DATE=$(date +%Y%m%d)
OUT=eagle1_draft_aware_r1_r2_gs_review_${DATE}.tar.gz
STAGE=$(mktemp -d)
B=$STAGE/bundle
mkdir -p $B/{source,run,docs}

# ---- source: the diff plus full copies of changed/new files ----------
git diff HEAD > $B/source/working_tree.diff || true
git status --short > $B/source/git_status.txt
git log -10 --oneline > $B/source/git_log.txt
git rev-parse HEAD > $B/source/git_head.txt
git branch --show-current > $B/source/git_branch.txt
mkdir -p $B/source/files
for f in $(git status --porcelain | awk '{print $2}' | grep -E '\.(py|sh|md|yaml)$' || true); do
  [ -f "$f" ] && install -D "$f" "$B/source/files/$f"
done

# ---- run artifacts (no checkpoints, no corpus shards) ----------------
for d in audit configs gradchecks manifests tables stats rcal geometry runtime figures reports logs; do
  [ -d "$RD/$d" ] && cp -r "$RD/$d" "$B/run/" 2>/dev/null || true
done
# shards are small CSVs of per-cycle taus — the raw evidence for every AL
mkdir -p $B/run/shards && cp $RD/shards/*.csv $B/run/shards/ 2>/dev/null || true
# cycle JSONLs can be large; keep them but compressed
if [ -d "$RD/cycles" ]; then
  mkdir -p $B/run/cycles
  for f in $RD/cycles/*.jsonl; do
    [ -f "$f" ] && gzip -c "$f" > "$B/run/cycles/$(basename $f).gz"
  done
fi
# drop oversized logs (corpus/training stdout) but keep their tails
find $B/run/logs -type f -size +2M -print0 2>/dev/null | while IFS= read -r -d '' f; do
  tail -c 200000 "$f" > "$f.tail" && rm "$f" && mv "$f.tail" "$f"
done

# ---- checkpoint manifest (paths + sizes + sha256), NOT the weights ---
{
  echo "# rotation checkpoints (NOT included in this bundle)"
  printf "%-52s %12s  %s\n" "file" "bytes" "sha256"
  for f in $RD/rotations/*.pt outputs/rotations/learned_chat_w4a4kv16/R.bin; do
    [ -f "$f" ] || continue
    printf "%-52s %12s  %s\n" "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
} > $B/run/CHECKPOINT_MANIFEST.txt

# ---- docs -------------------------------------------------------------
for f in docs/EAGLE1_DRAFT_AWARE_R1_R2_GS_STUDY.md \
         docs/EAGLE1_DRAFT_AWARE_R1_R2_GS_AUDIT.md; do
  [ -f "$f" ] && cp "$f" $B/docs/
done

tar -czf "$OUT" -C "$STAGE" bundle
rm -rf "$STAGE"
echo "bundle: $OUT  ($(du -h "$OUT" | cut -f1))"
sha256sum "$OUT"
