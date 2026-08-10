# Cleanup Actions Log — 2026-08-04

Repo: `~/eagle_spinquant_w4a4` @ `d6fa209` (branch
`exp/seagle-real-int4-fused-epilogue`). Pre-state snapshots in this
directory (GIT_STATUS / FILE_MANIFEST 5684 rows / LARGE_FILES /
CHECKSUMS / TREE). Policy: preservation over space; nothing
git-tracked deleted; no research asset touched.

## Action 1 — duplicate `_latest` bundle copies → symlinks
6 root `*_latest.tar.gz` files were full byte-identical copies of
their timestamped canonicals (verified twice: `cmp` at survey time
AND `cmp` immediately before each swap; sha256 of both sides recorded
in `PRE_CLEANUP_CHECKSUMS.sha256`). Each copy was replaced by a
relative symlink to its canonical — same name resolves to same bytes.
No canonical touched. Space reclaimed ≈ 856 MB.

| replaced copy | canonical (kept) |
|---|---|
| eagle1_generic_qat_pathwise_p3exp_rcal_latest.tar.gz | …_20260731_154652.tar.gz |
| eagle1_learned_rotation_granularity_folding_latest.tar.gz | …_20260803_133218.tar.gz |
| eagle1_official_fromscratch_ptq_vs_qat_latest.tar.gz | …_20260724_213243.tar.gz |
| eagle1_rotated_ep3p_projection_latest.tar.gz | …_20260803_103956.tar.gz |
| ep3p_projection_visualization_latest.tar.gz | …_20260801_104655.tar.gz |
| eagle_ptq_vs_qat_al_latest.tar.gz | …_20260723_143129.tar.gz |

References checked: `_latest` names appear only as display strings in
`scripts/print_*_final_output.py`; symlinks keep them resolvable.
This matches the pre-existing convention (10 other `_latest` names
were already symlinks).

## Action 2 — untracked Python/pytest caches deleted
133 untracked files under `__pycache__/`, `.pytest_cache/` deleted
(list: session scratchpad `cache_deleted.txt`; all regenerable
bytecode/caches). **104 git-tracked cache files kept untouched**
(`scripts/__pycache__`, `tests/__pycache__`, `tests/.pytest_cache` —
historically committed; deleting would dirty tracked state). Emptied
untracked cache dirs removed.

## Explicitly NOT touched
- `checkpoints/` (3.2G FP16 anchor), all `runs/` content (~160G on
  /data), `outputs/`, all timestamped bundles + `.sha256` files,
  extracted `*_review_bundle_*` dirs, `third_party/`,
  `kernels/**/build/` (in-use binaries; mostly git-tracked),
  torch-extensions JIT cache, `giveGPT.sh` local modification (user's),
  zero-byte `__init__.py` / bundle listing files (legitimate),
  stray `runs/*.log` (run provenance).
- Nothing committed; git working state for tracked files unchanged
  except pre-existing modifications the user already had.

## Post-cleanup verification
See `POST_CLEANUP_VERIFICATION.txt` (same directory).
