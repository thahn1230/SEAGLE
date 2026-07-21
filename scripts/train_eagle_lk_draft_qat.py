#!/usr/bin/env python
"""OUT OF SCOPE (scope correction 2026-07-22) — see docs/OUT_OF_SCOPE_QAT.md.

This study is quantization-aware ROTATION learning: every model weight is
frozen; only R_D (residual A) trains. Draft-core QAT was removed from the
study before any QAT training step ran. This entry point is disarmed and
retained for provenance only."""
import sys

sys.exit("OUT OF SCOPE: draft-core QAT is excluded from this study "
         "(docs/OUT_OF_SCOPE_QAT.md). Only the draft rotation R_D may "
         "be trained (scripts/train_eagle_lk_rotation.py).")
