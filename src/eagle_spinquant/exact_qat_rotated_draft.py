"""Backward-compatibility shim (scope correction 2026-07-22).

The study is quantization-aware ROTATION learning, not QAT: the only
trainable parameter is R_D (or the residual A). The implementation moved to
exact_quantized_rotation_forward.py; this alias keeps already-armed
pipelines importable. The `train_draft_core` pathway in the implementation
is OUT OF SCOPE for this study (see docs/OUT_OF_SCOPE_QAT.md) and must not
be enabled.
"""
from .exact_quantized_rotation_forward import *          # noqa: F401,F403
from .exact_quantized_rotation_forward import (           # noqa: F401
    ExactQATRotatedDraft, ste_weight_quant, ste_kv4, rotate_half,
    rope_cos_sin)
