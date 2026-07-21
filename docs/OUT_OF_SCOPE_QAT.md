# OUT OF SCOPE — QAT / draft-weight training (scope correction 2026-07-22)

Operator directive: this study is **quantization-aware rotation learning**,
not QAT. Every pretrained parameter (target weights, R_T, draft embedding,
projections, AR decoder, LM head, all norms) stays bitwise frozen. The only
trainable parameter is the draft rotation R_D — parameterized directly or
as the residual `R_D = R_T · C(A)`, A skew. P3 alpha is selected by a
CALIBRATION SWEEP, never trained jointly in the main experiments.

The final research question:

> Can an exact-path, full-vocabulary, acceptance-aware orthogonal draft
> rotation R_D improve over the shared target rotation R_T while every
> model weight remains frozen?

## Recorded, disarmed, excluded

| Item | Status |
|---|---|
| `scripts/train_eagle_lk_draft_qat.py` | disarmed (exits with scope error); retained for provenance |
| `train_draft_core=True` pathway in `exact_quantized_rotation_forward.py` | retained in code, never enabled; no artifact used it |
| Stage-2 candidates I (`LK2_QAT_I_sharedRT`) / J (`LK2_QAT_J_localRD`) | removed from queues BEFORE any training step ran; no checkpoints exist |
| Core-lr sweeps / capacity controls C2–C4 / LoRA-adapters | never implemented as runs |
| Rotation-vs-QAT comparisons, QAT upper-bound conclusions | excluded from the report |
| Trainable-alpha pilot `LK_G_ALPHA` (joint scalar alpha) | trained BEFORE the correction; kept ONLY as an optional scalar-only ablation, flagged pending approval; excluded from main conclusions. Stage-2 `LK2_ALPHA_s0` replaced by a rotation-only run |

Module renamed: `exact_qat_rotated_draft.py` → `exact_quantized_rotation_forward.py`
(class `ExactQuantizedRotationForward`; old names kept as import shims for
already-armed pipelines).

Alpha policy for finalists: post-training calibration sweep over
{8, 11.31, 16, 22.63, 32, 45.25, 64, 90.51, 128} (globally foldable),
applied identically to shared R_T and every candidate before comparison.
