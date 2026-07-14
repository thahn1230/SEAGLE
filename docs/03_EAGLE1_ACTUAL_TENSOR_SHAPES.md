# 03 — EAGLE-1 Actual Tensor Shapes (measured)

Captured by `scripts/30_capture_rotated_hidden.py` on the real
Llama-2-7b-chat + yuhuili/EAGLE-llama2-chat-7B during one decoding step. These are
the ACTUAL runtime shapes, not the conceptual sketch in docs/01.

Model geometry: hidden_size D = 4096, vocab V = 32000,
target layers = 32, draft layers = 1.

## Draft input interface (cnets.py:592-593)

Concat order (verified): **cat([embed_e, hidden_h]) -> fc(2D -> D); embedding FIRST**.
Hidden source (verified): **target post-final-RMSNorm last-layer output (outputs[0])**.

| tensor | meaning | shape | dtype |
|---|---|---|---|
| `e` | token embedding branch (draft embed_tokens output; ORIGINAL/unrotated basis) | `[1, 1, 4096]` | `torch.float16` |
| `h_into_draft` | target hidden feature passed to ea_layer.topK_genrate | `[1, 59, 4096]` | `torch.float16` |
| `z` | fc input = concat([e, h]) — embedding first, hidden second | `[1, 1, 8192]` | `torch.float16` |
| `f` | fc output = fused draft feature (hidden_size) | `[1, 1, 4096]` | `torch.float16` |

`z` last-dim = 2·D = 8192; the hidden block is columns `[D:2D]` = `[4096:8192]`.
This is why the Variant-B conjugation folds into `fc.weight[:, 4096:8192]`.

## Rotated target hidden

The SpinQuant-rotated target emits `h_hat` in the R1 basis (gamma_f folded into
lm_head): shape `[1, 59, 4096]`,
basis = R1-rotated residual (gamma_f folded into lm_head).

- ‖h_hat‖ = 491.59
- ‖unrotate(h_hat)‖ = 888.79  (recovers the original-basis feature)

The draft consumes an original-basis hidden, so the adapter applies
`h = (h_hat @ R1^T) * gamma_f` (Variant A) or folds it into fc (Variant B) before
the draft; the draft's predicted features are scored by the ORIGINAL lm_head.

Raw slices saved to `runs/debug_hidden_capture/draft_input_capture.pt`,
metadata to `runs/debug_hidden_capture/shape_meta.json`.
