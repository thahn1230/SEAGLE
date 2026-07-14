# w4a4_impl_fix — Phase 0 audit (no generation)

Mode: `w4a4_unfused_target_fused_draft_head`. Facts below are code-verified
(fp64, real weights) in `scripts/audit_fc_lm_head.py` /
`docs/fc_projection_fold_audit.md`; restated here for this correctness pass.

## fc projection
- `fc.weight.shape == [4096, 8192]` → **YES** (Linear 8192→4096).
- concat order (cnets.py:592 `torch.cat((inputs_embeds, hidden_states), dim=-1)`)
  = **[embedding, hidden]** → first half `W_e = fc.weight[:, :4096]`, second half
  `W_h = fc.weight[:, 4096:]`. Confirmed.

## Target final norm / gamma_f
- `gamma_f = target.model.norm.weight` (shape 4096, mean 1.772, std 0.116, min
  0.0034, max 2.844 — non-uniform). Stashed BEFORE SpinQuant fusion
  (`stash_original_tensors`); after fusion the live `model.norm.weight` is all
  ones (scale-free).

## Target lm_head: original vs fused
- Original `W_lm = lm_head.weight` [32000, 4096] (pre-rotation, stashed).
- SpinQuant fused head (what the live rotated model carries) =
  `W_lm @ diag(gamma_f) @ R1` (verified equal to the live fused head at
  max-abs 8.7e-5 earlier). It consumes `h_hat` (scale-free rotated).
- **This mode UNFUSES the target tail**: `model.norm` is patched to emit
  `h_unfused = (h_hat @ R1.T) * gamma_f` and `lm_head.weight` is restored to
  `W_lm` (original). Note `h_unfused = (RMSNorm0(x)@R1@R1.T)*gamma_f =
  RMSNorm0(x)*gamma_f = h` (original post-norm hidden). So the target exposes
  the ORIGINAL hidden `h` to the draft.

## Draft lm_head
- Draft owns NO lm_head (no checkpoint key, no module); the head is passed to
  `topK_genrate` at call time. In this mode the draft is given the FUSED head
  `W_lm @ diag(gamma_f) @ R1` (R1.T and gamma_f absorbed), which scores a
  scale-free rotated feature `a@R1`.

## Draft projection runtime pre-transforms (NOT folded)
- First (external) forward: `[R1.T on embedding, Identity on hidden]` — because
  the external hidden `h_unfused` is already in original basis and the draft's
  fc h-block expects... (the mode feeds it unchanged, per spec).
- Recurrent forwards: `[R1.T on embedding, R1.T on recycled hidden]`.
- These are explicit ops on `fc`'s input halves; `fc.weight` is unchanged.

## Skeptic's note on the correctness invariant
Under GREEDY decoding, EAGLE's output is the TARGET's greedy continuation by
construction (a draft token is accepted only if it equals the target argmax).
So Test D (AR-only == target+EAGLE tokens) is determined by the TARGET path, not
the draft. The draft's projection/head choices affect ACCEPTANCE (how many
tokens survive per target forward), not output correctness. This pass therefore
verifies: (1) the unfused target tail is logit-correct (Test A), (2) AR and
EAGLE produce identical greedy tokens (Test D) and identical verifier logits
(Test E), and separately REPORTS the acceptance that the user-specified draft
transforms yield — without claiming it is optimal.
