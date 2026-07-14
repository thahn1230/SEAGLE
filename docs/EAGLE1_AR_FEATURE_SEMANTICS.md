# EAGLE-1 AR-HEAD FEATURE SEMANTICS (code-verified, EAGLE v1 @ 4a9cf3a)

## What the AR head emits

The draft "AR head" is the single `LlamaDecoderLayer` (`ea_layer.layers[0]`,
note: **no `input_layernorm`** on layer 0 — `cnets.py:386`) after the fc
projection. Inside `topK_genrate` the decoder output `out_hidden` is:

1. scored directly: `last_headout = head(out_hidden...)` — **no final RMSNorm,
   no gamma applied** between the decoder output and the LM head;
2. recycled directly as the next `hidden_states` input to `self(...)`.

So mechanically the recurrent feature is the raw decoder-layer output `f`.
Semantically the draft was TRAINED so `f` imitates the target's post-norm
**gamma-included** feature `h_t` (that is the regression target of EAGLE-1
training, and the head that scores it is the target's `W_lm`, which expects
gamma-included features).

Contract used by B2:

    first-step input   (fused target):  a_t = n(x)·R1   gamma_already_included = false
    recurrent input    (rotated draft): h_d_R = h_d·R1  gamma_already_included = true
    draft head (rotated):               W_lm·R1         (no gamma — see basis contract)

The draft's own norms (`post_attention_layernorm`, gamma_l) are INTERNAL to the
decoder layer; `convert_draft_state(mode='r1')` fuses gamma_l into gate/up and
R1-conjugates all seven linears, so the conjugated layer maps `R1-basis in →
R1-basis out` exactly (unit-RMSNorm commutes with orthogonal R1).

## Draft final norm / draft_final_rms_gamma

There is NO separate draft final RMSNorm module in the EAGLE-1 v1 decoding path
— `D_FINAL_NORM` does not exist as a runtime op; the only final_rms_gamma in
the system is the TARGET's (`target_final_rms_gamma`,
`base_model.model.norm.weight`). Any "draft_final_rms_gamma" would have to come
from a hypothetical draft-side norm; none is executed, so the B2 fold uses
target_final_rms_gamma exactly once on the target→draft edge and never inside
the draft loop.

## First vs recurrent — why representations differ

- call #0 feature comes from the TARGET tail. Fused SpinQuant tail strips gamma
  (into the head), so the draft sees `a_t` (gamma-less).
- calls #1.. features come from the DRAFT decoder, which imitates
  gamma-INCLUDED `h_t` (rotated: `h_d_R`).

Hence one fixed projection cannot serve both: the first needs the extra inner
`D_gamma` (W_feature_first = R1ᵀ W_f D_γ R1), the recurrent must not
(W_feature_recurrent = R1ᵀ W_f R1). Prior measured proof: single permanently
folded projection (Variant B) = 2.17 acceptance vs split (B2/F_R_only) = 3.37.
