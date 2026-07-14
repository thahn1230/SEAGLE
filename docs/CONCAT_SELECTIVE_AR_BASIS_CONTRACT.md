# CONCAT-SELECTIVE AR BASIS CONTRACT (code-verified, EAGLE v1 @4a9cf3a)

Traced from `cnets.py:762-845` (topK_genrate) + `cnets.py:540-638` (Model.forward)
+ `cnets.py:378-432` (decoder layer; NO input_layernorm on layer 0).

| tensor | basis (this architecture) | gamma | producer → consumer |
|---|---|---|---|
| target tail output `a = n·R1` | rotated, unit-RMS | EXCLUDED | fused target norm → topK_genrate hidden arg |
| draft embedding `e` | ORIGINAL | n/a | ea_layer.embed_tokens → fc concat e-slice |
| fc concat input (first) | [original e \| rotated a] | hidden-slice only, inside W_first | cnets:593 |
| projection pre-R output | original y (semantics of stock fc) | applied once (first) | projection_*_preR |
| projection output after post-R1 | y·R1 (rotated) | — | post_projection_R1 → decoder layer |
| AR-head input | rotated (y·R1) | — | decoder layer 0 (R1-conjugated) |
| AR residual stream | rotated (R1-conjugated layer maps R1→R1 exactly; unit-RMSNorm commutes with orthogonal R1; gamma_l fused into gate/up) | internal | attn/mlp |
| AR-head output `out_hidden` | **r_d = h_d·R1** | semantically included (imitates gamma-included h) | decoder → recycled + head |
| recurrent fc input | [original e \| r_d] | NOT re-applied | cnets tree loop |
| draft LM-head input | r_d | included-by-imitation | head = `W_lm·R1` |
| target verifier input | target's own tokens/logits (untouched) | — | tree_decoding |

Confirmation of the expected primary contract:
- projection output after R **is** the rotated AR-head input ✓
- AR recurrent output **is** `r_d = h_d·R1` ✓ (conjugated decoder in = y·R1,
  out = f·R1; verified numerically by the tiny-chain test at <1e-6 every depth)
- recurrent projection input = [original e | rotated r_d] ✓
- no draft final RMSNorm exists in the loop; the ONLY final_rms_gamma in the
  system is the TARGET's, applied exactly once inside `projection_first_preR`.

First vs recurrent difference recap: the first hidden (`a`) lacks gamma (the
fused target moved it into the LM head), the recurrent hidden (`r_d`) imitates
a gamma-included feature — hence `W_h·D_γ·R1` vs `W_h·R1`.
