# SpinQuant-aware pure-R1 EAGLE draft — architecture audit (Phase 1)

Mode `w4a4_spinquant_draft_pure_r1`. Draft = yuhuili/EAGLE-llama2-chat-7B
(cnets.py `Model`). All facts code-verified; unit tests in
`scripts/validate_spinquant_draft_pure_r1.py --task units`.

## Draft module inventory (checkpoint keys + cnets.py)
| # | question | answer |
|---|---|---|
| 1 | fc projection shape | `fc.weight` = **[4096, 8192]** (Linear 8192→4096) |
| 2 | concat order | **[embedding, hidden]** (cnets.py:592 `cat((inputs_embeds, hidden_states))`) |
| 3 | fc is Linear(2D→D) | YES (D=4096) |
| 4 | fc.weight shape | [4096, 8192]; also `fc.bias` [4096] |
| 5 | W_e / W_h | `W_e = fc.weight[:, :4096]`, `W_h = fc.weight[:, 4096:]` |
| 6 | draft decoder layers | ONE layer (`layers.0`): self_attn + mlp + post_attention_layernorm |
| 7 | first layer input_layernorm | **NONE** (cnets.py `if self.index != 0`; layer 0 has no input_layernorm) |
| 8 | post_attention_layernorm | YES (`layers.0.post_attention_layernorm.weight`) |
| 9 | q/k/v/o_proj | `layers.0.self_attn.{q,k,v,o}_proj.weight` [4096,4096]; 32 heads, head_dim 128, no GQA |
| 10 | gate/up/down_proj | `layers.0.mlp.{gate,up}_proj` [11008,4096], `down_proj` [4096,11008] |
| 11 | draft feature returned | post-layer residual output (no final norm in the draft) |
| 12 | head used for scoring | passed to `topK_genrate` at call time (this mode substitutes `W_lm @ R1`) |
| 13 | draft owns lm_head? | NO — no `lm_head` key/module; target head passed externally |

All "expected but verify" assumptions hold: `fc.weight.shape == [4096, 8192]`,
concat `[embedding, hidden]`, `W_e/W_h` as sliced.

## Projection-layer R1 conjugation (Section 5 formula, verified fp64)
`W_PL_R = R1.T @ W_PL @ block_diag(R1, R1)`, i.e. `W_e_R = R1.T @ W_e @ R1`,
`W_h_R = R1.T @ W_h @ R1`, `b_R = b @ R1`. Verified:
`concat([e@R1, h@R1]) @ W_PL_R.T + b_R == (concat([e,h]) @ W_PL.T + b) @ R1`
to **rel-L2 1.2e-15 (fp64)**, 6.4e-4 (fp16). The reused
`rotation_aware.convert_draft_state(mode='r1')` produces exactly this
`W_PL_R` (fp32-storage match 2.5e-8).

## Draft R1/R2/R3/R4 (Section 6)
- **R1** (residual rotation): conjugates every draft linear + embed rows + head.
  Correctness-critical; makes the recurrent draft stream R1-basis so external
  `h_R` and recurrent `f_R` share one basis (single fc path, no swap). EXACT.
- **R2** (per-head V/O), **R3** (Q/K Hadamard), **R4** (MLP-down Hadamard) are
  SpinQuant activation-quantization rotations. They are ORTHOGONAL and
  **fp16-identity** (cancel in exact arithmetic): audited fp16-equivalence
  R2 3.1e-4, R3 5.8e-16, R4 6.6e-4 (`draft_rotation_audit.csv`). They do NOT
  change fp16 acceptance; they only reshape draft activations when the DRAFT is
  quantized. The fp16 generation path applies R1 only (sufficient and exact);
  R2/R3/R4 are audited and documented as quantization-time rotations.

## RMSNorm gamma folding (Section 6)
The draft has only `post_attention_layernorm` (no input_layernorm on layer 0).
`convert_draft_state` folds this `gamma_l` into `gate_proj`/`up_proj` (the
following linears) and sets the norm weight to ones (scale-free), so RMSNorm0
commutes with R1 exactly. The target's `gamma_f` is NOT folded into the draft —
it belongs to the target tail only; the draft head is `W_lm @ R1` (no gamma_f).

## Draft LM head (Section 7)
Pure-R1 feature `f_R = f @ R1` ⇒ head `W_lm_R = W_lm @ R1`:
`(f@R1) @ (W_lm@R1).T = f @ W_lm.T` (correct logits). Verified top-1 1.000.
The gamma_f-fused head `W_lm @ diag(gamma_f) @ R1` on `f_R` gives `(f*gamma_f)
@ W_lm.T` — top-1 only 0.875 (per-channel rescale, wrong basis; A6). The
original head `W_lm` on `f_R` gives top-1 0.0. So the pure-R1 path MUST use
`W_lm @ R1` and must NOT use the gamma_f-fused head.
