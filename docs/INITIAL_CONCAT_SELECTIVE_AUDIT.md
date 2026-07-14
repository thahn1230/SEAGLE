# INITIAL CONCAT-SELECTIVE AUDIT (Phase 1)

Date 2026-07-14 · branch `exp/eagle1-concat-selective-rotation` (from B2 HEAD
`9c59d94`, preserved) · GPUs 6,7 only. Answers to the 10 §25 questions:

1. **Concat order**: `cnets.py:593` `self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))`
   → physical order **[embedding | feature]** (re-asserted at runtime by
   `tests/test_b2_concat_order.py`, reused).
2. **Physical slices**: `W_e = fc.weight[:, :4096]`, `W_h = fc.weight[:, 4096:]`,
   `fc.bias` present ([4096]).
3. **Draft embedding basis**: **ORIGINAL**. Measured: `ea_layer.embed_tokens.weight`
   vs the untouched `meta-llama/Llama-2-7b-chat-hf` checkpoint
   `model.embed_tokens.weight` (read directly from safetensors) → rel-L2
   **0.000e+00** (bit-exact copy), own storage, frozen.
4. **Copied when**: at draft init, by `cnets.Model(load_emb=True, path=target
   checkpoint)` — read straight from the HF checkpoint file, never through any
   fused/rotated target module. Target R1 fusion happens on the TARGET module's
   separate storage; the draft copy is untouched. (Adapters that rotated the
   embedding did so only inside `install()` via converted state dicts and
   restored the original on `uninstall()`.)
5. **Target hidden sent to EAGLE**: stock target → `h = RMSNorm_γ(x)` (post-norm,
   gamma INCLUDED); fused rotated target → `a = n(x)·R1` (gamma EXCLUDED,
   rotated). `ea_model.py:124-145`, `modeling_llama_kv.py:1074`.
6. **Recurrent hidden fed back**: the draft decoder-layer output (`out_hidden`),
   re-consumed raw (no draft final norm). With an R1-conjugated decoder this is
   `r_d = h_d·R1`. Contract check for the new architecture: the projection
   output is explicitly rotated (post-PL R1) and the conjugated decoder maps
   R1-basis→R1-basis, so `r_d = h_d·R1` holds ✓ (docs/CONCAT_SELECTIVE_AR_BASIS_CONTRACT.md).
7. **Previous B2 rotates the embedding**: YES — `convert_draft_state` sets
   `embed_tokens.weight ← E@R1` (`in_basis_rows`), and the fc e-block is
   conjugated `R1ᵀW_e R1`.
8. **Previous B2 transforms both blocks together**: YES — fc =
   `R1ᵀ[W_e R1 | W_h (D_γ) R1]`, bias `b·R1` (output fold applied to the WHOLE
   projection). The NEW architecture instead keeps `W_e` untouched, folds only
   the hidden block (`W_h D_γ R1` first / `W_h R1` recurrent), keeps bias `b`,
   and applies R1 EXPLICITLY to the complete projection output.
9. **Rotation type**: **random Hadamard R1** (`outputs/rotations/random_hadamard/R.bin`,
   seed 0) + per-head random-QR R2 + structured-Hadamard R4 where used; RTN
   quantization. NOT learned SpinQuant rotations → all groups are labeled
   `random_rotation_control` (docs/CONCAT_SELECTIVE_SPINQUANT_AUDIT.md; learned-R
   PPL exists for the target: 10.44 vs random 10.63).
10. **Real W4A4 attached e2e?** Previously: target-side real INT4 e2e
    (`runs/rotstudy_realint4_*`) and projection-level kernel validation (Gate B)
    only — the draft-side real-W4A4 EAGLE generation was NOT run. This study
    adds it (Phase 8).

Algebraic relationship to previous B2 (exact arithmetic): the new
concat-selective function is IDENTICAL — `(eW_eᵀ + aR1ᵀD_γW_hᵀ + b)R1` — the
architectures differ in where the rotation lives (rotated embedding table +
fully conjugated weights + rotated bias vs original embedding + hidden-only
folds + explicit output R1), hence in fp16 roundoff and, more importantly, in
QUANTIZATION exposure: the new fc quantizes an ORIGINAL-basis `W_e` block and
sees original-embedding activations on the e-slice.
