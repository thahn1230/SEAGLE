# INITIAL BITWIDTH-AL COMPONENT-CAUSALITY AUDIT (Phase 1)

Date 2026-07-15 · branch `exp/eagle1-bitwidth-al-component-causality` (from
concat-selective HEAD `8a4711c`; B2 + concat-selective preserved) · GPUs 6,7.
Answers to the 11 §19 questions:

## 1-2. Which 3×3 cells have comparable results / which are missing

Directly comparable (same architecture contract, n=20×64 greedy,
`cs_matrix_20260714_2111`):
- **T16_D16** = Q00 stock = 3.4158 (rerun anyway at final scale).
- **T4_D16** = Q10 (A-explicit runtime bridge, original draft) = 3.0079 —
  matches §2.2 exactly.
- **T4_D4** = Q11_new (fused W4A4 target a_t + concat-selective, first block
  W_h·D_γ·R1) = 1.2140 — matches §2.4.

MISSING (6): T16_D8, **T16_D4 under the §2.3 contract** (the prior
Q01_new used a ROTATED fp16 target + gamma_R1 first block; §2.3 demands a
STOCK FP16 target supplying h_t with first block **[W_e | W_h]** — a new
first-projection variant), T8_D16, T8_D8, T8_D4, T4_D8. All nine cells will be
RERUN under one harness (pilot 20×64, final 80×128).

## 3. Old results that cannot be reused

The pure-R1 W8A8 numbers (`fake_w8a8_draft` study: C4 2.121, C5 2.457,
C6 1.648) used the h_R-tail interface + single-path pure-R1 draft — different
architecture; shown only in a separately-labeled regression table, never in
the 3×3.

## 4. Correct architecture per cell

| cell | target | interface | draft |
|---|---|---|---|
| T16_D16 | stock fp16 | h_t native | stock EAGLE |
| T8/T4_D16 | fused rotated, fake W8A8/W4A4, KV fp16 | a_t→R1ᵀ→γ→h_t (runtime, A-explicit) | original unrotated, fp16 |
| T16_D8/D4 | **stock fp16 (untouched, no rotation)** | h_t direct | concat-selective, first **[W_e\|W_h]**, recurrent [W_e\|W_h·R1], post-R1, head W_lm·R1 |
| T8/T4 × D8/D4 | fused rotated quantized | a_t direct | concat-selective, first [W_e\|W_h·D_γ·R1], recurrent [W_e\|W_h·R1], post-R1 |

## 5-6. Embedding / LM-head storage sharing

- Draft `embed_tokens`: **independent storage**, bit-exact copy of the
  untouched checkpoint (rel-L2 0.0, verified). Target embedding fusion never
  touches it.
- Draft LM head: stock EAGLE passes `base_model.lm_head` INTO topK_genrate
  (**alias**); every adapter in this repo substitutes an ISOLATED
  `adapter.head` (own nn.Linear, own storage) — so draft-head quantization
  never modifies the target verifier head. A runtime data_ptr audit script
  re-proves this (`docs/TARGET_DRAFT_PARAMETER_OWNERSHIP_AUDIT.md`).

## 7. Did the previous "AR head W4A4" include the vocabulary LM Head?

**NO.** `quant_ar` covered only the 7 decoder linears (q/k/v/o/gate/up/down);
the draft scoring head stayed fp16-isolated, and the embedding stayed fp16.
Draft-LM-head-only and embedding-only are NEW ablations in this study.

## 8. Previously missing component ablations

Draft LM head (all precisions), draft embedding ACTIVATION quant (table-weight
W4A16 existed once), target embedding (all), target LM head (all), target body
W16A8/W16A4, weight-only/act-only splits for every component, branchwise
concat quantization, mixed first/recurrent and projection/LM-head precisions.

## 9. Does activation quantization reduce only over the last dimension?

**YES — verified in code.** SpinQuant `ActQuantizer.find_params` (groupsize≤0,
our configuration): `reshaped_x = x.reshape((-1, x.shape[-1]))` then
`min(1)/max(1)` — reduction ONLY over the final feature dimension; one scale
per row (token/tree-node). `FakeW4A4Linear.forward` calls `find_params(x2)`
per forward on `[rows, in_features]` — dynamic per-token, no cross-batch/
sequence/tree/depth reduction. KV quantizers (when used) are per-head
group-128 — not part of the primary matrix (KV fp16).

## 10. Why did Stock and transformed FP16 acceptance differ?

Measured facts: stock 3.4158 vs concat-selective FP control 3.4116 (n=20×64);
19/20 prompts token-identical; ONE prompt (id 95) flips one late low-margin
token; Gate-A (8×48) was 8/8 token-exact with identical AL 3.7538. Exact
arithmetic equivalence is proven (fp64 4e-14). Hypothesis to be tested by
Phase-3 forensics: fp16 roundoff differences (transformed-weight casting +
explicit fp32 post-R1 vs stock's fused path) perturb draft logits at a
low-margin top-k boundary → different tree candidate → one different
verification-round split; final OUTPUT tokens still match the target's greedy
string until a late margin<roundoff position. Forensics will produce the
first-divergence tensor + margin certificates (§12).

## 11. Code changes required

- `study.QUANT_CFGS`: add `w8a8`, `w8a16`, `w16a8`, `w16a4` (mechanically safe:
  weight quant is gated by `spec.w_bits < 16`; `ActQuantizer` no-ops at
  bits==16; a_bits routed generically via `default_ptq_args`).
- `concat_selective_projection`: extend `QUANT_MODES` with `fake_w8a16`,
  `fake_w16a8`, `fake_w16a4` (FakeW4A4Linear already supports
  quant_weight=False / quant_act=False independently); add
  `first_hidden_mode ∈ {"gamma_R1" (fused target), "identity" (stock T16
  target, first block [W_e|W_h])}`; add `quant_embed_act` (embedding-OUTPUT
  activation quant wrapper), `quant_head` (isolated draft scoring head quant),
  branchwise-concat activation quant mode.
- New: isolated target embedding/LM-head fake-quant patcher (target-side,
  body fp16); parameter-ownership audit script; fixed-tree grader; forensics
  tracer; component-matrix runner (table-driven).
