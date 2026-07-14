# CONCAT-SELECTIVE ROTATION ARCHITECTURE

Row-vector convention: `Linear(x; W, b) = x Wᵀ + b`, weights `[out, in]`.
`R = R1` (random Hadamard, dense-materialized), `D_γ = diag(target_final_rms_gamma)`.
Physical concat order (cnets.py:593): **[embedding | feature]** →
`W_e = fc.weight[:, :4096]`, `W_h = fc.weight[:, 4096:]`, bias `b`.

## Dataflow (per the supplied diagram)

FIRST draft forward of each verification round:

    e (ORIGINAL draft embedding)         a = n·R1 (fused target tail)
        \                                   /
         concat(e, a)
           → hidden slice ONLY: R1ᵀ then γ     [folded: W_h ← W_h·D_γ·R1]
           → ORIGINAL semantic projection       [W_e untouched, bias b untouched]
           → complete D-dim output → R1         [explicit post_projection_R1]

RECURRENT forwards:

    e (ORIGINAL)                         r_d = h_d·R1 (rotated AR output)
        \                                   /
         concat(e, r_d)
           → hidden slice ONLY: R1ᵀ            [folded: W_h ← W_h·R1]
           → ORIGINAL semantic projection
           → complete output → R1

Weights actually built (`concat_selective_projection.build_concat_selective_weights`):

    projection_first_preR     = [W_e | W_h·D_γ·R1],  bias b
    projection_recurrent_preR = [W_e | W_h·R1],      bias b
    post_projection_R1        = explicit dense fp32 GEMM y → y·R1

Invariants (asserted in code + tests):
- R1ᵀ and γ touch ONLY the hidden slice; the embedding slice and `W_e` are
  bit-untouched; bias is the ORIGINAL `b` (the output R1 is explicit, so `b·R1`
  must NOT be pre-folded).
- γ enters exactly once (first path); the recurrent path absorbs only R1ᵀ.
- R1 is applied AFTER the 2D→D projection, to the complete output.

## Exact-arithmetic equivalence

    first:  (e·W_eᵀ + a·(W_h D_γ R1)ᵀ + b)·R1
          = (e·W_eᵀ + n·D_γ·W_hᵀ + b)·R1 = y_stock·R1
    recur:  r_d·(W_h R1)ᵀ = h_d·R1·R1ᵀ·W_hᵀ = h_d·W_hᵀ  ⇒ y_stock·R1

So the function equals stock EAGLE conjugated by R1 on its output feature —
identical to the previous B2 (rotated-embedding) architecture in exact math.
The DIFFERENCE is representational: where the rotation lives.

| | previous B2 | concat-selective (this study) |
|---|---|---|
| draft embedding | `E·R1` (rotated table) | `E` (original checkpoint copy) |
| e-block of fc | `R1ᵀ·W_e·R1` | `W_e` (untouched) |
| h-block first/rec | `R1ᵀ·W_f·D_γ·R1` / `R1ᵀ·W_f·R1` | `W_h·D_γ·R1` / `W_h·R1` |
| bias | `b·R1` | `b` |
| output rotation | folded into fc (R1ᵀ output fold) | EXPLICIT post-PL R1 |
| e-slice activation seen by quantizers | rotated embedding | ORIGINAL embedding |

AR decoder: R1-conjugated (same validated `convert_draft_state` decoder part —
maps R1-basis→R1-basis exactly; layer 0 has no input_layernorm). Draft head:
`W_lm·R1`. Recurrent contract `r_d = h_d·R1` holds because the explicit output
R1 delivers rotated features into the conjugated decoder
(docs/CONCAT_SELECTIVE_AR_BASIS_CONTRACT.md).

## post_projection_R1 realization

Dense fp32 GEMM ([*,4096]×[4096,4096]) — the stored random-Hadamard R1 is a
dense matrix (`R.bin`); a fast-Hadamard version would need its factored form.
Recorded per run in `meta_quant["post_projection_R1"]`. A fused variant
(folding the output R1 into the AR-head input weights) is exactly the previous
B2 fc and is measured as `F1`/`Q*_prevB2` — kept separate from the primary
explicit-placement architecture.
