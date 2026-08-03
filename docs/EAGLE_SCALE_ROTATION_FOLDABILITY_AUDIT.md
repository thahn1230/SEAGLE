# EAGLE Scale/Rotation Foldability Audit

Executable proofs: `scripts/audit_eagle_transform_foldability.py` →
`tables/foldability_audit.json` (all claims verified with the
deployment W4/A4 quantizers; scale options checked at **quantized-code
level**, never only dequantized output).

Categories: **F0** fully offline foldable (no runtime arithmetic) ·
**F1** offline foldable with duplicated parameter views · **F2**
runtime arithmetic mathematically required but fusable into an
existing kernel · **F3** necessarily online operation.

## 1. Orientation contract (all derivations below)

Storage `W_pt [out,in]`, forward `y = x W_pt^T + b`. For `x' = x T`
FP preservation forces `W'_pt = W_pt T^{-T}`; with `T = S Q`
(S = diag(m·I_D, I_D), Q orthogonal): `W'_pt = (W_pt S^{-1}) Q`.
Verified on tensors to 4.4e-15 relative (`fold_fp_equiv_rel`).

**Folding W does not remove x·T** when the activation quantizer sits
after the transform: the negative control (transformed weight +
un-transformed activation) breaks FP (`negative_control_breaks_fp:
true`) and reverts the quantization behavior to the naive baseline
(`quant_nmse_weightonly_fold` ≈ baseline, not folded-pipeline).

## 2. EP3-P scaling (S-options)

| option | realization | verdict |
|---|---|---|
| S0 | explicit runtime multiply `m·e` | reference |
| S1 | pre-scaled fp16 table `E·m_first`; recurrent applies `m_rec/m_first` | **F1**; codes IDENTICAL to S0 (`S0S1_code_identical: true` — same floats, same per-token (max−min)/15 scale) |
| S2 | dual table views `E·m_first`, `E·m_rec` | **F1**; codes identical to S0 (`S0S2_code_identical`); +64 MiB fp16 view |
| S1-rescale vs S2-direct | fp16 `(E·m_first)·(m_rec/m_first)` vs `E·m_rec` | code mismatch rate **5.9e-5** (last-bit fp16 rounding) — deployed studies used S1; S2 removes even this |
| S3/S4 | fused scale inside concat+A4 without materializing `E·m` | **F2**; same computation reordered — code parity proven at kernel level (K1 test) |

Deployed EP3-P = S1 (embedding-table view + one recurrent rescale
multiply). The recurrent rescale is the only runtime arithmetic and is
fusable (F2) or removable via S2 (F1, +64 MiB).

## 3. Projection rotations

| transform | class | proof/argument |
|---|---|---|
| branchwise `Q_e` (block-diag, e half) | **F1** | folds into table rows `E·m·Q_e` and `W_e` columns offline; exact (`branchwise_Qe_table_fold_exact: true`). One table view per path if pathwise. |
| branchwise `Q_h` on the FIRST path | **F2** | h is the live TARGET hidden state; folding upstream would modify the shared target (forbidden). Rotate at concat time; fusable. |
| branchwise `Q_h` on the RECURRENT path | **F2** unless a FULL draft-hidden basis change | folding = conjugating attention, MLP, residual stream, LM head AND the next-cycle projection consistently; a partial basis change breaks FP. Not attempted (out of scope; would alter validated interfaces). |
| cross-branch Q (interleaved/full FWHT, the R-EP3-P family) | **F2** | mixes e and h AFTER they are produced by different modules — offline folding is structurally impossible without changing model topology. Activation-side arithmetic is mathematically required (§1 negative control); the weight side folds offline. Butterfly cost 2D·log2(b) adds/token; fusable into concat+A4 (K2/K3 kernels). |

## 4. Other layers (existing validated folds)

| transform | class | note |
|---|---|---|
| target R_T, interface R1, head `W_lm·R1` | **F0** | validated offline folds (prior studies) |
| attention R2 (`W_v→W_v R2`, `W_o→R2ᵀW_o`) | **F0** | SpinQuant pair-fold, no runtime op |
| MLP R4 on `down_proj` input | **F3** | sits after the SiLU·up nonlinearity — cannot cross it; SpinQuant runs it online (`matmul_hadU_cuda`). The component-audit bug we fixed (restored-fp16 `down_proj` losing its online Hadamard) is the canonical example of why F3 ops must follow the weight fold, not the quantizer flag. |

## 5. Terminology used in the runtime report

- **Fully folded** — no runtime transform arithmetic (F0/F1).
- **Fused but not folded** — arithmetic inside an existing kernel, no
  extra launch (F2 with the K-kernels).
- **Zero additional kernel** — transform fused into concat+A4.
- **Zero measured overhead** — e2e latency ratio CI includes 1.0 with
  upper bound ≤ 1.005 (benchmark section; never claimed from Python
  timing alone).
