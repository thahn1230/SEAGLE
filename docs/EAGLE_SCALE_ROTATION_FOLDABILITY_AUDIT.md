# EAGLE Scale/Rotation Foldability Audit (F0-F3)

Executable proofs: `scripts/audit_eagle_transform_foldability.py`
(results `tables/foldability_audit.json`, all checks PASS on real
calibration tensors with the deployment quantizer) and the fused-kernel
code-parity suite (`tables/kernel_verify.json`, ≥0.99999 integer-code
agreement vs the official per-token asym A4).

Categories: **F0** fully offline foldable · **F1** offline with
duplicated parameters · **F2** runtime math required but fusable ·
**F3** necessarily separate online op.

| transform | class | producer→consumer | folding formula | runtime / storage |
|---|---|---|---|---|
| W-side of any T (S, Q) | **F0** | fc.weight → projection GEMM | `W'_pt = W_pt T^{-T}` = `(W_pt S^{-1}) Q` (SR); computed offline per path view | none / 64 MiB per fp16 view |
| EP3-P first-path scale m_first | **F1** | embedding table → concat | `E_first = m_first · E` (S1/S2); A4 codes **bitwise identical** to runtime multiply (checked: S0=S1=S2) | none / +250 MiB fp16 table view |
| EP3-P recurrent rescale m_rec/m_first | **F2** (or F1 via S2 dual view) | table(m_first) → concat | runtime e-slice multiply; fp16 rounding gives 5.9e-5 code mismatch vs direct m_rec — **S2 dual table view is the bitwise-exact folding** | 1 mul/channel (fused in K1) / +250 MiB for second view |
| branchwise Q_e | **F1** | embedding table | fold `Q_e` into table rows offline (checked exact); recurrent path shares the folded table | none / table view |
| branchwise Q_h (first path) | F1 | interface fold | absorb into the validated gamma_R1 interface fold | none |
| branchwise Q_h (recurrent) | **F3 unless full basis change** | draft hidden → concat | requires rotating the ENTIRE draft hidden basis (attn, MLP, residual, head, next projection) — partial change forbidden; not undertaken | — |
| cross-branch Q (pairwise/blockwise/full) | **F2** | embedding ⊕ decoder output → concat/A4 | no common producer weight exists for e and h ⇒ offline folding structurally impossible (negative control: rotated-W-only breaks FP, rel err ≫1e-2); mathematically required at runtime, **fused into the concat+scale+A4 kernel** (K2/K3): zero additional kernel launch, zero materialized intermediate | in-register; measured 0.046 ms @≤256 tok (BELOW the unfused baseline path) |
| attention R2 (V/O pair) | **F0** | W_v/W_o | SpinQuant head-wise pair absorption (already deployed) | none |
| MLP R4 (down_proj) | **F3→F2** | SiLU output → down GEMM | online Hadamard required (nonlinearity upstream); currently separate CUDA op; fusable as down-GEMM prologue in principle | 1 op / none |

Key negative control (§15): folding `T^{-T}` into W does **not** remove
the runtime `xT` when activations quantize after the transform —
weight-only fold changes the FP function (checked) — so "fully folded
R-EP3-P" is impossible for cross-branch Q; the correct claim is
**fused-but-not-folded** (Conclusion I), with measured fused cost
0.046-0.059 ms per call vs 1.3-2.2 ms for the explicit Python path
(25-40×), i.e., zero-additional-kernel and, at projection scale,
below measurement noise end-to-end.

Terminology used in the study: *zero additional kernel* = fused into
existing op (K1-K3); *fully folded* = F0/F1 only (EP3-P scale via
table views, W-side transforms, R2); *fused not folded* = F2
(cross-branch Q, recurrent rescale); *necessarily online* = F3 (R4;
recurrent-basis change rejected).
