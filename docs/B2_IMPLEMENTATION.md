# B2 IMPLEMENTATION

## Module map (spec §21 → this repo)

| spec file | implemented as |
|---|---|
| b2_projection.py | `src/eagle_spinquant/b2_projection.py` (weights + split module + adapter) |
| b2_dispatch.py | dispatch lives in `B2SplitDraftAdapter.install()` (patched `ea.forward` + wrapped `topK_genrate`) — same file, one authority for state |
| basis_contract.py | formulas as executable code: `m_gamma`, `build_b2_weights_arch_{a,b}` + `docs/B2_SPLIT_BASIS_CONTRACT.md` |
| projection_transform.py | `rotation_aware.in_fold/out_fold/convert_draft_state` (pre-existing, validated) + `study.fold_matrix` |
| trace_hooks.py | `B2SplitDraftAdapter.trace_rows` (per-call dispatch trace) + `FakeW4A4Linear.trace` (quant coverage) |
| inspect_eagle_projection_flow.py | `docs/EAGLE1_PROJECTION_LAYOUT.md` + `tests/test_b2_concat_order.py` (source+runtime proof) |
| validate_b2_fp_equivalence.py | `scripts/validate_b2_fp_equivalence.py` (STOP GATE A) |
| validate_b2_projection_dispatch.py | `tests/test_b2_tree_dispatch.py` + `tests/test_b2_state_reset.py` (assert on the real 7B trace) |
| validate_b2_real_w4a4.py | `scripts/validate_b2_real_w4a4.py` (STOP GATE B) |
| run_b2_acceptance_matrix.py | `scripts/run_b2_acceptance_matrix.py` |
| analyze/plot/report | `scripts/analyze_b2_results.py`, `scripts/plot_b2_results.py`, `scripts/build_b2_final_report.py` |

## Key design decisions

1. **Named modules, not pointer swap.** `B2SplitProjection` holds
   `projection_first` and `projection_recurrent` as separate `nn.Linear` /
   `FakeW4A4Linear` / real-int4 children so each carries its OWN quantizer
   state, scales and packed buffers. (The prior `TwoPathAdapter` `.data`-swap
   is kept as an independent equivalence witness, not used for quant runs.)
2. **Dispatch = per-cycle draft-forward index**, reset in the wrapped
   `topK_genrate`; proven exact for EAGLE-1 v1 (call #0 rows are always
   target-originated — `cnets.py:772-820`; no mixed batches, so no per-row
   selection is needed). Hard asserts: `projection_first` exactly once per
   cycle at index 0; unarmed forward refuses to run; per-prompt reset via
   `set_context`.
3. **Weight construction reuses the validated conversion**:
   `convert_draft_state(mode='r1')` supplies `projection_recurrent`
   (= `R1ᵀ[W_e R1 | W_f R1]`) and `fc_ext` supplies `projection_first`
   (= `R1ᵀ[W_e R1 | W_f D_γ R1]`); bias `b·R1` shared. Arch-A folded uses
   `fold_matrix` (`W_f·D_γ·R1`) on the h-block only.
4. **Quantization order** (spec §15): FP weight → split/rotate/fold →
   reconstruct full FP projection → quantize → pack. Never transform packed
   weights. Real kernels attach per projection with independent scale buffers
   (`scripts/validate_b2_real_w4a4.py`).
5. **Ownership**: draft head is an isolated copy substituted inside the wrapped
   `topK_genrate` (stock EAGLE aliases the target lm_head); draft embedding is
   the draft's own storage. Draft-quant configs therefore never quantize target
   verification weights.

## Measured implementation facts

- Commutator: ‖D_γR1 − R1D_γ‖_F / ‖D_γR1‖_F = 0.393 (real R1) — gamma does not
  commute; elementwise-γ in the rotated basis is wrong (N2 collapses to 1.04).
- fc weight structure: 100% of the 4096 output rows attain their absmax in the
  shared EMBEDDING block (mean |w|max 0.308 vs h-block 0.129 first / 0.073
  recurrent) → per-channel quant scales are e-block-dominated; the D_γ fold
  amplifies the first h-block ≈1.76×, degrading its effective int4 resolution.
  This predicts (and the matrix confirms) that `projection_first` is the most
  quantization-fragile draft component.
- STOP GATE A: rotated target greedy == stock (8/8); A-explicit = A-folded =
  B2-split = explicit-Mγ = 3.7538 (token-exact, n=8×48); FP04/N1/N2/N4 collapse
  to ≈1.13/1.04; FP03 −0.489 (8/8 prompts); N5 −0.241 (6/8).
- STOP GATE B: real QuaRot W4A4 executes for BOTH projections
  (rel-vs-emulation 1.8e-2; [4096,8192] packed 16.8MB each) and real tinygemm
  W4A16 (rel-vs-fp16 0.101); M=1..110 shapes verified.
