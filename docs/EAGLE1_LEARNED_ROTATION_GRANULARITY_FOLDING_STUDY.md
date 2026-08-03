# Post-EP3-P Bottleneck, Learned Rotation, Granularity, and Folding (LRGF)

**Run**: `runs/eagle1_learned_rotation_granularity_folding_20260803_133218`
**Branch**: `exp/eagle1-learned-rotation-granularity-folding`
(source `1cc5ac4`, verified). Tests 15/15. GPU 7 excluded per operator
throughout the second half of the run.

## 0. Headline answers

- **Q-A**: Projection was the right first intervention (its naive
  quantization alone costs −0.99 tau; EP3-P recovers it to −0.067),
  and after EP3-P the bottleneck **moves to the AR decoder**:
  restoring it to FP16 gains **+0.128 [+0.067, +0.190] SIG**; MLP
  +0.075 SIG; **restoring the first projection gains +0.00** — EP3-P
  exhausted the projection problem. (Conclusions A + B.)
- **Q-B**: **Acceptance-aware learned rotations beat every fixed
  arm** — the first frozen-weight method to significantly beat EP3-P
  since this program started: ACC-LR validation dAL **+0.183 [+0.072,
  +0.304]**, dRCAL **+0.168 [+0.070, +0.283]** vs EP3-P (vs FIXED-R:
  +0.172/+0.192, all CIs exclude 0); full MT-Bench confirmation dAL
  **+0.092 [+0.014, +0.163] SIG**, dRCAL +0.067 [−0.012, +0.141]
  borderline. EAGLE-objective LR merely ties FIXED-R; NMSE-trained LR
  fails to transfer *again* (Conclusion D; proxy-selection failure
  reproduced on schedule).
- **Q-C**: Cross-branch mixing at **any block size ≥ 2** captures the
  local-NMSE gain (b2…b8192 flat within 0.0002); learned 32-ch Cayley
  blocks suffice (Conclusion E).
- **Q-D**: EP3-P scaling is **fully foldable** (S1/S2 embedding-table
  views are quantized-code-identical to the runtime multiply);
  cross-branch rotation is **structurally unfoldable but fully
  fusable**: one Triton kernel (concat+scale+rotation+A4) with
  ≥0.99999 integer-code parity vs the official quantizer, 0.046 ms
  — below the unfused baseline path (Conclusions H for scale, I for
  rotation).

## 1. Q-A — component audit (33 valid arms, MT-Bench 80, T1)

Quantize-one (from FP16 draft, tau; FPDRAFT ceiling 3.2678):
embedding 3.306*, head 3.259, first-proj naive 2.371, rec-proj naive
2.996, both naive 2.276, **both EP3-P 3.201**, Q 3.127, K 3.132,
V 3.127, O 3.137, QKV 3.126, QKVO 3.133, gate 3.135, up 3.130,
down 3.209, gate+up 3.134, MLP 3.165, full AR 3.106, EP3-P+AR (=FULL
deploy) 3.0728. (*embed arm exceeds ceiling within noise.)

Restore-one oracle (from FULL 3.0728, paired bootstrap):

| restore | gain | CI | significant |
|---|---|---|---|
| AR decoder (B12) | **+0.128** | [+0.067, +0.190] | **yes** |
| MLP (B11) | +0.075 | [+0.015, +0.132] | yes |
| both projections (B2) | +0.076 | — | yes |
| attention QKVO (B7) | +0.024 | n.s. | no |
| recurrent proj (B1) | +0.046 | n.s. | no |
| **first proj (B0)** | **−0.001** | n.s. | no |

Cross-checks: A5≡B12 and FULL≡A18 configs measured twice through
different flag paths — identical tau to 4 decimals; FULL reproduces
the GQ-study EP3-P (3.0728) exactly. A masking bug (restored-fp16
`down_proj` silently dropping its folded-R4 online Hadamard) was
caught by the 2.13-tau collapse signature, fixed, and re-measured;
the fix is regression-tested.

## 2. Error flow (§5)

`tables/error_flow.json`: per-location NMSE traces show EP3-P and
shared-Q reduce projection-output error (0.0415 → 0.0249) but the AR
block re-amplifies: by `out_hidden` the configs converge (quant error
dominated by QKV/MLP), and recurrent depths compound it —
consistent with the oracle result that the AR decoder is now the
binding constraint. Draft-logit top-1 agreement tracks tau ordering.

## 3. Q-B — learned rotations (frozen weights, STE deployment quantizers)

Setup: `RotCore` = validated exact-quantized C7 pipeline; only
rotation parameters train (asserted); teacher caches t_q (W4A4
target) and t_0 (FP16) on 256 ShareGPT rows (t0–tq positional
agreement 0.7965 — 20% of positions are reference-inconsistent, the
RC mask rate). 400 steps, Adam 2e-3, orth ≤ 9e-7 all arms.

| arm (seed-best) | heldout tau (c4:20) | val RCAL | mtbench tau |
|---|---|---|---|
| EP3-P ref | 3.0187 | 1.641* | 3.0728 |
| FIXED-R ref | 3.0413 | — | (rerun) |
| NMSE-LR | 3.0844 | — | — |
| EAGLE-LR | 3.0412 (seeds 3.0412×3) | 1.722 (hh variant) | 3.1046 |
| **ACC-LR** | 3.1745 | **1.785** | **3.1495** |
| **RC-LR** | 3.2275 (s1; s0 3.061, s2 3.116) | 1.761 | 3.1517 |
| HYBRID-LR | 3.0996 | — | — |
| EAGLE-LR pathwise | 3.0199 | — | — |

(*mtbench capture value from the REP3P run.)

- **Selection (pre-registered §22)**: max validation RCAL → **ACC-LR
  s0**. Min-NMSE counterfactual would pick NMSE-LR (3.0844 heldout)
  — the proxy fails again, exactly as in the R-EP3-P study.
- **Surrogate-loss paradox, reported honestly**: ACC/RC training
  curves *rise* (survival surrogate worsens under single-anchor
  sampling noise) yet these arms deploy best. The surrogate value is
  not a fitness readout (spec 8.3 warned "do not report as actual
  AL"); its *gradient direction* still moves the rotation toward
  acceptance-relevant geometry. High-variance single-anchor sampling
  likely acts as implicit regularization; a proper analysis (more
  anchors, variance reduction) is future work.
- Shared Q beats pathwise (3.0412 vs 3.0199 heldout) — consistent
  with R-EP3-P (Conclusion F not used).
- RC-LR seed variance is large (0.17 spread) on the 20-prompt
  heldout; ACC-LR chosen by the RCAL rule is the stable claim.

## 4. Q-C — granularity (42-row structural grid + learned variants)

Structural (untrained, seed-best of 8, first path): identity 0.0415;
branchwise/dual ≈ 0.0415 (no gain); **pairwise (block-2 interleaved)
0.0254**; cross b4…b8192 0.0251–0.0254 (flat); full 0.0252;
Householder K≤64 ≈ identity (K reflections cannot mix 8192 dims
enough); butterfly 0.0252. Verdict: the cross-branch *structure* is
everything, the block size (beyond 2) is irrelevant locally —
choose the cheapest (pairwise/b16) for deployment; learned Cayley-32
adds the acceptance-relevant refinement on top (Conclusion E).

## 5. Q-D — foldability and fused kernels

See `docs/EAGLE_SCALE_ROTATION_FOLDABILITY_AUDIT.md` (proofs in
`tables/foldability_audit.json`):

- S0=S1=S2 **quantized-code identical** (not just dequant-close);
  recurrent rescale vs dual-table differs at 5.9e-5 code rate (fp16
  last bit) — S2 dual views are the bitwise-exact folding (+250 MiB).
- Weight-only folding negative control **breaks FP** — xT is
  mathematically required whenever activations quantize after the
  transform: cross-branch Q is unfoldable, period.
- Fused Triton kernel (concat + path scale + rotation + dynamic A4):
  integer-code agreement ≥ 0.99999 vs the official quantizer on all
  shapes (first/rec, tree sizes, odd token counts), 0.046–0.059 ms vs
  1.3–2.2 ms explicit Python path; **zero additional kernel launch**,
  no materialized intermediates. Conclusions **H** (scale) + **I**
  (rotation).

## 6. Decision rules (§23)

A **used** · B **used** (AR decoder) · C **used** (learned beats
fixed: ACC-LR vs FIXED-R val CIs exclude 0; mtbench AL SIG) ·
D **used** (acceptance-aware necessary: ACC/RC ≫ NMSE-LR at similar
or worse projection NMSE) · E **used** (small cross-branch blocks
sufficient) · F **not used** (pathwise loses) · G **not used** (AR
bottleneck has no valid frozen-weight rotation; QAT remains its fix)
· H **used** (scale fully foldable) · I **used** (rotation fusable
not foldable) · J **not used** (learned rotation IS worthwhile:
gains exceed the ~0 fused overhead).

## 7. Related work / novelty framing

SmoothQuant (2211.10438) scale migration; QuaRot (2404.00456)
Hadamard outlier rotation; SpinQuant (2405.16406) Cayley-learned
rotations with NMSE-type objectives. Novel here is none of the
machinery but the calibration TARGET: rotations trained against
**speculative-acceptance surrogates** (soft prefix survival, and its
reference-consistency-masked variant) with **RCAL-based selection**,
plus the negative results that (i) layer-NMSE objectives repeatedly
fail to transfer for EAGLE fusion, and (ii) the EAGLE distillation
objective itself only ties fixed rotations. Foldability analysis
under speculative-decoding execution (dual-path scaling, per-path
weight views, tree-shaped A4) is also, to our knowledge, new.

## 8. Limitations

Heldout sets are small (c4:20 calib; 40-prompt captures) — RC-LR
seed spread shows the noise floor; MT-Bench RCAL CI borderline;
ACC/RC surrogate dynamics unexplained beyond the honest paradox note;
AR-decoder bottleneck left to QAT (no rotation fix exists);
cross-dataset for ACC-LR not run; fake-quant only.

## 9. Artifacts

48-item `FINAL_OUTPUT.txt`; tables/ (component_audit, error_flow,
granularity_grid, foldability_audit, kernel_bench/verify,
lrgf_mechanism, learned_rot_heldout, rcal_metrics); stats/ bootstrap;
rotations/ (ckpts + json); 19 figures + 3D companions; tests
15/15; bundle `eagle1_learned_rotation_granularity_folding_<ts>.tar.gz`.
