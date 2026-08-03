# R-EP3-P: Projection-Local Orthogonal Rotations on Top of EP3-P

**Run**: `runs/eagle1_rotated_ep3p_projection_20260803_103956`
**Branch**: `exp/eagle1-rotated-ep3p-projection` (source commit
`d1cefba35a4956a3f56c923bc72ca6b9afb3cf6b`, verified)
**Anchor / target rotation**: SHA256 in `manifests/provenance.json`
(anchor `a7c6ccc8…`, same validated artifacts as the GQ and
visualization studies). Tests 17/17.

## 0. Headline

A cross-branch structured orthogonal rotation composed after EP3-P
scaling cuts the calibration W4A4 projection-output NMSE by ~40%
(first 0.0415 → 0.0251, recurrent 0.0327 → 0.0188; fp16-target
control: 0.204 → 0.023), with weight-W4 NMSE halved — the local
quantization problem is genuinely, mechanically improved. **The
improvement does not transfer to speculative acceptance**: the best
rotated arm ties EP3-P on full MT-Bench (dAL +0.036 CI [−0.039,
+0.109], dRCAL +0.007 CI [−0.074, +0.081]) and the proxy-best
pathwise arm is significantly *worse* deployed (dAL −0.107 CI
excludes 0). **Conclusion E** — after EP3-P, the fusion-projection
error is no longer the acceptance bottleneck — with Conclusion B
supported *locally* (mixing is what fixes the projection) and
Conclusion G at current kernel maturity.

## 1. Motivation: the measured scalar sweet spot

The EP3-P visualization study showed the scalar migration factor is a
coupled compromise: at the optimum m, 46% of embedding-side weight
entries already quantize to code 0, while the activation e/h RMS gap
is only partially closed (0.355), and pushing m further collapses the
weight side (100% zero-codes by m≈220) while widening the per-token A4
step under the h-slice. Hypothesis: a function-preserving orthogonal
rotation Q can redistribute coordinate-wise magnitudes so neither
quantizer faces the imbalance.

## 2. Transform and orientation (audited)

`x' = x T`, `T = S Q` (SR, primary) or `Q S` (RS, ablation);
PyTorch storage `W_pt [out=4096, in=8192]`, forward `y = x @ W_pt.T +
b`, so `W'_pt = W_pt T^{-T}`; for SR this is `(W_pt S^{-1}) Q` — i.e.,
scale the input columns then right-multiply each row by Q, the same
`apply()` used on activations (derivation + unit test
`test_rotated_ep3p_weight_orientation`; dense-vs-implicit agreement
1e-16). Q is implicit: `P_out · H_block · D_sign · P_in`
(normalized blockwise FWHT, ±1 diagonal, deterministic interleave) —
orthogonal by construction, stored as seeds + hashes only. **Q is not
R_T, not R1, not R_D**: it preconditions only the assembled
feature-fusion projection input; all validated interface semantics
(gamma exactly once on first path, no gamma on recurrent, embedding
untouched upstream) are preserved and tested. Deployment: identity-Q
arm reproduces EP3-P *bit-for-bit* (tau 3.2105 = 3.2105 on the smoke
set).

## 3. Search (never on MT-Bench)

- **S1** identity betas reproduce EP3-P: first 0.40 (j_out 0.0415 —
  matches the visualization study to 4 decimals), rec 0.43.
- **S2** 2,238 candidates (5 families × block sizes 16…8192 × 16
  seeds × SR/RS at fixed betas + rotation-only rows):

| family | best j_out first | best j_out rec |
|---|---|---|
| identity (EP3-P) | 0.0415 | 0.0327 |
| e_only / h_only / dual | 0.0415–0.0419 (no gain) | 0.0322–0.0326 |
| **cross / full (mixing)** | **0.0251** | **0.0195** |
| rotation-only (β=0) | 0.5949 | 0.4362 |

  Three facts: (i) only **cross-branch mixing** helps the projection —
  within-branch redistribution does nothing; (ii) scaling remains
  essential (rotation-only is catastrophic) — rotation and EP3-P are
  complementary *locally* (F, locally); (iii) block size is
  irrelevant once every block spans both branches (b=16: 0.0253 vs
  b=8192: 0.0251) — the cheapest transform captures the gain. Seed
  variance is negligible.
- **S3** beta recalibration under rotation: optimum shifts down
  (first 0.40→0.38, rec cross 0.45→0.40/full 0.41) and the curve
  flattens (beta overlay figures) — rotation does widen the low-error
  beta region, as hypothesized.
- **S4** 25 pairs → **S5** top-9 on held-out 20-prompt acceptance →
  top-3 + ablation arms on full MT-Bench. The held-out-acceptance
  winner pairs cross(first) with **dual(rec)** — acceptance rejects
  recurrent-path mixing that the proxy prefers, an early sign of the
  proxy/end-metric disconnect.
- **R6** blockwise-Cayley refinement (frozen weights, STE deployment
  quantizers, 3 seeds/path, 135s each, orth err 3e-8): 0.0252→0.0250
  / 0.0195→0.0192 — fixed structured rotations are already
  near-optimal in this class; not deployed.

## 4. End-to-end results (MT-Bench 80, W4A4 target + W4A4 draft, KV16)

| arm | tau | RCAL | SAL | AFS |
|---|---|---|---|---|
| NPTQ | 1.2521 | 0.233 | — | 0.914 |
| EP3-G | 2.9955 | 1.580 | — | — |
| EP3-P (re-run) | 3.0728 | 1.641 | 0.361 | 0.848 |
| R-EP3-P calib-best (cross+dual, β .38/.44) | 2.9741 | 1.563 | 0.333 | 0.847 |
| R-EP3-P **shared-Q** (cross both) | **3.1229** | **1.648** | 0.391 | 0.829 |
| R-EP3-P full-concat | 3.1106 | 1.596 | 0.369 | 0.836 |
| R-EP3-P block-diag | 2.9786 | 1.559 | 0.328 | 0.851 |
| fixed-beta + best Q | 3.0657 | 1.624 | 0.378 | 0.838 |
| LP3-RD (external) | 3.1594 | 1.682 | — | — |
| LP3-QAT (external) | 3.3090 | 1.821 | — | — |

Paired prompt-cluster bootstrap (3000):

- **EP3-P → shared-Q R-EP3-P**: dAL +0.036 [−0.039, +0.109], dRCAL
  +0.007 [−0.074, +0.081] — no significant gain.
- **EP3-P → calib-best pathwise**: dAL **−0.107 [−0.188, −0.018]**
  (significantly worse), dRCAL −0.079 [−0.162, +0.002].
- Block-diag → shared-Q: dAL **+0.152 [+0.053, +0.247]** — when you
  rotate at all, cross-branch beats block-diagonal end-to-end (B).
- Shared → pathwise: **−0.143 [−0.240, −0.040]** — pathwise rotation
  *hurts*; D is rejected in reverse.
- Recalibrated-beta → (same Q, orig beta): fixed-beta arm not worse
  (−0.106 pathwise arm vs fixed −0.036 n.s.) — beta recalibration on
  the proxy did not help deployment.
- LP3-RD → R-EP3-P: −0.167 [−0.267, −0.069], dRCAL −0.119 [−0.227,
  −0.016] — the residual-rotation baseline stays ahead.
- R-EP3-P → LP3-QAT: +0.359 / +0.258 — QAT-on-P3 dominates all
  frozen-weight arms.
- Reproducibility control: this run's EP3-P vs the GQ run's EP3-P
  capture — CIs span 0 ✓.

## 5. Mechanism (measured, §21)

- **M4 (mixing)**: first-path cross rotation distributes embedding
  energy uniformly: e-contribution fraction 0.088 ± 0.020 per rotated
  coordinate (vs 0/1 bands for dual) — contribution figures.
- **M2/M3 (weight side)**: W row absmax −23% and W_e-region zero-code
  rate 0.300→0.143; W4 NMSE 0.0196→0.0121. This is the dominant local
  effect: mixing rescues the embedding-side weight from the row
  quantizer, the exact failure the U-curve study identified.
- **M1 (activation)**: absmax 5.3→4.2 (first); A4 NMSE ~flat
  (0.0212 vs 0.0327 identity at the respective betas).
- **Proxy↔end correlation**: across deployed arms tau and RCAL
  correlate (r=0.82), but *local j_out does not rank deployed arms*:
  the j_out-best configuration is deployed-worst among rotated arms.
  Post-EP3-P projection NMSE (~0.02–0.04) is simply below the
  end-to-end noise floor — the acceptance bottleneck has moved
  elsewhere (verifier drift SAL ≈ 0.33–0.39, AR-head/decoder error,
  exposure bias). Consistent with the GQ finding that EP3-P vs LP3
  (NMSE 0.030 vs 0.042) was already indistinguishable in AL.

## 6. Overhead (§20, honest levels)

- Implicit FWHT math cost is trivial: 2D·log2(b) adds/token (b=16:
  ~65k adds, 260KB traffic).
- **Current unfused torch implementation**: 0.29–0.83 ms per rotated
  forward — 8–24× the (tiny) projection linear itself; end-to-end
  ~+2% ms/token (45.2→46.0, shard timing under scheduler load —
  indicative only). No packed-INT4 kernels exist in this repo
  (fake-quant study), so no measured INT4 overhead is claimed.
- Memory: rotation stored as seeds (KBs); transformed weight views
  same size as EP3-P's (64 MiB fp16 per path view).
- With the observed acceptance tie, ANY nonzero overhead makes
  deployment unjustified today (**Conclusion G** at current kernel
  maturity); a fused b=16 interleaved transform would be near-free if
  a future draft ever becomes projection-limited again.

## 7. Decision rules (§22)

- **A** (rotation resolves the scalar compromise): **NOT used** — the
  beta curve does flatten/lower locally, but MT-Bench RCAL does not
  significantly improve.
- **B** (cross-branch mixing necessary): **used, scoped** — necessary
  for the *local* gain (block-diag gains nothing locally and is
  −0.152 AL deployed vs shared cross).
- **C** (within-branch sufficient): rejected (complement of B).
- **D** (pathwise rotations needed): **rejected — reversed**: shared
  Q beats pathwise Q deployed (+0.143 AL).
- **E** (local gain, no acceptance gain): **USED — primary
  conclusion.** 40% local NMSE reduction, zero significant AL/RCAL
  movement.
- **F** (rotation and EP3-P complementary): **used locally only**
  (rotation-only 0.59 vs scale+rotate 0.025); not an end-to-end
  claim.
- **G** (not worthwhile after runtime cost): **used** at current
  kernel maturity.

## 8. Related work and novelty (audit)

- **SmoothQuant** (Xiao et al., arXiv 2211.10438): diagonal
  activation→weight scale migration — EP3-P is a structured two-scalar
  special case on a concat input; our per-channel extension experiment
  (visualization study addendum) matches its formula.
- **QuaRot** (Ashkboos et al., 2404.00456): Hadamard rotations to
  kill activation outliers for W4A4 LLMs — same transform family as
  our full/cross rotations, applied network-wide; our Q is a single
  projection-local preconditioner over a *concatenation of two
  differently-scaled sources*, which is the novel setting.
- **SpinQuant** (Liu et al., 2405.16406): learned (Cayley) rotations
  at network interfaces — our R6 is the same optimization idea scoped
  to the draft fusion projection; result: structured-fixed ≈ learned
  here.
- Recent calibration-optimized rotation lines (e.g., OSTQuant/DFRot
  class, 2024–2025) optimize rotations against calibration
  objectives; our contribution is not the rotation machinery but the
  *negative transfer result*: for EAGLE draft fusion under a W4A4
  verifier, projection-local rotation fixes the local objective and
  buys no acceptance — evidence that acceptance-level evaluation
  (AL/RCAL), not layer NMSE, must gate such methods. No prior work
  found evaluating projection-local rotations against speculative
  acceptance; the name R-EP3-P is ours. Claims scoped accordingly.

## 9. Limitations / publication-safe claims

- Proxy tensors: 16-prompt C4 calib (deterministic, held-out);
  acceptance selection on 20-prompt C4; RCAL on 40 MT-Bench prompts.
- e2e latency from shards is load-contaminated; micro-bench is clean.
- RS order deployed only via proxy (SR deployed end-to-end); Cayley
  arm not deployed (proxy-equal to fixed).
- Safe claims: (1) cross-branch structured rotation + EP3-P is
  FP-exact and cuts W4A4 fusion-projection NMSE ~40% (~9× on the
  fp16-target first path); (2) under a W4A4 verifier this does not
  move EAGLE acceptance (CIs span 0) and proxy-optimal configs can
  significantly hurt; (3) local layer NMSE is an unreliable selector
  once past the structural-collapse regime — use AL/RCAL.

## 10. Artifacts

46-item `FINAL_OUTPUT.txt`; candidates/*.jsonl (2,238 rows) +
s1–s5 JSONs + Cayley checkpoints; cycles/ (6 fresh captures + reused
GQ baselines); stats/rcal_bootstrap_mtbench.json; figures in plots/,
activations/, weights/, outputs/, heatmaps/, log_scale/ with NPZ/CSV;
tables/ (projection metrics, mechanism, overhead, fp16 control);
bundle `eagle1_rotated_ep3p_projection_<ts>.tar.gz` + `_latest`.
