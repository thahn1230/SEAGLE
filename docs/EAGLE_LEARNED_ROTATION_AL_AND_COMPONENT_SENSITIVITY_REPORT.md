# EAGLE learned-rotation AL & component sensitivity — FINAL REPORT

Run: `runs/eagle_learned_rotation_al_sensitivity_20260716_192113`
Branch `exp/eagle1-learned-rotation-al-sensitivity` (from `731e2c1`).
Evaluator: 80 MT-bench prompts, greedy, 128 new tokens, `mc_sim_7b_63`
(26 nodes), seed 0, KV FP16. Metric (verified from code): per verification
cycle `tau = accepted draft tokens + 1 bonus`; AL = mean tau; also reported
`mean_accepted_draft_tokens = mean_tau − 1`. All quantization is **fake
quant** (no latency claims): SpinQuant RTN, w_clip MSE, per-token
asymmetric activations, no GPTQ, no no-clip. Target W8A8/W4A4 both use the
single learned chat rotation `learned_chat_w4a4kv16`
(sha256 4b7e91d2a7531bb8…). Draft policy label:
`draft_proj+AR_W{b}A{b}_embedding_FP16_lmhead_FP16`.
Statistical procedure: per-prompt paired comparisons, 10,000 bootstrap
resamples, seed 0.

## 14.1 Executive summary

- **3×3 matrix complete (9/9)**; highest AL T16_D16 = 3.628, lowest
  T16_D4 = 1.046. The matrix reproduces the prior corrected matrix
  within noise — target-side wikitext-PPL gains from the learned rotation
  do NOT transfer to acceptance length.
- **W8A8 is a meaningful intermediate point on the target axis only**
  (T8_D16 −0.024, statistically zero) and on the draft axis under rotated
  interfaces (T8_D8 −0.216); D8 under the stock T16 interface loses −1.50.
- **Most sensitive draft component: the FIRST projection**, under both
  targets (fp16: −2.566 AL; w4a4: −1.950) — first > recurrent (−1.607 /
  −1.326) ≫ AR head (−0.135 / −0.156) ≫ LM head ≈ embedding ≈ 0.
- **Separate e/h activation scales recover only a fraction** (P2:
  1.253/2.076 vs shared 1.060/1.279) because the W4 weight grid, not the
  activation scale, is the binding constraint: the e-block dominates the
  per-row weight absmax.
- **Exact embedding scaling (P3) recovers most of the loss with a single
  shared activation scale**: projection-only 3.071 (85% of the FP16
  projection reference) under target fp16 and 3.216 (97%) under target
  w4a4. α is one folded global scalar per interface mode (45.25 = 2^5.5
  identity; 32.0 = 2^5 gamma_R1), calibrated on wikitext-2-train
  projection-output NMSE — never on MT-bench.
- **The fix transfers to the full W4A4 draft**: 1.046 → **2.956** (target
  fp16) and 1.256 → **3.052** (target w4a4). The draft-quality probe
  confirms the mechanism (draft top-1 agreement vs FP16 draft: 0.125 →
  0.625; KL 0.177 → 0.036).
- **No AL increase is degradation-induced** in this study: no
  configuration exceeds its FP16 reference beyond CI noise; the prior
  fixed-tree grader for the learned W4A4 target (189 rounds) already
  classified its residual accept-increases as benign (11 benign vs 1
  degradation-induced).

## 14.2 Exact 3×3 matrix (learned rotation, 80×128)

| cell | mean tau | accepted draft/cycle | 95% CI | Δ vs T16_D16 [CI] | quality note |
|---|---|---|---|---|---|
| T16_D16 | 3.6276 | 2.6276 | [3.528, 3.728] | — | exact_match 0.925 |
| T16_D8 | 2.1299 | 1.1299 | [2.080, 2.181] | −1.498 [−1.573, −1.426] | identity interface |
| T16_D4 | 1.0463 | 0.0463 | [1.040, 1.053] | −2.581 [−2.681, −2.483] | identity interface |
| T8_D16 | 3.6040 | 2.6040 | [3.498, 3.710] | −0.024 [−0.073, +0.026] | ≈ free (AL); path-sensitive† (exact_match 0.575) |
| T8_D8 | 3.4113 | 2.4113 | [3.315, 3.509] | −0.216 [−0.273, −0.157] | path-sensitive† (exact_match 0.4625) |
| T8_D4 | 1.2712 | 0.2712 | [1.258, 1.284] | −2.356 [−2.453, −2.262] | path-sensitive† (exact_match 0.4875) |
| T4_D16 | 3.3394 | 2.3394 | [3.240, 3.439] | −0.288 [−0.374, −0.204] | path-sensitive† |
| T4_D8 | 2.9428 | 1.9428 | [2.860, 3.028] | −0.685 [−0.766, −0.605] | path-sensitive† |
| T4_D4 | 1.2557 | 0.2557 | [1.241, 1.272] | −2.372 [−2.469, −2.278] | path-sensitive† |

† Established carry-over label: the W4A4 target's own verification is
partially execution-path-sensitive (cross-path top-1 0.906–1.0); its
naive-vs-EAGLE exact_match ≈ 0 in this study is that phenomenon (both
outputs are valid greedy trajectories of the same quantized model on
different execution paths), not an output-preservation bug — T16 rows show exact_match 0.925–0.9375,
while T8 (W8A8-target) rows show an intermediate degree of the same
phenomenon (exact_match 0.46–0.58; milder than W4A4's ≈0).

Full per-prompt/per-cycle data: `precision_matrix_per_prompt.csv`,
`precision_matrix_per_cycle.parquet`; extended §7.1 metrics (acceptance
rate, rejected-work ratio, verifier calls/token, first-rejection depth
histograms) in `precision_matrix_summary.csv`.

## 14.3 Component sensitivity (Study B, mean tau)

| config | target fp16 | Δ | target w4a4 | Δ |
|---|---|---|---|---|
| draft_all_FP16 | 3.6276 | — | 3.3303 | — |
| embedding_only_W4A4 | 3.6307 | +0.003 | 3.3619 | +0.032 |
| LM_head_only_W4A4 | 3.6350 | +0.007 | 3.3242 | −0.006 |
| AR_head_only_W4A4 | 3.4923 | −0.135 | 3.1738 | −0.156 |
| projection_recurrent_only | 2.0204 | −1.607 | 2.0039 | −1.326 |
| **projection_first_only** | **1.0615** | **−2.566** | **1.3805** | **−1.950** |
| projection_all | 1.0604 | −2.567 | 1.2792 | −2.051 |
| full_policy_W4A4 | 1.0463 | −2.581 | 1.2557 | −2.075 |
| diag embed W4A16 | 3.6296 | +0.002 | 3.3641 | +0.034 |
| diag embed-out A4 | 3.6233 | −0.004 | 3.3438 | +0.014 |

Ranking (both targets, by absolute and relative AL loss, accepted-token
loss, and rejected-work increase — order identical on all four criteria):
**first projection > recurrent projection ≫ AR head ≫ LM head ≈ embedding
≈ 0**. The first projection remains the dominant failure source; its blow
is softer under the gamma_R1 interface (rotated a_t input) than under the
stock-target identity interface, consistent with the established
activation-outlier mechanism.

## 14.4 Projection-scale comparison (Study C)

Projection-only W4A4 context (all else FP16):

| variant | target fp16 | target w4a4 |
|---|---|---|
| P0 projection_FP16 | 3.6276 | 3.3303 |
| P1 shared concat scale | 1.0604 | 1.2792 |
| P2 separate e/h scales | 1.2526 | 2.0764 |
| **P3 α-scaled shared scale** | **3.0710** | **3.2156** |

Full-draft W4A4 transfer:

| variant | target fp16 | target w4a4 |
|---|---|---|
| baseline projection | 1.0463 | 1.2557 |
| separate scales | 1.2036 | 1.9983 |
| **α-scaled** | **2.9558** | **3.0518** |

Reconstruction (calibration NMSE, `projection_scale_reconstruction__*.csv`):
P1 shared alpha=1 NMSE ≫ P2 separate > P3 at chosen α (objective 0.246
identity / 0.072 gamma_R1); the alpha sweep
(`embedding_alpha_sweep__*.csv`) shows a wide flat optimum per mode (within ~8% over 2^5–2^6 under
identity; within ~4% over 2^4.75–2^5.5 under gamma_R1), mirroring the
measured h-vs-e absmax ratio (~308× under identity h_t; ~20× under
gamma_R1 a_t, first path; `activation_distribution_stats__*.csv`).

Why P3 ≫ P2: separate activation scales fix only the input quantization;
the reparameterization additionally divides the e-block weight columns by
α, collapsing the e-block's dominance of the per-output-channel W4 grid
(w_e/w_h absmax ratio ~2.8–19.9 → ~α-fold smaller), so the SAME W4 budget
finally resolves the h-block weights. The draft probe agrees: top-1
agreement vs the FP16 draft is 0.125 (P1) / 0.25 (P2) / 0.625 (P3),
KL 0.177 / 0.141 / 0.036 (small-n diagnostic, 8 contexts).

## 14.5 Interpretation (required questions)

1. **Does wikitext-PPL improvement from the learned rotation transfer to
   EAGLE AL?** No. The learned-rotation matrix equals the random-Hadamard
   control matrix within noise (prior study; reproduced here cell-exactly).
2. **Target vs draft precision?** Draft precision dominates: D4 costs
   −2.36 to −2.58 while T4 costs −0.288 (and T8 ≈ 0).
3. **Is the projection still the primary bottleneck?** Yes — the only
   component whose solo quantization collapses AL.
4. **First vs recurrent?** First is more sensitive under both targets
   (−2.566 vs −1.607 fp16; −1.950 vs −1.326 w4a4).
5. **Does separate slice scaling solve it?** No — partial recovery only
   (1.25/2.08). Activation scales are not the binding constraint at W4.
6. **Can exact embedding scaling recover most of the benefit with a shared
   scale?** Yes: 85–97% of the projection-only reference and 81–92% of the
   full-draft reference, with CIs excluding the baselines by wide margins.
7. **Is scalar scaling sufficient under asymmetric quantization?** Yes —
   P3 (one shared scale + one shared zero point after α) beats P2
   (independent per-slice scales AND zero points) decisively, because the
   weight-grid geometry it fixes matters more; independent zero points are
   not necessary once ranges are matched.
8. **Does the fix survive full-draft W4A4?** Yes: 1.046→2.956 (fp16) and
   1.256→3.052 (w4a4); the residual gap to the fp16 draft (~0.3–0.7)
   is still predominantly remaining projection damage under target fp16
   (projection-only P3 already shows −0.56 of the −0.67 full-draft gap);
   the AR head adds only ~−0.1.
9. **Recommended deployment configuration**: learned-rotation target
   (W8A8 free in AL but partially path-sensitive; W4A4 at −0.29 AL, strongly path-sensitive) + draft
   full-policy W4A4 **with the α-folded projection reparameterization**
   (α per interface mode, zero runtime cost — folded into E' and W');
   embedding and LM head may be quantized freely (≈0 effect). If maximum
   AL is required, keep the draft projections at ≥8 bit (T8_D8 = 3.411).

No "solves" claim is made for P3: it recovers most but not all AL
(residual −0.56 [CI −0.61, −0.50] vs the fp16-draft reference under
target fp16), with quality diagnostics (probe, exact_match on T16 row)
supporting that the recovered acceptances are genuine draft-quality
improvements, not verifier degradation.

## Stop gates

- **Gate A (provenance)**: PASS — commit 731e2c1 base; learned rotation
  sha256 recorded; no random-Hadamard artifact loaded (the 3×3 matrix runner logs the
  resolved R.bin path + R1 hash `ee63bf94ecd5` on its rotated builds; all
  other rotated builds resolve the same named `learned_chat_w4a4kv16/R.bin`
  via `r_bin_path`, which raises on a missing named rotation); GPTQ
  disabled; w_clip on; KV FP16.
- **Gate B (rotated FP16 equivalence)**: PASS — stock vs learned-rotation
  FP16 target: greedy 8/8 identical 64-token continuations, prefill top-1
  agreement 1.0, logit rel-L2 ≤ 0.006 (fp16 GEMM scale), EAGLE AL paired
  Δ +0.021 [0.000, 0.045] (within 0.05 equivalence margin).
- **Gate C (quantizer effectiveness)**: PASS — every quantized module
  reports its kernel + bits in per-run `grid__*.json` / matrix meta
  (`FakeW4A4Linear(w4a4)`, `BranchwiseActLinear(w4,eA4,hA4)`, n_forward /
  n_act_quant counters > 0); W4A4 cells reproduce established collapse
  signatures, impossible with FP16 leakage.
- **Gate D (draft-only isolation)**: PASS — per-config pre/post snapshots
  (object ids, data_ptrs, weight SHA256, fixed-batch target logits
  bit-equality) enforced as a hard runtime failure; 34/34 configs clean
  (test_draft_target_module_isolation_records).
- **Gate E (projection FP invariance)**: PASS — exact algebra test for
  first- and recurrent-style weights at α ∈ {0.5, 2, 37.27, 512}
  (fp32 exact; fp16 within GEMM tolerance).
- **Gate F (complete matrix)**: PASS — 9/9 cells.

Determinism: T16_D16 run twice on the same GPU — per-prompt AL and
per-cycle acceptance lists bit-identical.

## Deviations & caveats

- GPUs 0–5 only (per instruction); GPU 6–7 untouched; one config per GPU.
- α was calibrated per interface mode (identity vs gamma_R1) because the
  hidden-slice dynamic range differs ~16× between modes; each mode uses
  ONE global folded scalar as specified.
- W4A4-target rows carry the established path-sensitivity label; their
  exact_match-vs-naive ≈ 0 is expected under that label.
- Draft-agreement probe uses 8 contexts (directional diagnostic only).
- All results are fake-quant; no real-kernel latency claims.
