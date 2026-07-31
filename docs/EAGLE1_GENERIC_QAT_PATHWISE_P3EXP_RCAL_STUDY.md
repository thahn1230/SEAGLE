# Generic-QAT vs Training-Free Structural P3, Pathwise Exponent Migration, and Reference-Consistent Acceptance Length (RCAL)

**Run**: `runs/eagle1_generic_qat_pathwise_p3exp_rcal_20260731_154652`
**Branch**: `exp/eagle1-generic-qat-pathwise-p3exp-rcal` (from `3f77b6c`)
**Anchor**: `checkpoints/eagle1_fresh_fp16_anchor/anchor.pt`
(SHA256 `a7c6ccc85ac1e1db7aa8d03e8f06294ed9f1b0faa7bdd869c3756f5fe4fbbaa0`, verified)
**Status**: COMPLETE (2026-08-01). Tests 21/21. Offline replay
equivalence 689/689 cycles across 3 spot-checked methods.

> **RCAL measures fidelity to an FP16 reference target. It is NOT a
> measure of absolute semantic correctness.** A method can be perfectly
> reference-consistent and still be wrong wherever the FP16 system would
> be; a quantized verifier that "fixes" a reference mistake is scored as
> drift. This caveat applies to every RCAL/SAL/LAL/AFS number below.

## 1. Questions and headline answers

1. **GOAL A — can generic QAT recover the structural projection
   failure?** **No.** A bitwise-generic W4A4 draft QAT (alpha pinned to
   1.0, no migration path, audited) recovers NPTQ 1.2521 → 1.74
   (MT-Bench tau, T1 best-val), but stays ~1.3 tau below training-free
   LP3/EP3-P (2.99/3.05), with paired CIs excluding zero in **both** AL
   and RCAL. Weight geometry confirms the mechanism: GQAT leaves the
   W_e/W_h imbalance essentially untouched (§8).
2. **GOAL B — does exponent/pathwise migration beat legacy P3?** The
   exponent parameterization *recovers* the legacy calibration from
   first principles (fp16 beta 0.46 → m 45.89 ≈ legacy 45.25; int4
   EP3-G beta 0.42 → m 32.90 ≈ legacy 32.0) — an independent
   validation — but pathwise asymmetry gives **no statistically
   significant** improvement (Conclusions D/E rejected).
3. **GOAL C — are AL gains reference-consistent?** **Yes, all of
   them.** Across 12 pre-registered method pairs, no deceptive AL gain
   was found: every significant deployed-AL gain comes with a
   significant RCAL gain of the same sign (Conclusion H; G never
   fires). Quantized-verifier drift exists (SAL ≈ 0.34–0.43 per cycle
   on T1, AFS 0.84–0.91) but it does not *manufacture* ranking gains.

## 2. Method definitions (final names)

| Name | Definition |
|---|---|
| F16 | fp16 target + fp16 stock draft (upper anchor) |
| NPTQ | naive W4A4 draft PTQ, no migration (alpha = 1) |
| GQAT | **generic** W4A4 draft QAT: `migration_mode="none"`, factor 1.0, no embedding multiply, no W_e division, no P2, no rotation training; same quantizer contract/anchor/teacher/budget as prior QAT arms |
| LP3 | training-free structural D4P3, legacy calibrated alpha (45.254834 fp16-target / 32.0 int4-target) |
| EP3-G | exponent P3 `m = D**beta`, single global beta |
| EP3-P | exponent P3, independent (beta_first, beta_recurrent) |
| LP3-RD | LP3 + learned independent draft rotation R_D (`LK2_AUXG_s0`) |
| LP3-QAT | prior (non-generic) QAT on the P3 fold, re-evaluated here |

Targets: T0 = fp16; T1 = SpinQuant W4A4 (KV16,
`learned_chat_w4a4kv16`). GQAT is the validated QAT trainer with alpha
pinned to 1.0 — `E*1.0`/`W/1.0` are bitwise identities, so step 0 ≡
NPTQ. Audited: 14 `manifests/gqat_audit_*.json`, all
`migration_mode=none`.

## 3. Pre-registered decisions (recorded before results)

- AFS thresholds {0.90, 0.95, 0.98}; EP3-P primary **Rule 3** = max
  AL_q s.t. AFS ≥ 0.95 among MT-Bench top-3 (fallback max AFS — the
  fallback WAS needed on int4, see §7); non-inferiority margin 0.05 in
  both RCAL and AL; beta search never touches MT-Bench until the final
  top-3; LR pilot {3e-7, 1e-6, 3e-6, 1e-5} by 20-prompt calib AL.

## 4. RCAL protocol and controls

Per deployed cycle (greedy): record prefix length, flattened proposal
tree, Tq-accepted `(best_q, R_q)`; verify the *same* tree from the
*same* prefix under FP16 T0 in lockstep on a second GPU, reference KV
always compressed along the **deployed** trajectory. `R_RC` = LCP of
accepted proposal sequences. Aggregates ratio-of-aggregates, FP64.

- **Identity control (passed)**: T0-as-both gives RCAL = AL_q = AL_0 =
  2.5608, SAL = LAL = 0, AFS = 1 (148 cycles); identities to 1e-12.
- **Offline replay equivalence (passed)**: trajectory reconstruction
  from records alone reproduces inline lockstep S_0 exactly — RC_F16
  240/240, RC_LP3_T1 278/278, RC_LP3RD_T1 171/171.
- **T0-target internal consistency**: for every T0-deployed method,
  SAL = LAL = 0 and dRCAL ≡ dAL_q by construction — observed exactly
  (item 15–18 T0 rows), a live sanity check on the pipeline.
- Acceptance-length contract audited
  (`docs/EAGLE_ACCEPTANCE_LENGTH_CONTRACT.md`): official tau = 1 +
  accepted proposal length; RCAL metrics live on proposal tokens.

## 5. First/recurrent distributions and the exponent search

- e/h RMS ratio of the raw projection inputs: fp16 target first
  **0.0070** vs recurrent **0.0091**; int4 target first **0.0127** vs
  recurrent **0.0089**. Two orders of magnitude of e-slice
  under-scaling in every case (the structural D4 failure), with a
  path asymmetry whose **direction reverses across targets** (the int4
  first path passes the gamma_R1 interface basis).
- NMSE-optimal exponents mirror this: fp16 (beta_first 0.46, beta_rec
  0.44); int4 (0.40, 0.44). Best pairs by NMSE track the 1-D optima.
- **Legacy alphas are re-derived**: beta 0.46 → m 45.89 (legacy
  45.254834), EP3-G int4 beta 0.42 → m 32.90 (legacy 32.0). The
  exponent grid finds the legacy calibration without seeing it.

## 6. GQAT training

- LR pilot monotone in LR; both variants select 1e-5 (**grid-top
  boundary caveat** — GQAT numbers are a lower bound in LR).
- Best-val is EARLY: calib AL peaks at step 500–1000 and declines
  (T0 s0 best step 500, calib 1.4473); final < best on MT-Bench for
  every seed (e.g., T1: best 1.71–1.74 vs final 1.53–1.55). Generic
  QAT overfits its distillation objective without improving deployed
  acceptance.
- Seeds are tight: T1 best {1.7212, 1.7360, 1.7102}; T0 best {1.5280,
  1.5144, 1.5310}.

## 7. Main results (MT-Bench, official tau; RCAL block on 40-prompt captures)

| Method | tau T0 | tau T1 | AL_q | AL_0 | RCAL | SAL | LAL | P_A | R_A | AFS |
|---|---|---|---|---|---|---|---|---|---|---|
| F16 | 3.5762 | — | 2.4404 | 2.4404 | 2.4404 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| F16D (fp16 draft@T1) | — | 3.2744 | 2.1816 | 2.0349 | 1.7628 | 0.4189 | 0.2721 | 0.808 | 0.866 | 0.836 |
| NPTQ | 1.0444 | 1.2521 | 0.2558 | 0.2550 | 0.2334 | 0.0224 | 0.0217 | 0.912 | 0.915 | 0.914 |
| GQAT (s0 best) | 1.5280 | 1.7212 | 0.7121 | 0.6841 | 0.6291 | 0.0830 | 0.0550 | 0.883 | 0.920 | 0.901 |
| LP3 | 2.9065 | 2.9973 | 1.9337 | 1.7914 | 1.5816 | 0.3521 | 0.2098 | 0.818 | 0.883 | 0.849 |
| EP3-G | 2.9146 | 2.9955 | 1.9192 | 1.7898 | 1.5802 | 0.3391 | 0.2096 | 0.823 | 0.883 | 0.852 |
| EP3-P | 2.9146 | 3.0517 | 1.9665 | 1.8361 | 1.6069 | 0.3595 | 0.2291 | 0.817 | 0.875 | 0.845 |
| LP3-RD | — | 3.1594 | 2.0628 | 1.9359 | 1.6815 | 0.3813 | 0.2544 | 0.815 | 0.869 | 0.841 |
| LP3-QAT (s0 best) | 3.2164 | 3.3090 | 2.2544 | 2.1007 | 1.8208 | 0.4336 | 0.2799 | 0.808 | 0.867 | 0.836 |

(T1 rows for AL_q…AFS; T0 captures give SAL=LAL=0 identically.)

- EP3-P primary pairs (Rule 3): fp16 = diagonal (0.46, 0.46), AFS
  1.0; int4 = **fallback** (0.40, 0.45), AFS 0.8452 — **no int4 top-3
  pair reached AFS ≥ 0.95**; every W4A4-target method sits at AFS
  0.836–0.914, i.e., the pre-registered 0.95 bar is unreachable under
  this target, which is itself a finding: W4A4 verifier drift is a
  property of the *target*, not the draft (F16D, with an fp16 draft,
  has the same AFS 0.836).
- LP3-QAT seeds T1 best {3.3090, 3.3147, 3.3112} — reproduces the
  prior study within noise.

### Paired bootstrap (prompt-cluster, 3000 reps, delta = B − A)

| Pair (A→B) | dAL_q [CI] | dRCAL [CI] | deceptive? |
|---|---|---|---|
| LP3 → GQAT (T1) | −1.2216 [−1.317, −1.132] | −0.9525 [−1.055, −0.856] | no |
| EP3-P → GQAT (T1) | −1.2544 [−1.339, −1.174] | −0.9779 [−1.075, −0.885] | no |
| LP3 → EP3-P (T1) | +0.0328 [−0.025, +0.088] | +0.0254 [−0.038, +0.085] | no |
| EP3-G → EP3-P (T1) | +0.0472 [−0.042, +0.127] | +0.0268 [−0.041, +0.089] | no |
| GQAT → LP3-QAT (T1) | +1.5423 [+1.425, +1.670] | +1.1917 [+1.078, +1.318] | no |
| EP3-P → LP3-QAT (T1) | +0.2879 [+0.208, +0.376] | +0.2139 [+0.138, +0.297] | no |

T0 pairs mirror these with dRCAL ≡ dAL_q (see §4): GQAT −1.2952/−1.2892
vs LP3/EP3-P; LP3-QAT +1.5949/+0.3058 vs GQAT/EP3-P; LP3↔EP3-P −0.006
(CI spans 0); EP3-G ≡ EP3-P (identical config, delta exactly 0).

## 8. Does generic QAT learn P3-like rebalancing? **No.**

W_e/W_h RMS ratio: anchor **13.428**; the P3 fold would put it at
**0.297** (T0) / **0.420** (T1). After 3000 GQAT steps: **12.49** (T0)
/ **13.05** (T1) — ~7%/3% of the required log-distance. The
quantizer-only gradient signal does not discover the ~45x/32x
rebalancing that P3 encodes in closed form; consistent with the ~1.3
tau gap. (`generic_qat_weight_ratio_trajectory_3d.png`)

## 9. Tree vs sequential verifier fidelity + shape-dependent A4

| capture | Tq tree-vs-seq | Tq-seq vs T0-seq | T0-seq vs deployed |
|---|---|---|---|
| RC_F16 | 0.9753 | **1.0000** | 0.9753 |
| RC_LP3_T1 | 0.9099 | 0.8366 | 0.8346 |
| RC_GQAT_T1 | 0.9055 | 0.8414 | 0.8230 |
| RC_EP3P_T1 | 0.9111 | 0.8383 | 0.8312 |

- fp16 seq-vs-seq = 1.0 exactly (control passes).
- **Shape-dependent A4 WARNING fires**: under the W4A4 target, tree
  execution disagrees with sequential execution on ~9% of tokens
  (dynamic activation quantizers see different token mixes under tree
  masks). Even fp16 has a 2.5% tree-vs-seq gap (attention/position
  numerics), so ~6.5pt is A4-specific.
- Quantization drift (seq-vs-seq 0.84) is the dominant drift source,
  execution shape second.

## 10. Cross-dataset finalists (seed 0, SINGLE-SEED, labeled)

T1 tau: LP3-QAT {sharegpt 3.408, c4 3.270, gsm8k 3.767, humaneval
3.843} > EP3-P {3.124, 3.056, 3.460, 3.848} ≈ LP3 {3.100, 3.008,
3.508, 3.773} ≈ EP3-G >> GQAT {1.752, 1.690, 1.822, 1.915} >> NPTQ
{1.241, 1.210, 1.272, 1.296}. Same ordering as MT-Bench on every
dataset; EP3-P's edge over LP3 is dataset-dependent (+humaneval/c4,
−gsm8k). fp16-target rows in `FINAL_OUTPUT.txt` item 32.

## 11. Overhead (§10)

EP3-P vs LP3 deployed: time ratio **1.001** (49.58 vs 49.54 ms/token),
peak GPU memory delta **0 MiB** — the pathwise design (shared embedding
table + one elementwise rescale of the recurrent e-slice) is free.

## 12. Decision rules A–H — verdicts

- **A — "Generic QAT cannot recover the structural projection
  failure": USED.** GQAT below LP3 and EP3-P by >1.2 AL and >0.95
  RCAL, all four CIs exclude 0 (margin 0.05 trivially exceeded).
- **B — "Generic QAT learns around the imbalance": NOT used.** Neither
  the AL condition nor the W_e/W_h rebalancing condition holds (§8).
- **C — "Training-free P3 competitive with generic QAT": USED
  (superseded by A's direction).** LP3/EP3-P are not merely
  non-inferior at margin 0.05 — they dominate.
- **D — "Path-specific exponent migration improves global P3": NOT
  used.** EP3-P vs EP3-G dRCAL +0.0268, CI [−0.041, +0.089] spans 0
  (AFS also does not improve: 0.845 vs 0.852).
- **E — "First/recurrent paths require different migration strengths":
  NOT used.** The optimal betas DO differ (int4 0.40 vs 0.45; the
  distributions differ, §5) but the pre-registered bar — a significant
  full-MT-Bench RCAL or constrained-AL improvement — is not met, and
  distribution differences alone are explicitly insufficient.
- **F — "D4P3 and QAT are complementary": USED.** LP3-QAT exceeds both
  GQAT (dAL +1.54, dRCAL +1.19) and EP3-P (+0.288, +0.214), all CIs
  exclude 0, both targets.
- **G — "AL gain is verifier-drift-driven": NOT used.** No pair
  triggers the deceptive flag (dAL_q>0 with dRCAL≤0).
- **H — "AL gain is reference-consistent": USED** for every positive
  pair (LP3-QAT over GQAT/EP3-P; P3-family over GQAT/NPTQ): dAL_q > 0
  with dRCAL > 0; precision stays in the method-family band (P_A
  0.81–0.88 across T1 methods; gains do not degrade P_A below peers).

## 13. Related work and novelty

See `docs/RCAL_RELATED_WORK_AND_NOVELTY_AUDIT.md`. Closest: QSpec
(arXiv 2410.11305), QuantSpec (2502.10424), ML-SpecQD (2503.13565),
HSD (2601.05724). None perform paired same-proposal replay under a
reference verifier with LCP decomposition; the name RCAL was not
found. Scope: no claim that verifier-drift analysis per se is novel;
the metric construction (lockstep same-tree replay, deployed-trajectory
KV, SAL/LAL/AFS decomposition with identity controls) is the
contribution.

## 14. Publication-safe claims

1. Generic (structure-agnostic) low-bit draft QAT does not recover the
   EAGLE concat-projection scale failure at matched budget; closed-form
   migration (P3) dominates, and QAT helps *on top of* it (A, C, F).
2. The exponent parameterization independently re-derives the legacy
   calibrated alphas (§5) — evidence the calibration is a property of
   the weight geometry, not the calibration set.
3. Deployed-AL rankings under a W4A4 verifier are, in this setting,
   reference-consistent (H; no deceptive gains) — but absolute
   deployed AL overstates FP16-consistent acceptance by SAL ≈ 0.34–0.43
   tokens/cycle, and AFS ≥ 0.95 is unattainable under this target.
4. Pathwise migration strengths *are* distributionally motivated and
   *are* free at runtime, but do not yield significant end-metric
   gains at 40–80 prompt scale (D/E).

## 15. Limitations

- RCAL is reference-fidelity, not semantic correctness (banner).
- GQAT LR at pilot grid top (boundary) — GQAT is a lower bound in LR;
  also only 3000 steps (best-val peaked by step 1000, so budget is
  unlikely to be the binding constraint).
- RCAL captures use 40 MT-Bench prompts (~2.5–2.9k cycles/method);
  cross-dataset rows single-seed; beta-pair AL/RCAL surfaces evaluated
  only on pre-registered top-9/top-3 subsets (masked cells).
- Task quality proxied by target wikitext-2 PPL (5.9851/6.0326,
  prior-run provenance, identical builds) + T0 token agreement; no
  end-task exact-match harness.
- fake-quant only (no real INT4 kernels); ms/token numbers compare
  like-for-like fake-quant paths.

## 16. Artifacts

- 38-item summary: `FINAL_OUTPUT.txt` (run dir)
- Figures: 12 conventional + 16 required 3D (+6 int4 companions), all
  with companion heatmaps and CSV/NPZ data (`plots/`, `plots/data/`)
- Raw: `cycles/*.jsonl` (per-cycle records incl. proposal trees),
  `shards/*.csv`, `stats/rcal_bootstrap_mtbench.json`, `tables/*`
- Tests: 21/21 (`manifests/test_log.txt`)
- Bundle: `eagle1_generic_qat_pathwise_p3exp_rcal_<ts>.tar.gz` + `_latest`
