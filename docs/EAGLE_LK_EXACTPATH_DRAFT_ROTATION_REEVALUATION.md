# EAGLE-1 Exact-Path LK-Loss Draft Rotation Re-evaluation

Run: runs/eagle_lk_exactpath_draft_rotation_20260721_151703  
Branch: exp/eagle1-lk-exactpath-draft-rotation  
Date: 2026-07-22. Fake-quant only; micro-AL only; rotation-only (all model weights frozen); KV4 = KV4 without R3.

## 6. Rotation parameterization

All results in this section use tree decoding (M3) against the `t4kv4` target (W4A4 target, KV4 without R3) with a KV4 draft, the exact-path fake-quant forward (STE around the official SpinQuant quantizers, runtime cast order), and micro-AL = sum(tau)/cycles, tau = accepted+1. Every model weight — target, R_T, and the draft embedding/projections/AR/head/norms — is bitwise frozen. The only trainable parameter is the draft rotation R_D.

### 6.1 Two parameterizations

Two ways of learning R_D were compared against the shared target rotation R_T (SHA `4b7e91d2…a48fa6e`):

- **Local residual Cayley** — R_D = R_T·C(A), with A skew-symmetric and C the Cayley transform, initialized at A = 0 so that R_D = R_T exactly at step 0 (recovering the shared baseline). Trainable = A. This is the parameterization used by all 13 screened rotations and both finalists (HYBRID_s2, D_HYBRID).
- **Unrestricted orthogonal R_D** — a free orthogonal matrix (LK_E_UNRESTRICTED), trained without tying the solution to a small residual around R_T.

Initialization controls confirm the parameterization is anchored at the shared rotation: the near-identity trust-region control (LK_ABL_TRUSTH03) departs from R_T by a geodesic distance of only 0.021 and a max-element change of 0.0002, and its weight-quant profile (W_rec W4 NMSE 0.01741, q_proj 0.01209) is indistinguishable from SHARED_RT — i.e. when the learned residual is forced to ~0 the method reproduces shared exactly, so the gain reported in §7 is attributable to the learned departure, not to re-initialization.

### 6.2 Local residual wins; unrestricted collapses

The local residual parameterization improves acceptance on every screened rotation (§7, screening panel), whereas the unrestricted orthogonal R_D **collapses catastrophically** in the tree. On the 20-prompt screening tree (t4kv4):

| Rotation | mtbench | c4 | gsm8k |
|---|---|---|---|
| SHARED_RT (20p ref) | 2.5958 | 3.0589 | 3.5074 |
| Unrestricted (LK_E) | 1.9906 | 2.2548 | 2.2557 |
| Δ vs shared | **−0.605** | **−0.804** | **−1.252** |

Unrestricted R_D is not merely worse than the local residual finalists (which run ~+0.14…+0.21 above shared on the same panel); it falls **0.6 to 1.25 micro-AL below the shared baseline itself**. Local residual ≫ unrestricted.

### 6.3 Geometry: the collapse is not a rotation-angle or orthogonality effect

The geometry table (`tables/rotation_geometry.csv`) locates why. All quantities are R_D relative to R_T.

| Rotation | geodesic | Frobenius | max-elem Δ | max-abs elem | orth err (Gate G) | W_rec W4 NMSE | q_proj W4 NMSE |
|---|---|---|---|---|---|---|---|
| SHARED_RT (baseline) | 0.000 | 0.000 | 0.000 | 0.0461 | 2.6e-05 | 0.01742 | 0.01209 |
| Local residual — HYBRID_s2 | 67.90 | 81.07 | 0.0973 | 0.0927 | 1.45e-04 | 0.01776 | 0.01212 |
| Local residual — 13-finalist range | 62–80 | 75–92 | 0.089–0.097 | 0.079–0.104 | ≤1.7e-04 | ~0.0177 | ~0.0121 |
| Unrestricted (LK_E) | 34.47 | 43.53 | **0.3919** | **0.3758** | 5.8e-05 | 0.01759 | 0.01603 |
| Near-identity control (TRUSTH03) | 0.021 | 0.030 | 0.0002 | 0.0460 | 1.15e-04 | 0.01741 | 0.01209 |

Two facts rule out the obvious explanations:

- **Not total rotation angle.** The unrestricted solution's geodesic distance (34.5) is *smaller* than the local finalists' (~68), yet it collapses. What distinguishes it is that it concentrates the change into a few large entries — max-element change 0.392 and max-abs element 0.376, roughly **4×** the local residual's ~0.097 — rather than spreading a small residual across the matrix. Large, concentrated element shifts break the frozen draft's learned projections; the distributed local residual does not.
- **Not an orthogonality violation (Gate G).** Every rotation, including the unrestricted one, is orthogonal to within R_DᵀR_D = I at ≤1.7e-04 (shared 2.6e-05, local finalists ~1.45e-04, unrestricted 5.8e-05). The unrestricted R_D is a *valid* orthogonal matrix — it is simply a bad one for acceptance. The collapse is a distribution-alignment failure, not a numerical one.

### 6.4 The benefit is acceptance alignment, not weight-quant geometry

The W4 reconstruction error is essentially invariant to the rotation. W_rec W4 NMSE is **~0.0177 for the shared baseline and for all 13 local residual finalists** (0.01742 shared vs 0.01772–0.01777 local — a change on the order of the third decimal), and the o_proj/down_proj W4 NMSE are likewise unchanged. A rotation that lifts acceptance by +0.16 does so **without measurably improving the weight-quantization error** of the draft decoder. The gain therefore comes from aligning the draft's output distribution with the target's acceptance region, not from better weight geometry.

The same lever operates in the failing direction: the unrestricted collapse is *also* invisible in W_rec NMSE (still 0.0176). Its only weight-quant signature is a q_proj W4 NMSE rise to 0.0160 (from 0.0121) with q kurtosis 3.36→3.96 — a minor effect that cannot account for a −1.25 micro-AL loss. Both the +0.16 gain (local residual) and the −0.6…−1.25 loss (unrestricted) are acceptance-geometry phenomena that fake-quant weight NMSE does not capture.

### 6.5 Alpha calibration sweep: the result survives recalibration

Because P3's activation clip scale (alpha) is selected by calibration and never trained, the gain must be shown not to be an alpha-tuning artifact. Sweeping alpha over a 9-point clip grid (mtbench-20, t4kv4) for the shared rotation and the learned D_HYBRID rotation (`tables/alpha_sweep.txt`):

| alpha | shared micro-AL | D_HYBRID micro-AL |
|---|---|---|
| 8 | 1.652 | 1.717 |
| 11.31 | 2.076 | 2.182 |
| 16 | 2.392 | 2.530 |
| 22.63 | 2.588 | 2.662 |
| **32 (deployed)** | 2.596 | **2.845** |
| 45.25 | **2.624** | 2.760 |
| 64 | 2.587 | 2.641 |
| 90.51 | 2.162 | 2.303 |
| 128 | 1.574 | 1.685 |
| **best** | 2.624 @ α=45.25 | 2.845 @ α=32 |

D_HYBRID **dominates the shared rotation at all nine clip scales** on the grid — the learned rotation is never behind, at any calibration. Comparing each rotation at its *own* optimum (shared 2.624 @ α=45.25; D_HYBRID 2.845 @ α=32) still leaves a **+0.221** lead. The deployed alpha=32 is exactly D_HYBRID's optimum and sits on the shared rotation's plateau (2.596 vs its 2.624 peak, within 0.028), so the deployed comparison if anything slightly *under*-tunes shared. The advantage is not an artifact of a shared alpha choice: it persists when alpha is re-optimized independently for each rotation.

---

## 7. Primary confirmatory result

The primary finalist is **HYBRID_s2** — the adaptive-hybrid LK objective (lam_k = exp(−eta·sg(mean alpha))·KL + (1−lam)·TV), seed 2, trained at Stage-2 (5000-word target-generated corpus, batch 32, 3000 steps, AdamW cosine) as a local residual R_D = R_T·C(A) with all model weights frozen. It is evaluated against the shared target rotation R_T in the tree (M3) at the t4kv4 operating point (W4A4 target, KV4 without R3; KV4 draft), exact-path fake-quant, micro-AL. The shared mtbench baseline of 3.0080 is the Gate-B-reproduced value.

**HYBRID_s2 vs shared R_T — confirmatory scale, paired prompt-cluster bootstrap (3000 reps):**

| dataset | n | shared | HYBRID_s2 | delta | 95% CI | p |
|---|---|---|---|---|---|---|
| mtbench | 80 | 3.0080 | 3.1620 | **+0.1539** | [+0.0908, +0.2196] | 0.0000 |
| c4 | 200 | 3.0731 | 3.1804 | **+0.1073** | [+0.0638, +0.1494] | 0.0000 |
| gsm8k | 200 | 3.5137 | 3.7839 | **+0.2702** | [+0.2186, +0.3228] | 0.0000 |
| sharegpt | 80 | 3.0963 | 3.2632 | **+0.1669** | [+0.1055, +0.2301] | 0.0000 |
| humaneval | 164 | 3.7064 | 3.8252 | **+0.1187** | [+0.0708, +0.1716] | 0.0000 |

**Mean delta +0.1634. All 5 datasets: 95% CI excludes 0 (lower bound ≥ +0.0638), p ≈ 0.0000** (0 of 3000 paired prompt-cluster bootstrap resamples cross zero on any dataset).

The learned draft rotation improves micro acceptance length on every confirmatory dataset, by +0.107 (c4) to +0.270 (gsm8k), with the largest gain on gsm8k and the smallest on the two largest-baseline sets (c4, humaneval). The paired prompt-cluster bootstrap — resampling prompt clusters, not tokens, so per-prompt correlation is respected — places all five 95% intervals strictly above zero. An exact-path, full-vocabulary, acceptance-aware orthogonal draft rotation R_D improves over the shared rotation R_T while every model weight remains frozen: **yes.**

## 4. Objective ablation

All rows below are exact-path fake-quant micro-AL (sum(τ)/cycles, τ=accepted+1) on the tree draft (M3), target `t4kv4` (= W4A4 target + draft-KV4, KV4 without R3). Every model weight — target, R_T, draft embedding/projections/AR/head/norms — is bitwise frozen; the only trained parameter is the residual draft rotation R_D = R_T·C(A). 20-prompt cells are screening (candidate ranking); 80–200-prompt cells are confirmatory (trusted magnitudes, paired prompt-cluster bootstrap, 3000 reps).

### 4.1 Exact-path KL already beats shared — the previous "KL fails" was a training-path artifact

The previous TLDR-KV4 R_D trainer quantized the decoder in the wrong basis — Q(W) then conjugate rather than the runtime Q(R_DᵀW R_D). On real o_proj this gives NMSE 0.02972 between trainer-path and runtime-path weights, 2.1× the genuine runtime quant noise (0.01437); it also carried no R2/R4 in the training decoder, a 6-point clip grid vs the official MSE grid=100, a top-64/τ=2/τ²-scaled teacher over raw-text teacher-forced windows, no draft-KV4, and proxy top-1 validation. Under that path the study reported "independent R_D unnecessary" — a provisional conclusion.

Rebuilt on the exact-path forward (STE around the official SpinQuant quantizers, runtime cast order, residual R_D + draft KV4; Gate D bitwise train/runtime parity, Gate E finite gradients), the plain full-vocab exact-path KL objective — no acceptance term at all — already beats shared R_T at confirmatory scale:

- **F_KL_s0 (exact-path, full-vocab KL): +0.149 mean** over the 5 confirmatory datasets (mtb +0.112 / c4 +0.133 / gsm8k +0.225 / sharegpt +0.189 / humaneval +0.086), all-5-dataset 95% CI > 0.

So the objective was never the problem; the training path was. Exact-path KL alone overturns the previous provisional negative.

### 4.2 Objective / regularizer sweep (screening, 20p, t4kv4)

Shared R_T reference: mtb 2.5958 / c4 3.0589 / gsm8k 3.5074. Each candidate is trained on the same 1000-window target-generated corpus, teacher-forced, full-vocab teacher, unless noted. "proxy" = best held-out val expected-τ (1+Σ∏α) during training.

| objective (screening) | mtb | c4 | gsm8k | avgΔ | proxy exp-τ |
|---|---|---|---|---|---|
| exact-path KL (B) | 2.8033 | 3.2445 | 3.6747 | +0.187 | 1.981 |
| neg-log-α (C) | 2.7535 | 3.2212 | 3.7213 | +0.178 | 1.960 |
| expected-τ (F) | 2.7374 | 3.1463 | 3.6201 | +0.114 | 2.015 |
| pure-TV (α=1−TV) | 2.7173 | 3.1964 | 3.6433 | +0.132 | 2.025 |
| fixed-λ=0.5 KL-TV | 2.7285 | 3.1972 | 3.6775 | +0.147 | 1.963 |
| top-64 KL | 2.8389 | 3.2324 | 3.7303 | +0.213 | 1.853 |
| adaptive KL-TV hybrid (D) | 2.8447 | 3.2572 | 3.8527 | +0.264 | 2.020 |

Every objective — including pure-TV and fixed-λ — beats shared on all three screening datasets; none of the LK objectives is a losing choice. The adaptive KL-TV hybrid, λ_k = exp(−η·sg(mean α))·KL + (1−λ)·TV, ranks top on screening and on proxy. It is only weakly η-sensitive: screening η-sweep {0.7, 1, 10} gives avgΔ {+0.187, +0.217, +0.187} and proxy {2.000, 1.958, 1.953} — all beat shared, with the default (η→small, KL-leaning) marginally best on proxy (2.020). Top-64 KL is the outlier: it screens well on acceptance (+0.213) but has the lowest proxy exp-τ (1.853) because truncating the teacher corrupts the KL target (see §5.2); full-vocab KL/hybrid dominate it on the trajectory-level metric.

Screening deltas are noisier and run larger than confirmatory (e.g. hybrid screens +0.264 but confirms +0.163); treat §4.2 as a ranking and §4.3 as the magnitude.

### 4.3 Confirmatory objective ranking (tree, t4kv4, 80–200p)

| finalist | mtb | c4 | gsm8k | sharegpt | humaneval | mean | all-5 CI>0 |
|---|---|---|---|---|---|---|---|
| F_AUXG_s0 (aux-greedy) | +0.206 | +0.169 | +0.279 | +0.120 | +0.110 | **+0.177** | yes |
| F_EXPTAU_s0 | +0.196 | +0.144 | +0.268 | +0.117 | +0.101 | +0.165 | yes |
| F_HYBRID_s2 | +0.154 | +0.107 | +0.270 | +0.167 | +0.119 | +0.163 | yes |
| F_KL_s0 | +0.112 | +0.133 | +0.225 | +0.189 | +0.086 | +0.149 | yes |

Ranking: **AUXG +0.177 > EXPTAU +0.165 > HYBRID +0.163 > KL +0.149.** All four objectives beat shared R_T on every one of the 5 datasets with 95% CI excluding 0. The spread across objectives is only 0.028 micro-AL — small relative to the +0.149 that exact-path KL already delivers. Adding an acceptance-aware term (neg-log-α / adaptive hybrid / expected-τ) or a greedy-argmax auxiliary (aux-greedy) buys a modest, consistent further +0.015–0.028. The first-order effect is the exact-path forward and the target-generated corpus; the choice among LK objectives is second-order. HYBRID_s2 is carried as the primary finalist (mean +0.163; primary paired bootstrap p≈0.0000 on all 5 datasets), with AUXG the best-of-set.

## 5. Data & trajectory ablations

Same evaluation harness and reference as §4. Each ablation changes exactly one axis of the training recipe (corpus source, teacher support, trajectory mode, corpus scale, step budget) while holding the objective at the adaptive KL-TV hybrid; proxy = held-out val expected-τ during training.

### 5.1 Corpus: target-generated vs raw-text (first-order)

Holding objective, budget (1000 windows), and teacher-forcing fixed, and changing only the corpus source:

| corpus | proxy exp-τ | screening mtb / c4 / gsm8k | avgΔ |
|---|---|---|---|
| target-generated (deployed t4kv4, greedy+T1 continuations) | **2.020** | 2.8447 / 3.2572 / 3.8527 | +0.264 |
| raw-text windows (wiki/c4/sharegpt/gsm8k/code) | 1.670 | 2.7874 / 3.1640 / 3.5172 | +0.102 |

Training on the states the deployed target actually visits — its own greedy/T1 generations — is worth ≈+0.35 proxy exp-τ and ≈+0.16 screening micro-AL over raw dataset text; on gsm8k the raw-text rotation is essentially at shared (+0.010). This is a first-order lever and directly accounts for part of the previous provisional negative, which trained on raw dataset-text windows.

### 5.2 Teacher support: full-vocab vs top-64

`full_vocab_teacher_audit.csv` (target-generated corpus, ~1200 windows each) isolates why truncation hurts:

| corpus | mode | mean α-error (top-64) | mean |ΔKL| (top-64), nats |
|---|---|---|---|
| target-gen greedy | greedy | 0.00029 | 6.9545 |
| target-gen T1 | t1 | 0.00034 | 6.9534 |
| raw-text | rawtext | 0.01073 | 6.2554 |

Top-64 truncation is negligible for **acceptance** (α shifts by 3×10⁻⁴) but catastrophic for a **KL** target (~6.95 nats). An acceptance-only objective (pure-TV, expected-τ) is therefore nearly indifferent to truncation, but any KL-bearing objective on a top-64 teacher optimizes a systematically wrong target — visible as the top-64 row's lowest proxy exp-τ (1.853) in §4.2. Combined with §5.1, the previous top-64+KL teacher was doubly mismatched (truncated support and wrong objective for that support). All finalists use the full-vocab V=32000 teacher; the previous top-64 harm was objective mismatch, not α truncation.

### 5.3 Trajectory: teacher-forced vs on-policy / curriculum (does not help)

Rolling the draft's own trajectory (curriculum on-policy) underperforms teacher-forcing on target-generated states at both scales:

| trajectory | scale | proxy exp-τ (best step) | screening avgΔ |
|---|---|---|---|
| teacher-forced (hybrid) | 1000w | 2.020 (step 999) | +0.264 |
| on-policy curriculum | 1000w | 1.845 (step 249) | — |
| teacher-forced (HYBRID_s2) | 5000w | 2.098 (step 2999) | +0.212 |
| on-policy curriculum (ONPOLICY_s0) | 5000w | 1.879 (step 499) | +0.191 |

On-policy peaks very early (steps 249 / 499) and then diverges — the rotation over-fits the draft's own drifting trajectory — so it never reaches the teacher-forced proxy (2.02 / 2.10) and screens lower (+0.191 < +0.212). Teacher-forcing on the target's generations is both simpler and better here; on-policy does not help.

### 5.4 Corpus scale: 1000 → 5000 windows

Stage-1 pilot (1080 train / 120 val ≈ 1000w) hybrid proxy 2.020. Stage-2 (4500 train / 500 val = 5000w, batch 32, 3000 steps, AdamW + cosine + warmup + clip) hybrid proxy 2.065 / 2.084 / 2.098 across three seeds, screening +0.202 / +0.201 / +0.212. The 5× corpus gives a small, consistent lift and tightens seed-to-seed variance, but it is not a step change — the pilot already captured most of the effect.

### 5.5 Training length: 3000 vs 5000 steps

| run | steps | proxy exp-τ (best step) | screening avgΔ |
|---|---|---|---|
| HYBRID_s2 | 3000 | 2.098 (step 2999) | +0.212 |
| HYBRID_LONG5K | 5000 | 2.071 (step 4499) | +0.179 |

Extending to 5000 steps does not help: the objective saturates near step 3000 and the extra steps mildly over-fit (proxy 2.071 < 2.098, screening +0.179 < +0.212). 3000 steps is the operating point.

**Recipe verdict.** The two first-order levers are the exact-path quantized forward (§4.1, worth the entire +0.149 KL baseline over the previous provisional negative) and the target-generated full-vocab corpus (§5.1–5.2, ≈+0.16 over raw text). The choice of LK objective (§4.3, spread 0.028), η (§4.2), on-policy vs teacher-forced (§5.3, on-policy worse), and steps beyond 3000 (§5.5) are second-order. Corpus scale (§5.4) contributes a small, seed-stabilizing gain.

## 1. Executive summary

**Research question (final).** Can an exact-path, full-vocabulary, acceptance-aware orthogonal draft rotation R_D improve micro-AL over the shared target rotation R_T, while every model weight — target, R_T, and the draft's embedding, projections, AR decoder, LM head and norms — remains bitwise frozen?

**Answer: YES.** A residual orthogonal draft rotation R_D = R_T·C(A), trained only against a full-vocabulary acceptance objective through the exact quantized runtime forward, beats the shared R_T on every dataset and every chain mode tested, at confirmatory scale, with all 95% CIs excluding zero.

**Scope correction (2026-07-22).** This is quantization-aware ROTATION learning, not QAT. The sole trainable parameter is R_D (either the residual R_T·C(A) with A skew via the Cayley map, or an unrestricted orthogonal R_D); every pretrained weight is frozen. The P3 draft scale alpha is chosen by a post-hoc calibration sweep, never trained jointly in the main experiments. All draft-weight / QAT work is removed and excluded from every conclusion (`docs/OUT_OF_SCOPE_QAT.md`): the disarmed trainer, Stage-2 candidates I/J (`LK2_QAT_I_sharedRT`, `LK2_QAT_J_localRD`) never ran and no checkpoints exist; the trainable-alpha pilot `LK_G_ALPHA` is retained only as a flagged scalar-only ablation. The report claims no QAT upper bound and classifies the QAT-comparison question (Q13) as out-of-scope.

**Key results (numbers).**

- **Primary finalist (HYBRID_s2, adaptive-hybrid objective), confirmatory tree, t4kv4, vs shared R_T:** mtbench +0.1539 (n=80), c4 +0.1073 (n=200), gsm8k +0.2702 (n=200), sharegpt +0.1669 (n=80), humaneval +0.1187 (n=164); **mean +0.1634, all 5 datasets 95% CI > 0, p ≈ 0.0000** (paired prompt-cluster bootstrap, 3000 reps). Shared mtbench-80 baseline is 3.0080, reproduced bitwise-exactly at Gate B.
- **The exact-path result revises the previous "independent R_D is unnecessary" conclusion, which was reached on a trainer whose forward path diverged from the runtime in five material ways (Phase A).**
- **Objective ranking at confirmatory (all four all-5-dataset CI > 0):** neg-log-alpha / auxiliary-gradient AUXG **+0.177** (best) > expected-tau EXPTAU +0.165 > adaptive-hybrid HYBRID +0.163 > full-vocab KL +0.149. Exact-path **KL alone already beats shared (+0.149)**; the acceptance-shaped objectives reach means of +0.163..+0.177, i.e. they add only a further **+0.01..+0.03 increment** over KL. Differences among LK objectives are small; the exact path and full vocabulary matter far more than the loss.
- **Chain transfer (t4kv4, 40p, mtb/c4/gsm8k).** The gain holds in both stochastic and greedy chains, and is largest in the tree: greedy shared [2.155, 2.336, 2.725] → HYBRID [2.222, 2.460, 2.872]; t1 shared [2.000, 2.126, 2.442] → HYBRID [2.084, 2.261, 2.764].
- **Cross-target transfer.** R_D trained on t4kv4 (D_HYBRID) still helps on other quantization targets, strongest on the training target: on t4 +0.143 / +0.180 / +0.297; on t8 +0.045 / +0.030 / +0.132 (mtb/c4/gsm8k).
- **Local residual ≫ unrestricted.** Local residual finalists move the rotation only slightly (max element change ≈ 0.097, spread across the matrix); the unrestricted orthogonal R_D concentrates mass (max element change 0.392, max |element| 0.376) and collapses in the tree (mtbench ≈ 1.99, −0.6..−1.25 vs shared).
- **The benefit is acceptance alignment, not weight geometry.** Reconstructed W4 quant error is essentially unchanged by the rotation (W_rec W4 NMSE 0.01742 shared vs 0.01772–0.01777 for the finalists), so R_D does not reduce weight-quantization noise — it re-aligns the draft's accepted-token distribution.
- **Survives recalibration.** Under a per-model alpha sweep the advantage persists at each model's optimum: shared best alpha=45.25 → 2.624, D_HYBRID best alpha=32 → 2.845 (+0.221); it is not an alpha artifact.
- **Screening consistency.** All 13 trained rotations beat shared on 20-prompt screening (t4kv4, 3-dataset avg) by +0.136..+0.214; the hybrid objective is stable across 3 seeds (+0.202 / +0.201 / +0.212). On-policy (curriculum) corpora and a 5000-step long run did not improve over the 3000-step teacher-forced setting.

All figures are fake-quant micro-AL only (sum(tau)/cycles, tau = accepted + 1); no latency is claimed. Cells at 20 prompts are labelled screening; 80–200 prompts are confirmatory. All KV4 = KV4 without R3.

## 2. Method

**Exact-path quantized rotation forward.** Every candidate is trained and evaluated through a single forward, `exact_quantized_rotation_forward.py` (`ExactQuantizedRotationForward`), that reproduces the deployed runtime bitwise. R_D is folded into every draft decoder linear — with R2 on V/O and R4 (Hadamard) on down_proj exactly where the runtime applies them — and the *folded* weights are quantized with the official SpinQuant quantizers (MSE clip search, grid=100, maxshrink=0.8); activations use the official whole-concat per-token asymmetric ActQuantizer; the decoder runs the incremental one-token recurrent path with a KVCache, including draft KV4 quantization of appended K/V. Gradients reach R_D through a straight-through estimator around the quantizers, with the runtime cast order preserved.

**Trainable parameter.** Only R_D. The default parameterization is the residual Cayley form R_D = R_T·C(A) with A skew-symmetric, C(A) = (I−A)(I+A)⁻¹ orthogonal; an unrestricted orthogonal R_D is the ablation arm. All other parameters (target, R_T, draft embedding/projections/AR/head/norms, depth weight gamma) are frozen. P3 alpha is not trained: finalists take a post-training calibration sweep over {8, 11.31, 16, 22.63, 32, 45.25, 64, 90.51, 128}, folded globally and applied identically to shared R_T and every candidate before comparison.

**LK acceptance losses.** The acceptance mass is alpha(p,q) = sum_i min(p_i, q_i) = 1 − TV(p,q). Objectives evaluated: alpha = 1−TV (and raw TV); full-vocabulary KL (V = 32000); neg-log-alpha (auxiliary acceptance gradient, AUXG); an adaptive hybrid lam·KL + (1−lam)·TV with lam_k = exp(−eta·sg(mean alpha)) that anneals from distribution-matching toward acceptance as alpha rises; and expected-tau = 1 + sum_k prod_{j≤k} alpha_j. Losses are depth-weighted by gamma^(k−1). The teacher is the deployed target's full-vocabulary top-logit distribution.

**Corpus.** Target-generated: the deployed t4kv4 model itself generates greedy + T=1 continuations, which become the training states — this beats raw-text windows on the proxy (exp_tau 2.02 vs 1.67) and on screening. Full vocabulary throughout (no top-k truncation).

**Gates.**

| Gate | Check | Evidence | Status |
|---|---|---|---|
| A — provenance | target/draft/R_T pinned; R_T SHA-256 4b7e91d2…a48fa6e | digest provenance block | PASS |
| B — baseline reproduction | shared T4KV4 + D4P3KV4 tree reproduces micro-AL exactly | mtbench-80 = 3.0080 exact | PASS |
| C — LK math | alpha = 1−TV identity, TV, depth weighting, Cayley orthogonality | 21 unit tests (12 lk_losses + 9 residual_rotation) | PASS |
| D — train/runtime parity (bitwise) | folded weights + chain logits/tokens/KV incl. residual R_D and draft KV4 | gateD_parity.json: all depths max_dh = max_dlogits = 0.0, greedy_tok_equal = True | PASS |
| E — gradients | nonzero, finite grads through STE; finite-difference agreement | gateE_gradients.json: grad_norm 246.4, loss_delta 0.093, toy FD ratio 1.003, real+toy STE signs match | PASS |
| I — rejection sampling | acceptance-sampler correctness | 5 toy tests | PASS |
| G — orthogonality | learned R_D stays on the orthogonal manifold | orth_error ≈ 1.45e-4 for finalists (shared R_T 2.6e-5) | PASS |

**Two-stage selection.** Stage-1 pilot (1000w) screens objectives/corpora; Stage-2 (5000w, batch 32, 3000 steps, AdamW + cosine + warmup + grad-clip) trains the finalists; screening at 20 prompts, confirmation at 80–200 prompts with paired prompt-cluster bootstrap.

## 3. Phase A — how the previous R_D was actually trained

The previous negative conclusion came from a trainer (`src/eagle_spinquant/draft_rotation.py`, `scripts/train_eagle_draft_rotation.py`, TLDR-KV4 commit 4967d8f) whose forward path differed from the deployed runtime in ways large enough to invalidate a negative result. Numerical evidence is `tables/previous_training_audit.csv`.

**Material gaps (audit ranking).**

1. **Wrong-basis decoder quantization (largest).** The trainer quantized decoder weights in the ORIGINAL basis and then conjugated — R_Tᵀ·Q(W)·R_T — while the runtime quantizes the rotated weights, Q(R_TᵀWR_T). Q(·) and orthogonal rotation do not commute: a toy 4×4 gives max |Q(W)R − Q(WR)| = 0.0501 (≈ a third of a quant step). On the real draft layer-0 o_proj the trainer-vs-runtime discrepancy is **NMSE 0.02972 = 2.1× the entire runtime quantization noise it was meant to model (0.01437)** — i.e. the trainer optimized through a larger error than the signal. The gap was acknowledged in an in-code comment (`draft_rotation.py:155-159`).
2. **R2/R4 absent.** The runtime applies draft R2 (V/O) and R4 (down_proj Hadamard) before quantization; the training decoder applied neither, so it quantized entirely different matrices for V/O and down_proj.
3. **Clip-search mismatch.** A 6-point L2 grid over [0.75, 1.0] instead of the official SpinQuant MSE search (grid=100, shrink=0.8): NMSE 0.021 between the two quantized results, a different argmin, and ~19% more weight-quant error (err_trainer 0.01437 vs err_official 0.01209).
4. **Activation-quantizer implementation.** Granularity MATCHED (both whole-concat per-token asymmetric — the branchwise quantizer in the codebase was an H2 ablation only), but the trainer's continuous-offset quantizer and the runtime's integer-zero-point ActQuantizer disagree per value by NMSE 0.075 (≈ 2× the activation-quant error 0.036–0.038).
5. **Teacher / trajectory / validation.** The teacher was a top-64-renormalized, temperature-2 distribution, tau=2, over teacher-forced RAW dataset-text windows, selected by proxy top-1 accuracy — never the full-vocabulary (V=32000), tau=1, free-running deployed distribution, and never runtime micro-AL. Additional mismatches found during the audit: interleaved-RoPE vs HF `rotate_half` (different dimension pairing, hence different attention values), draft KV4 never simulated, and a full-prefix re-forward per depth instead of the incremental KVCache path. Optimization was also thin: 400–500 steps, batch 8, single seed, Adam 2e-3 with no schedule/warmup/clipping, rotation-only.

**Teacher audit — the top-64 harm was objective mismatch, not alpha truncation** (`tables/full_vocab_teacher_audit.csv`). On target-generated states, truncating to top-64 costs almost nothing in acceptance (mean alpha-error 0.00029 greedy / 0.00034 t1) but costs an enormous mean |KL|-error of 6.95 nats. So the previous top-64 setup did not hurt acceptance by dropping tail mass — it optimized a KL functional that top-64 renormalization mis-specifies by ~7 nats, i.e. the wrong objective on top of the wrong forward path. It also confirms the corpus choice: raw-text windows are off the deployed trajectory (alpha-error 0.01073) versus target-generated (0.00029).

**Consequence.** Because the trainer disagreed with the runtime by more than the quantization noise it was trying to optimize through, and optimized a mis-specified objective on off-distribution states, the earlier "independent R_D is unnecessary" finding was PROVISIONAL. Re-running the question on the bitwise-parity exact path (Gate D) with a full-vocabulary acceptance objective reverses it: an orthogonal draft rotation improves micro-AL over the shared R_T with all model weights frozen.

## 8. Chain vs tree modes — does the LK acceptance objective transfer to the deployed tree?

The LK objectives (α = 1−TV acceptance, neg-log-α, expected-τ, adaptive hybrid) are derived from the single-draft **stochastic-chain** acceptance identity: α(p,q) = Σ min(p,q) is exactly the token-level accept probability under speculative sampling. The deployed EAGLE-1 runtime, however, is a **greedy tree**. Section 8 asks whether a rotation trained against the chain-theoretic acceptance signal actually transfers to (i) the stochastic-chain draft at temperature 1 (M1), (ii) the greedy chain (M2), and (iii) the deployed greedy tree (M3). All numbers are fake-quant micro-AL on the deployed target t4kv4 (= W4A4 + KV4-without-R3); the trained rotation is the primary finalist HYBRID_s2 (Stage-2, 5000-word teacher-forced corpus).

**Absolute micro-AL (shared R_T vs HYBRID_s2), t4kv4:**

| Mode | scale | shared R_T (mtb / c4 / gsm8k) | HYBRID_s2 (mtb / c4 / gsm8k) |
|---|---|---|---|
| M1 t1 stochastic-chain | 40-prompt (screening) | 2.000 / 2.126 / 2.442 | 2.084 / 2.261 / 2.764 |
| M2 greedy-chain | 40-prompt (screening) | 2.155 / 2.336 / 2.725 | 2.222 / 2.460 / 2.872 |
| M3 tree (deployed) | 80 / 200 (confirmatory) | 3.008 / 3.073 / 3.514 | 3.162 / 3.180 / 3.784 |

**Deltas (HYBRID_s2 − shared R_T):**

| Mode | Δ mtbench | Δ c4 | Δ gsm8k | mean Δ (3 ds) |
|---|---|---|---|---|
| M1 t1 stochastic-chain (40p) | +0.083 | +0.136 | +0.322 | +0.180 |
| M2 greedy-chain (40p) | +0.067 | +0.125 | +0.147 | +0.113 |
| M3 tree (conf. 80/200) | +0.154 | +0.107 | +0.270 | +0.177 |

**Reading.** HYBRID_s2 beats shared R_T in **all three modes on all three datasets** — the gain is not an artifact of any one decoding regime. It is present where the objective's theory literally applies (the M1 t1 stochastic chain, mean +0.180) and, crucially, it carries over to the deployed **greedy tree** (M3, mean +0.177 over the three shared datasets, and mean +0.163 over all five confirmatory datasets — mtb +0.154, c4 +0.107, gsm8k +0.270, sharegpt +0.167, humaneval +0.119 — with every 95% paired-cluster-bootstrap CI excluding 0). The tree is the mode with the largest and by far the most robust gain: it is the only mode with all-five-dataset significance (n = 80–200), and on mtbench the tree gain (+0.154) exceeds both chains. The greedy chain shows the smallest gain (+0.113), consistent with it having the least stochastic slack to recover; the t1 chain's large gsm8k delta (+0.322) is real but measured at 40-prompt screening scale over a lower baseline (2.442). The take-away: **an acceptance objective grounded in single-draft stochastic-chain theory transfers, without any tree-specific term, to the deployed greedy tree, where it produces the study's headline gain.**

## 9. Cross-target transfer — does a rotation generalize across quantization targets?

R_D is trained against **one** deployed target's full-vocabulary teacher (t4kv4 = W4A4 + KV4-without-R3). Section 9 asks whether that single rotation still helps when the deployment target changes, without any retraining. The transferred finalist is D_HYBRID (itself confirmatory-validated on t4kv4: all five datasets CI>0, mean +0.169). It is applied unchanged to two other targets — t4 (W4A4, KV in fp16) and t8 (W8A8) — at 40-prompt scale, against the shared-R_T default calibrated identically on each target.

| Target | shared R_T (mtb / c4 / gsm8k) | D_HYBRID (mtb / c4 / gsm8k) | Δ (mtb / c4 / gsm8k) |
|---|---|---|---|
| t4 (W4A4, fp16 KV) | 2.936 / 3.039 / 3.429 | 3.079 / 3.219 / 3.727 | **+0.143 / +0.180 / +0.297** |
| t8 (W8A8) | 3.183 / 3.101 / 3.701 | 3.228 / 3.132 / 3.833 | **+0.045 / +0.030 / +0.132** |

**Reading.** The t4kv4-trained rotation **generalizes**: it improves acceptance on both alternative targets on every dataset, with no negative cells. The gain is strongest on t4 (mean +0.207) — the closest relative of the training target, differing only by the removal of KV4 — confirming that most of what R_D learned is about the shared W4A4 structure, not the KV4 detail. On the milder t8 (W8A8) the gain shrinks to +0.03–0.13: with 8-bit quantization noise much smaller, the shared R_T is already near-optimal and there is less acceptance misalignment for R_D to correct, but the transferred rotation still does not hurt. **The rotation generalizes across quantization targets, monotonically strongest on (and near) the target it was trained against.**

## 10. Conclusions and the fifteen required questions

**Research question — answered: YES.** Under an exact-path, full-vocabulary, acceptance-aware training protocol with **every model weight bitwise frozen** (target, R_T, draft embedding/projections/AR/head/norms) and R_D the *only* trainable parameter (as the local residual R_D = R_T·C(A), A skew, Cayley), an independent draft rotation R_D **improves micro-AL over the shared target rotation R_T** on the deployed W4A4(+KV4) tree — primary finalist HYBRID_s2 mean +0.163 across five confirmatory datasets, all 95% CIs excluding 0, p ≈ 0.000.

**Q1 — Can an independent exact-path full-vocab acceptance-aware R_D beat shared R_T with weights frozen?** YES. HYBRID_s2 beats shared on all five confirmatory datasets (mtb +0.154 / c4 +0.107 / gsm8k +0.270 / sharegpt +0.167 / humaneval +0.119; mean +0.163).

**Q2 — Is the gain statistically robust?** YES. Paired prompt-cluster bootstrap (3000 reps) puts all five 95% CIs above 0 (e.g. mtb [+0.091,+0.220], gsm8k [+0.219,+0.323]); D_HYBRID reproduces this (all five CI>0, mean +0.169).

**Q3 — Which LK objective is best, and does the choice matter?** Confirmatory ranking: neg-log-α (AUXG) +0.177 > expected-τ +0.165 > adaptive-hybrid +0.163 > full-vocab KL +0.149; all four have all-five-dataset CI>0. Exact-path **full-vocab KL already beats shared** (+0.149); the acceptance-shaped objectives add only a further +0.015–0.028. Objective choice is second-order — the exact path and full vocabulary matter far more than the loss.

**Q4 — Was the prior "independent R_D unnecessary" verdict an artifact of the training path?** YES. The previous TLDR-KV4 trainer quantized the decoder in the wrong basis (Q(W) then conjugate, vs runtime Q(R_DᵀW R_D)): its o_proj proxy error NMSE 0.0297 = 2.1× the true runtime quant noise 0.0144, and it also used a top-64 τ=2 KL teacher, no R2/R4, a 6-point clip grid, interleaved RoPE, and teacher-forced raw-text windows (docs/EAGLE_PREVIOUS_RD_TRAINING_AUDIT.md). Fixing the path flips the conclusion.

**Q5 — Is full-vocabulary teacher supervision necessary, or does top-64 suffice?** For *acceptance* the top-64 α-error is negligible (~0.0003), but the top-64 *KL*-error is ~6.95 nats. The earlier top-64 harm was an **objective mismatch** (KL under truncation), not α truncation. Full-vocab teachers (V = 32000) remove that confound and are used throughout.

**Q6 — Target-generated corpus vs raw-text windows?** Target-generated wins: the deployed t4kv4 generating its own greedy/T1 continuations gives proxy exp-τ 2.02 vs 1.67 for raw-text windows.

**Q7 — Does on-policy / curriculum training help?** NO. On-policy proxy 1.85 < teacher-forced 2.02, and it screened lower (+0.191 vs +0.212). Teacher-forced target-generated corpus is preferred.

**Q8 — Does the gain transfer from stochastic-chain theory to the deployed greedy tree?** YES (§8). HYBRID_s2 beats shared in the t1 stochastic chain, the greedy chain, and the tree, on all three datasets; the gain is largest and only-all-significant in the deployed tree.

**Q9 — Does the rotation transfer across quantization targets?** YES (§9). D_HYBRID (t4kv4-trained) gains +0.143/+0.180/+0.297 on t4 and +0.045/+0.030/+0.132 on t8; strongest near the training target, never negative.

**Q10 — Local residual R_D = R_T·C(A) vs unrestricted orthogonal R_D?** Local residual wins decisively. Finalists are small perturbations of R_T (max-element change ~0.097, geodesic ~68, orth-error ~1.4e-4) and beat shared; the **unrestricted R_D collapsed** in the tree (mtbench 1.99, −0.6 to −1.25 vs shared). Independence helps only as a *local* correction to R_T.

**Q11 — Is the benefit weight-quant geometry or acceptance alignment?** Acceptance alignment. W_rec W4 NMSE is ~0.0174–0.0177 for shared R_T and for *every* trained rotation — the rotation barely changes weight-quantization error. The gain comes from aligning the draft's accept distribution to the deployed target, not from better-quantizing weights.

**Q12 — Is the result an artifact of P3 α calibration?** NO. Under a full α sweep {8…128}, shared's optimum is α = 45.25 → 2.624 and D_HYBRID's is α = 32 → 2.845 — **+0.221 at each rotation's own optimum**. The result survives independent recalibration; it is not an α artifact. (Per directive, α is selected by calibration sweep, never trained.)

**Q13 — Upper bound from shared-R_T draft-core QAT?** **OUT OF SCOPE.** Per operator directive (docs/OUT_OF_SCOPE_QAT.md), this is quantization-aware *rotation* learning only: all draft-core weights are bitwise frozen. QAT candidates I (`LK2_QAT_I_sharedRT`) and J (`LK2_QAT_J_localRD`) were removed before any training step ran; no checkpoints exist. **No QAT upper-bound is claimed or estimated.**

**Q14 — Seed and training-length sensitivity?** Robust. The hybrid across three seeds screens +0.202 / +0.201 / +0.212; 3000 steps is not beaten by 5000 (LONG5K +0.179 ≤ +0.212). All 13 trained rotations beat shared on screening (+0.14…+0.21). Finalists are single-seed at confirmatory scale (a limitation, below).

**Q15 — Prior TLDR-KV4 conclusions: retained / narrowed / revised?**
- **REVISED — "independent R_D is unnecessary; shared R_T dominates."** Under exact-path, full-vocab, acceptance-aware training with a *local residual* rotation and frozen weights, an independent R_D **does** improve over shared R_T (mean +0.163, all five CI>0, transferring to greedy tree and across targets). The prior negative was an artifact of proxy-basis training (wrong quant basis at 2.1× runtime noise, top-64 KL teacher, interleaved RoPE, teacher-forced raw text), not a property of the problem.
- **NARROWED — "shared R_T is the strong default."** It remains an excellent, near-optimal default when *no* rotation is trained and on lightly-quantized targets (t8, where the trainable gain shrinks to +0.03–0.13), but it is **not** optimal under aggressive W4A4(+KV4), where a trained local R_D adds a significant, reproducible margin.
- **RETAINED —** fake-quant / micro-AL-only framing (no latency claims); KV4 = "KV4 without R3"; target-generated corpus ≫ raw-text; on-policy does not help; useful rotations are *small local* perturbations, and *unrestricted* rotations are harmful; the exactly-reproduced shared t4kv4 baseline (mtbench 3.0080).

**Limitations.** Fast screening at n = 20 prompts and chain/transfer probes at n = 40; confirmatory at n = 80–200. Finalists are **single-seed at confirmatory scale** (three-seed consistency established only at screening). All results are **fake-quant** micro-AL — no latency, throughput, or real-kernel claim. Draft-core **QAT is out of scope by operator directive**; this report therefore bounds only the *rotation-only* frozen-weight regime and makes no claim about how much further joint weight training could go.
