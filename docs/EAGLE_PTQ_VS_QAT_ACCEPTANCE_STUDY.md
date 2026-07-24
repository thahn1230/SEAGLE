# Training-Free PTQ versus Draft QAT for Quantized EAGLE-1

Run `runs/eagle_ptq_vs_qat_al_20260723_143129` · branch
`exp/eagle1-ptq-vs-qat-al-study` · 2026-07-23/24 · GPUs 0–5 (RTX 4090) ·
Llama-2-7b-chat (f5db02db) + yuhuili/EAGLE-llama2-chat-7B (44e37ec) ·
R_T `outputs/rotations/learned_chat_w4a4kv16/R.bin`
(sha256 4b7e91d2…a48fa6e) · metric: official micro Average Acceptance
Length tau = Σ(accepted+1)/cycles, MT-bench 80, greedy, tree mc_sim_7b_63,
paired prompt-cluster bootstrap 3000 reps.

## Executive summary

**Recommendation: use strict structural training-free PTQ (D4P3 with
calibrated α) as the deployment default and as the main paper
contribution; treat draft training strictly as an optional fine-tuning
add-on with a small LR — never with the original from-scratch recipe.**

- Under the pre-registered primary contract (final checkpoint of the
  faithful original EAGLE recipe, identical 6000-step budget), QAT is not
  merely unnecessary — it is **harmful**: tau drops by −0.44 (FP16
  target) and −0.65 (INT4 target) versus strict training-free PTQ, with
  95% CIs excluding zero. Both pre-registered non-inferiority criteria
  pass overwhelmingly in favor of training-free PTQ.
- The failure is the **recipe, not the concept**: the from-scratch LR
  (3e-5, warmup 2000) destroys the converged public draft
  (val top-1 0.797→0.61 trough), and even FP16 retraining then quantizes
  catastrophically (C9 = 1.61). Two labeled sensitivity analyses show the
  flip side: best-val early stopping (+0.23/+0.25 tau over PTQ) and a
  fine-tuning LR of 3e-6 (+0.24/+0.27, stable) both make QAT *modestly*
  better than strict PTQ.
- Even QAT's best case recovers only a small slice of what structure
  already recovered: P3 migration alone contributes +1.86/+1.78 tau over
  naive W4A4; the best trained variant adds at most +0.27 on top, and a
  frozen-weight local rotation R_D already delivers +0.15 of that
  (INT4), leaving ≤ +0.13 as the genuine weight-training margin.
- Speculative fidelity is not bitwise even in FP16 (90% exact-match,
  0.915 mean prefix agreement over 40 prompts): the tree-vs-sequential
  execution shape flips near-tie tokens. No configuration should be
  called "lossless" without this qualifier; all INT4 comparisons here
  share one verifier implementation and quantization contract.

## 1. Design

19 cells (C1–C14, N1–N5; N4≡C5, N5≡C12), two targets (FP16;
INT4 = learned-R_T SpinQuant W4A4 KV16, RTN+w_clip, no target QAT), one
draft INT4 contract **D4P3** (first/recurrent projections + AR linears
W4A4, P3 α-migration, embedding+LM head FP16, KV16, no silent FP16
fallback). Interface structures per spec §7: identity (FP16 target),
gamma_R1 shared basis (INT4 target), restored (INT4 target + stock FP16
draft). The runtime draft is internally R1-rotated in *every* mode
("identity" only removes the first-path input fold); QAT trains in
exactly the deployed basis and cast order (Gate D, bitwise).

**QAT** (audit `tables/qat_pipeline_audit.json`): original EAGLE
objective 1.0·SmoothL1(features) + 0.1·softCE(logits), AdamW 3e-5
β(0.9,0.95), clip_grad_value 0.5, linear warmup 2000→decay, input noise
(rand−.5)·0.2·512/L, max_len 2048, head+embedding frozen, all other
draft params trainable; STE around the *official* SpinQuant quantizers
at the deployed INT4 sites only; fused frozen-target teacher; budget
6000 steps (bs1×accum4 ≈ 2 epochs over 12,000 pinned ShareGPT convs,
sha-pinned, 62 convs excluded for eval-pool overlap — Gate H) identical
across every trained arm. Deviations listed in
`manifests/study_config.json`. Seeds: C3/C7 ×3; teachers: deployed FP16
or deployed INT4 target per arm (Gate G algebra: input = interface
feature, regression target = (h-basis)·R1).

**Alpha**: pre-registered 9-point grid on the disjoint c4 calib pool
(offset 500): winner **α\* = 45.254834 for both targets**
(fp16: 2.670 at α\*, vs 2.412 at 32; int4: 2.999 vs 2.939; extremes 8
and 128 collapse to 1.4–1.7). The same α\* is used for PTQ and QAT
deployment (§17 fairness). Legacy α=32 C5 kept as `C5a32sens`
(2.9973; α\* re-run 3.0327, ∆ +0.035, p=0.17 — n.s. but sign-consistent).

## 2. Primary result table (MT-bench 80, greedy; final-checkpoint contract)

| ID | Target | Draft | Wt-train | Rot-train | tau | seed σ | U_oracle |
|---|---|---|---|---|---:|---:|---:|
| C1 | FP16 | stock FP16 (Gate B: = 3.5762 known) | – | – | 3.5762 | | 0.534 |
| C2 | FP16 | strict TF PTQ D4P3 @α\* | – | – | 2.9065 | | 0.395 |
| C3 | FP16 | INT4 QAT (3 seeds) | QAT | – | 2.4688 | 0.038 | 0.305 |
| C4 | INT4 | stock FP16, restored interface | – | – | 3.2744 | | 0.472 |
| C5 | INT4 | strict TF PTQ D4P3 @α\* | – | – | 3.0327 | | 0.422 |
| C6 | INT4 | INT4-target-adapted FP16 retrain | FP16 | – | 3.0535 | | 0.426 |
| C7 | INT4 | INT4 QAT (3 seeds) | QAT | – | 2.3779 | 0.033 | 0.286 |
| C7b | INT4 | C6-initialized QAT | QAT | – | 2.0289 | | 0.213 |
| C8 | FP16 | FP16 retrained | FP16 | – | 2.9251 | | 0.399 |
| C9 | FP16 | C8 then D4P3 PTQ | FP16 | – | 1.6064 | | 0.126 |
| C10 | INT4 | C6 then D4P3 PTQ | FP16 | – | 2.6436 | | 0.341 |
| C11 | FP16 | rotation-optimized PTQ (LK R_D) | – | R_D | 2.7888 | | 0.371 |
| C12 | INT4 | rotation-optimized PTQ (LK R_D) | – | R_D | 3.1816 | | 0.453 |
| C13 | INT4 | FP16-teacher FP16 retrain @INT4 | FP16 | – | 2.5141 | | 0.314 |
| C14 | INT4 | FP16-teacher QAT @INT4 | QAT | – | 2.1179 | | 0.232 |
| N1 | FP16 | naive W4A4, no P3 | – | – | 1.0444 | | 0.009 |
| N2 | INT4 | naive W4A4, no P3 | – | – | 1.2521 | | 0.052 |
| N3 | FP16 | P2 branchwise scales | – | – | 1.2012 | | 0.042 |
| N3b | INT4 | P2 branchwise scales | – | – | 1.9818 | | 0.204 |
| N4 | INT4 | P3 shared R_T (≡C5) | – | – | 3.0327 | | 0.422 |
| N5 | INT4 | P3 local R_D (≡C12) | – | R_D | 3.1816 | | 0.453 |

Full CIs: `stats/bootstrap_pairs_mtbench.json`; per-cell CIs in
`tables/primary_table.csv`.

## 3. The four reference levels (§12)

| Level | FP16 target | INT4 target |
|---|---:|---:|
| Stock FP16 baseline (C1 / C4) | 3.5762 | 3.2744 |
| Empirical trainable "ceiling" (C8 / C6, this budget) | 2.9251 | 3.0535 |
| Oracle-draft ceiling (measured, same tree/verifier) | 5.8220 | 5.8190 |
| Policy-theoretical ceiling (L_max 5 + 1) | 6.0 | 6.0 |

C1 is **not** an upper bound of anything except the pretrained
checkpoint: stock FP16 reaches only U_oracle = 0.534 of the oracle-draft
ceiling. The "empirical trainable ceiling" at this budget sits *below*
the frozen stock draft — retraining subtracted capability (see §5).

## 4. Pre-registered criteria and decision logic

**Criterion A** (LCB of tau_PTQ − tau_QAT > −0.05): LCB = **+0.387**
(FP16) and **+0.589** (INT4) → **PASS** (strict PTQ is not merely
non-inferior; it is superior, p≈0, 3000-rep paired bootstrap with the
same prompt resample across seeds).
**Criterion B** (recovery_ratio ≥ 0.90 ∧ gap ≤ 0.10 ∧ no target-quality
regression): recovery_ratio = 1.31 / 1.58 (structural recovery *exceeds*
the QAT-recoverable gap because QAT ends below naive+structure), gap
negative, target untouched by the draft method → **PASS**.

Decision logic (§22): **Conclusion B — "training-free structural PTQ is
QAT-competitive"** (here: dominant) and **Conclusion E — "rotation-only
optimization closes the (negative) QAT gap"** hold. A (QAT necessary),
C (target adaptation matters more), D (quant exposure necessary — C3<C9
is false: 2.47 > 1.61, but C7 (2.38) < C10 (2.64) fails the conjunction),
F (small-but-significant positive gain) do not.

## 5. Why QAT lost under the faithful recipe — and the sensitivity flip

Validation trajectories (`logs/train_*.jsonl`): every trained arm
degrades as LR ramps (C8 top-1 0.797→0.614 by step 4000, partial recovery
to 0.711 as LR decays; C3 falls to 0.57–0.61 without recovery — quant
noise compounds the excursion). Training loss rises alongside val loss:
optimization instability, not overfitting. The original recipe was tuned
for 20-epoch from-scratch training; applied to the *converged* public
checkpoint at a 6000-step budget it destroys more than it adds. The most
extreme casualty is C9: FP16-retrained weights quantize to 1.61 tau
(vs 2.91 from the pretrained weights) — training moved the weights into
a quantization-hostile region.

Because QAT-at-step-0 *is* the strict-PTQ deployment (identical weights,
identical quantizers), the deployment-tau trajectory is exact:

  FP16 target: 2.907 (step 0 ≡ C2) → **3.13–3.15 (step 500)** → 2.47 (step 6000)
  INT4 target: 3.033 (step 0 ≡ C5) → **3.28–3.30 (step 500)** → 2.38 (step 6000)

**Sensitivity S1 — best-val checkpoint selection** (val-top3, every 500
steps): C3_best = 3.133/3.148/3.040 (σ 0.05), C7_best =
3.282/3.303/3.281 (σ 0.010). Paired deltas vs strict PTQ: **+0.226
[0.174, 0.275] (FP16), +0.249 [0.187, 0.316] (INT4)** — QAT wins under
this selection rule; Criteria A/B would *fail* in this variant.
**Sensitivity S2 — fine-tuning LR** (3e-6, warmup 200, 3000 steps, seed
0): val improves monotonically (0.705→0.717 FP16; 0.748→0.751 INT4);
deployment **C3lr = 3.1495 (+0.243 [0.199, 0.289]), C7lr = 3.3065
(+0.274 [0.211, 0.341])** — the same gain as best-val selection, stably,
and +0.125 [0.059, 0.195] above the frozen-weight rotation-optimized
C12.

Honest synthesis: *the answer to "is QAT necessary?" is no — and the
answer to "does simply training the EAGLE draft the normal way work?" is
also no.* What works is a small-LR fine-tune (or aggressive early
stopping), worth ≈ +0.24–0.27 tau over strict training-free PTQ, of
which +0.15 is already available with all weights frozen via the local
rotation R_D (INT4).

## 6. Structural decomposition (INT4 target)

naive W4A4 1.252 → P2 scales 1.982 (+0.73) → **P3 shared R_T 3.033
(+1.05)** → +local R_D 3.182 (+0.149 [0.088, 0.208]) → best trained
(C7lr) 3.307 (+0.125 over R_D). Structure does the heavy lifting;
training refines the last few percent. Under the FP16 target the LK R_D
*hurts* (C11 2.789 < C2 2.907, −0.118): that rotation was optimized for
the gamma_R1 interface geometry — rotation optimization is
interface-specific and must be re-run per deployment contract.

## 7. Teacher matching and staged initialization

- C6 (INT4-teacher FP16 retrain) 3.054 vs C13 (FP16-teacher, same
  deployment) 2.514: **teacher-target matching is worth +0.540
  [0.469, 0.609]**.
- C7 2.344 (s0) vs C14 (FP16-teacher QAT @INT4) 2.118: +0.227
  [0.183, 0.270] for the matched teacher.
- C7b (C6-initialized QAT) 2.029, −0.316 vs direct C7: staged
  adaptation *compounds* the recipe damage; do not chain trainings under
  this recipe.
- But note C6 < C4 (3.054 vs 3.274): at this budget even matched-teacher
  retraining lost to the frozen stock draft behind a restored interface
  (best-val C6 3.414 > C4 — the same selection story).

## 8. Ceilings, target quality, speculative fidelity

Oracle-draft ceilings (perfect draft, same tree/verifier/EOS/length,
measured on real sequential-greedy continuations): 5.822 (FP16) / 5.819
(INT4); policy-theoretical max 6.0 (deepest chain 5 + verifier token).
The best systems reach U_oracle ≈ 0.45–0.53 — half the tree policy's
budget is unexploited even at FP16; quantization losses (0.395–0.422)
are of the same order as the FP16 headroom itself.

Target quality (provenance: TLDR-KV4 incremental protocol, identical
build): WikiText-2 PPL 5.985 (FP16) vs 6.033 (W4A4KV16), CE +0.0079
nats. Yet greedy *token-level* agreement between the INT4 and FP16
targets is only 0.038 mean prefix agreement — near-tie logit flips
diverge trajectories within ~5 tokens. PPL-style metrics dramatically
understate functional divergence; acceptance must be measured against
the *deployed* target (which this study does throughout).

Speculative fidelity (§16, Gate L; sequential vs EAGLE-tree under the
SAME target, 40 prompts, greedy): FP16 exact-match 0.90, mean prefix
agreement 0.915 — *not* bitwise lossless (execution-shape fp16 rounding
flips near-ties). INT4: `tables/speculative_fidelity_int4.json`
(dynamic per-token A4 is execution-shape dependent; reported number in
the table file; same verifier for all INT4 arms so comparisons are
internally consistent). No configuration is described as lossless in
this study.

## 9. Cost accounting

PTQ: α calibration = 10 evals × ~10 min ≈ 1.7 GPU-h (once per
deployment contract); rotation-learning (C11/C12, separate line): ~3
GPU-h (LK study artifact, reused). QAT: 3.1–7.1 GPU-h per seed × 9
runs ≈ 47 GPU-h (+ 2 probes ≈ 8 GPU-h), peak 21–22 GiB, ~30–35 M tokens
per run; details `tables/preparation_cost.json`. Deployment memory
`tables/model_memory.json` (D4P3 draft ≈ 0.30× of FP16 draft bytes on
quantized linears; embedding/head FP16 dominate the draft's residual
footprint). The +0.24–0.27 tau of tuned QAT costs ~5 GPU-h/seed plus
teacher access; the +1.78 of P3 structure costs a calibration sweep.

## 10. Answers to the 14 report questions

1. Stock FP16–FP16 an upper bound? **No** — it is a pretrained
   checkpoint at U_oracle 0.53; best-val/low-LR arms show trained
   drafts can beat *quantized* baselines, and the oracle sits at 5.82.
2. Policy-theoretical ceiling: **6.0** (tree depth 5 + 1).
3. Oracle-draft ceiling: **5.822 / 5.819** (measured).
4. Stock FP16 reaches **53.4%** of the oracle ceiling.
5. FP16 retraining at this budget/recipe: **−0.65** (C8 vs C1); with
   best-val selection −0.13; genuinely positive only vs quantized cells.
6. Strict P3 PTQ recovers **+1.86 / +1.78** over naive W4A4 — 89–96% of
   the distance from naive to the *best observed* system.
7. Rotation-only optimization: **+0.149** (INT4; CI>0), **−0.118**
   (FP16 — interface-mismatched rotation; must be retrained per
   interface).
8. Full QAT: **−0.44/−0.65** (faithful recipe, final ckpt);
   **+0.23/+0.27** (best-val or small-LR).
9. QAT vs retrain-then-PTQ: QAT ≫ C9 under FP16 (2.47 vs 1.61) but
   C7 < C10 under INT4 (2.38 vs 2.64) — no consistent quant-exposure
   advantage under the faithful recipe (Conclusion D not supported).
10. INT4 target requires target-conditioned retraining? **Yes if you
    train at all**: matched teacher +0.54 (FP16 draft) / +0.23 (QAT).
11. Training-free PTQ statistically non-inferior to QAT? **Yes**
    (pre-registered primary; Criteria A and B pass). Under best-val /
    small-LR sensitivity selections the ordering reverses (reported).
12. Does QAT's gain justify its cost? Only the tuned variant, and only
    if ~5 GPU-h/seed + ShareGPT/teacher access is cheap for the
    deployment; the answer is "no" for the faithful recipe.
13. Main paper contribution: **the structural training-free method**
    (P3 migration + rotation-consistent interface + calibrated α),
    with the QAT trajectory result as the cautionary counterpoint.
14. Under speculative-fidelity constraints: all conclusions above
    compare like-for-like under one verifier; nothing is claimed as
    lossless; FP16-vs-INT4 target function divergence (0.038 token
    agreement) means acceptance-vs-deployed-target is the only valid
    frame — which all cells here use.

## 11. Gates

A ✓ (provenance) · B ✓ (C1 = 3.5762 exact) · C ✓ (validated P3 code) ·
**D ✓ bitwise** (fresh + trained-ckpt reload, both structural modes, at
non-power-of-2 α; two 1-ULP-class parity bugs found and fixed: CPU-vs-GPU
fp16 α-division rounding, STE reconstruction re-rounding) · E ✓
(objective preserved) · F ✓ (trainable scopes audited + tested) · G ✓
(teacher transforms, algebra-tested) · H ✓ (pinned splits, 62 exclusions,
calib offset-500) · I ✓ (3+3 seeds complete; C7_s2 re-run after an OOM
induced by a foreign process during final validation) · J ✓ (oracle toy
self-tests) · K ✓ (official aggregation tested) · L ✓ (fidelity audit
run; nothing labeled lossless). 17 study tests + 85 pre-existing pass.

## 12. Artifacts

`tables/` primary_table.csv, qat_pipeline_audit.json,
gateD_qat_parity(.json/_trained.json), target_quality.json,
speculative_fidelity_*.json, acceptance_by_depth.json,
seed_variance.json, dataset_tau.csv, preparation_cost.json,
model_memory.json · `stats/` bootstrap_pairs_mtbench.json,
verdicts.json · `plots/` 14 figures (PNG+PDF+CSV) · `oracle/`,
`shards/` (69 shard CSVs incl. cross-domain), `ckpts/` (22),
`manifests/` (study_config incl. deviations, scheduler state, data
manifest) · bundle `eagle_ptq_vs_qat_al_20260723_143129.tar.gz` (+
`_latest`).
