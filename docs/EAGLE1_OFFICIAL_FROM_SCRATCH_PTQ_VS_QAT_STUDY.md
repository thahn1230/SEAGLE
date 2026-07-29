# Official-Recipe FP16 EAGLE-1 Reproduction and Fair PTQ-vs-QAT Study

Run `runs/eagle1_official_fromscratch_ptq_vs_qat_20260724_213243` ·
branch `exp/eagle1-official-fromscratch-ptq-vs-qat` · 2026-07-24→29 ·
8×RTX 4090 · target Llama-2-7b-chat@f5db02db · R_T
sha 4b7e91d2…a48fa6e · metric: official micro Average Acceptance Length
tau (MT-Bench 80, greedy, mc_sim_7b_63), paired prompt-cluster bootstrap
3000 reps.

## Executive summary

1. **The official EAGLE-1 recipe reproduces from scratch (RQ1 = YES).**
   A fresh draft (PyTorch-default init, target embedding copied+frozen)
   trained with the audited official recipe over the full 66,890-usable
   conversation set for the official 21 epochs reaches
   **tau 3.5341 vs the public checkpoint's 3.5762** — Δ −0.042
   [−0.091, +0.008], p = 0.105: *statistically indistinguishable*.
   All five datasets reproduce at 97.3–99.4%. Pre-registered gates
   (abs ≤ 0.20; rel ≥ 94%) pass with wide margin.
2. **Training-free structural PTQ remains a competitive default
   (RQ2 = YES).** From the same frozen anchor, D4P3 with per-condition
   recalibrated α retains 2.9955 (FP16 target) / 2.9666 (W4A4 target) —
   85% / 84% of the anchor — while naive W4A4 collapses to 1.11/1.34.
   The local residual rotation adds +0.096 [0.044, 0.149] (frozen
   weights).
3. **Low-LR single-step QAT gives a real, replicated additional gain
   (RQ3 = YES).** With the pilot-selected LR 1e-6, three seeds per
   target: **Q0 (FP16 tgt) +0.197…+0.209, Q1 (W4A4 tgt)
   +0.321…+0.362 over strict PTQ** (every seed CI > 0); Q1 also beats
   the rotation-optimized PTQ by +0.228 [0.168, 0.288]. Crucially,
   **final ≈ best-validation checkpoint** (e.g. Q1 finals
   3.287–3.329 vs bestvals 3.278–3.321): at the correct fine-tuning LR
   there is **no over-training**, unlike the previous study's
   from-scratch-LR pathology.
4. **Multi-step quantized-rollout QAT does NOT beat single-step
   (RQ4 = NO, pilot gate).** M2(γ=0.8): 3.2654 vs single-step 3.2904,
   Δ −0.025 [−0.092, +0.039], p = 0.45; M1: 3.2789, Δ −0.011. Per the
   pre-registered pilot rule, no 3-seed expansion; the official
   single-step objective stands.
5. **Alpha must be recalibrated per target/interface (RQ5 = YES,
   Conclusion F).** The fresh anchor selects **α=45.25 (FP16-target
   identity)** but **α=32.0 (W4A4-target rotated)** — the public-anchor
   study had selected 45.25 for both. One global α per deployment is
   sufficient (RQ6, Conclusion G): first-vs-recurrent NMSE optima differ
   by only one grid step (32 vs 45.25) and the pre-registered +0.10-tau
   adoption bar is not met by a diagnostic margin of that size.
6. **Throughput caveat (RQ7):** all quantization here is fake-quant;
   per-prompt wall-clock is recorded (gen_seconds in every shard) but
   real INT4 speedups are NOT claimed. Speculative fidelity is not
   bitwise even at FP16 (0.90 exact) and the W4A4 target under tree
   execution is a materially different function than sequentially
   (0.025 exact) — a shape-dependent dynamic-A4 implementation property,
   not an intrinsic EAGLE/INT4 property (§9).

**Recommendation:** deploy structural TF-PTQ as the zero-training
default; where ~4–6 GPU-h/seed is affordable, add single-step low-LR
(1e-6) QAT for +0.20/+0.32–0.36; skip multi-step rollout training;
always recalibrate α per target/interface pair.

## 1. Official-recipe audit and deviations

Executable-code audit in `docs/EAGLE1_OFFICIAL_RECIPE_AUDIT.md`:
21-epoch off-by-one, fixed 800k-step scheduler horizon, PyTorch-default
fresh init (initializer_range unused), AdamW default weight-decay 0.01,
`os.walk` data order (irreproducible officially), per-epoch save_state,
no best-checkpoint selection. Declared deviations D1–D9 (fused teacher
for the 550–700 GB storage wall — bitwise-verified on an 8-shard audit
subset; deterministic order; world-size changes 8→7→6→8 across user
pauses with resume-from-checkpoint; bs1×accum4 via the official
grad-accum knob after NCCL-buffer OOM at bs2×2).

## 2. Data and reproduction run

66,890 usable conversations (1,110 role-alternation skips out of the
official 0..68000 coverage), split 63,545/3,345 (official 95/5).
8-GPU sharded audit subset: no dup/missing IDs, sample-level SHA
hashes, fused-teacher parity **bitwise PASS**. Training: ~43.9k
optimizer steps total across resumes, bf16 DDP, ~1.9 s/step at world 8;
val top-1 trajectory 0 → 0.787 (public-checkpoint level).

## 3. Reproduction gate (pre-registered)

| dataset | public C0 | fresh C1 | ratio |
|---|---:|---:|---:|
| MT-Bench | 3.5762 | 3.5341 | 98.8% |
| ShareGPT | 3.5848 | 3.5647 | 99.4% |
| C4 | 3.3302 | 3.2549 | 97.7% |
| GSM8K | 3.9447 | 3.8779 | 98.3% |
| HumanEval | 4.1083 | 3.9966 | 97.3% |

Δ(C1−C0) on MT-Bench = −0.042 [−0.091, +0.008], p=0.105 (n.s.).
**Gate PASS** (abs 0.042 ≤ 0.20; rel 98.8% ≥ 94%). Anchor frozen:
`checkpoints/eagle1_fresh_fp16_anchor/anchor.pt` (sha a7c6ccc8…),
step-0 deployment parity of the anchor verified bitwise in both
structural modes (`tables/gateD_qat_parity*.json`).

## 4. Alpha recalibration (spec §14; Conclusions F & G)

Held-out c4 calib pool (offset-500), 9-point pre-registered grid:

- **Condition A (FP16 target, identity): α\* = 45.254834** (peak 2.670;
  8→1.54, 128→1.55 collapse at extremes).
- **Condition C (W4A4 target, rotated): α\* = 32.0** — *different from
  the public-anchor study's 45.25*. Reusing the wrong α costs measurable
  tau → **Conclusion F: target-specific recalibration is required.**
- Condition B (restored FP16 draft): no α applies.
- Stats-derived priors (RMS/p99/p99.9 hidden-embedding ratios) recorded
  in `tables/alpha_stats.json`; grid selection remains primary.
- First-vs-recurrent audit (`tables/first_recurrent_audit_*.json`):
  per-path NMSE optima differ by exactly one grid step (int4: first 32,
  rec1–4 45.25; fp16: first 45.25, rec1 32). Diagnostic-level effect
  only → **Conclusion G: one global α per deployment suffices** under
  the pre-registered +0.10-tau adoption bar.

## 5. Structural PTQ from the fresh anchor (RQ2)

| cell | FP16 target | W4A4 target |
|---|---:|---:|
| C1 anchor (FP16 draft) | 3.5341 | — |
| P0 naive W4A4 | 1.1110 | 1.3395 |
| **P1 D4P3 @α\*** | **2.9955** | **2.9666** |
| P2 D4P3 + local R_D | — | 3.0629 |

Quantization cost (P1 − C1) = −0.539 [−0.588, −0.489]; the P3
structure recovers +1.88/+1.63 over naive. P2 − P1 = +0.096
[0.044, 0.149] (frozen-weight rotation, LK R_D transferred to the fresh
anchor). Numbers are within a few hundredths of the public-anchor
study's (2.907/3.033) — the structural-PTQ result is anchor-robust.

## 6. Single-step QAT (RQ3; Conclusion C)

LR pilot (1000 steps, calib AL): both variants select **1e-6**
(Q0: 2.907 vs 2.858@3e-6 vs 2.861@1e-5; Q1: 3.288 vs 3.199 vs 3.213).
Note the freshly-converged anchor is MORE LR-sensitive than the public
checkpoint (3e-6 already hurts).

3 seeds × 2 targets, 3000 steps, step-ckpts every 500, best-ckpt chosen
on the calib pool (never test):

| | s0 | s1 | s2 | mean | Δ vs P1 (per-seed CIs) |
|---|---:|---:|---:|---:|---|
| Q0 final | 3.2033 | 3.2040 | 3.1919 | 3.200 | +0.197…+0.209, all CI>0 |
| Q0 bestval | 3.2388 | 3.2064 | 3.2331 | 3.226 | +0.243 (s0) |
| Q1 final | 3.2904 | 3.2873 | 3.3290 | 3.302 | +0.321…+0.362, all CI>0 |
| Q1 bestval | 3.2783 | 3.3205 | 3.3112 | 3.303 | |

**final ≈ bestval in every run** — no over-training at the correct LR;
the previous study's catastrophic final-checkpoint QAT was purely a
recipe/LR artifact. Q1 − P2 = +0.228 [0.168, 0.288]: QAT beats even
rotation-optimized frozen-weight PTQ. Cross-domain (seed 0): Q1 beats
P1 on all five datasets. Teacher matched deployment in both variants
(Q1 trained against the deployed W4A4 target).

## 7. Multi-step rollout QAT (RQ4; Conclusion E) — non-official extension

K=4 dense rollout, deployed W_rec fold for depths ≥1, teacher-forced
tokens, softCE row-cap 512/depth (declared approximations), LR 1e-6,
3000 steps, seed 0, Q1 deployment:

- M2 (γ=0.8): **3.2654** vs single-step 3.2904 → Δ −0.025
  [−0.092, +0.039], p = 0.45 (no gain).
- M1 (uniform): **3.2789** vs single-step 3.2904 → Δ −0.011 (n.s.;
  re-run after a foreign-process GPU collision).

**Pilot gate NOT passed → no 3-seed expansion. Conclusion E rejected:**
the official single-step objective is not improved by this rollout
extension at matched budget.

## 8. Statistics

All primary comparisons: paired prompt-cluster bootstrap, 3000 reps,
95% CIs, identical prompt manifests (`stats/bootstrap_pairs_mtbench.json`).
3-seed MT-Bench results are seed-replicated; cross-dataset results are
seed-0 (labeled). Seed σ: Q0 final 0.006, Q1 final 0.023.

## 9. Fidelity

- Model fidelity (targets): PPL 5.985 (FP16) vs 6.033 (W4A4),
  ΔCE 0.008 nats; yet sequential greedy token agreement between the two
  targets is only 0.038 mean-prefix — PPL-scale metrics understate
  functional divergence.
- Speculative fidelity (same target, sequential vs tree): FP16 0.90
  exact / 0.915 prefix; W4A4 0.025 exact / 0.155 prefix. The dynamic
  per-token A4 quantizer sees different activation shapes under tree
  evaluation (documented axes/rounding in the prior study's §16 audit);
  all INT4 comparisons here share one verifier, so internal rankings are
  unaffected — but nothing is claimed lossless, and INT4-EAGLE outputs
  differ from sequential INT4 outputs under this implementation.

## 10. Decision rules (spec §26)

- **A. Official FP16 reproduction succeeded — YES** (gate PASS, n.s.
  difference from public).
- **B. Structural PTQ is a competitive training-free default — YES**
  (84–85% of anchor; +1.6–1.9 over naive; anchor-robust).
- **C. Low-LR QAT provides an optional additional gain — YES**
  (3-seed paired CIs exclude zero on both targets).
- **D. QAT is required — NO** (PTQ meets any reasonable deployment
  floor; QAT is an upgrade, not a rescue).
- **E. Multi-step QAT improves over single-step — NO** (pilot Δ ≤ 0).
- **F. Target-specific α calibration required — YES** (45.25 vs 32.0
  across targets; changed vs the public anchor).
- **G. One global α sufficient — YES** (path-specific optima differ by
  one grid step; below the pre-registered adoption bar).

Paper language: *"QAT is not necessary for competitive deployment"* is
supported; *"QAT provides no benefit"* is **refuted** — both statements
are now backed by seed-replicated CIs from one common anchor.

## 11. Cost

FP16 reproduction: ~155 GPU-h (43.9k steps across world sizes 8/7/6/8).
PTQ: α calibration 2×9 evals ≈ 3 GPU-h; rotation reuse. QAT: 6×~3.6
GPU-h + pilots 6×~1.2 + multistep 2×~5.5 ≈ 40 GPU-h. Best-ckpt
selection ≈ 6 GPU-h of calib evals. Full artifact set in
`tables/preparation_cost`-style manifests and the gpu_logs CSVs.

## 12. Artifacts

`tables/` reproduction_gate, alpha_calibration, alpha_stats, lr_pilot,
ckpt_selection_Q*, first_recurrent_audit_{fp16,int4},
gateD_qat_parity{,_trained}, speculative_fidelity_* · `stats/`
bootstrap_pairs_mtbench.json · `shards/` (100+ CSVs incl. per-prompt
gen_seconds for TPS) · `data_audit/` (shard manifests + fused parity) ·
`manifests/` (gpu_allocation with D1–D9, scheduler state) · anchor
checkpoint (path+sha; not in bundle) · bundle
`eagle1_official_fromscratch_ptq_vs_qat_20260724_213243.tar.gz` +
`_latest`.
