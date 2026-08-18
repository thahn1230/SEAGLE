# Strict SEAGLE-RT: From-Scratch Native Rotated-Interface Training

Run `runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552` · branch
`exp/eagle1-strict-seagle-rt-native-w4a4-target` (base b2eb71e) ·
2026-08-16 → 08-18 · 8×RTX 4090 · target Llama-2-7b-chat@f5db02db
W4A4 SpinQuant (R.bin 4b7e91d2) · metric: official cycle-pooled
micro-tau, mc_sim_7b_63, greedy, max-new 128; mean4 = arithmetic mean
of {MT-Bench:80, GSM8K:200, ShareGPT:80, HumanEval:164}; manifests
sha-identical to canonical (Gate F PASS).

**SEAGLE-RT (strict)** = EAGLE-1 draft, FRESH official init, trained
the FULL official schedule (21 epochs, 41,685 steps, TC constants
asserted identical to the validated reproduction) while consuming the
DEPLOYED W4A4 target's NATIVE rotated feature a_t = (h/γ_f)@R1_T as
BOTH input and regression label; soft-CE scored by the deployed fused
head W_lm·D_γ·R1 (sha16 0335833c). No restore op anywhere (runtime
counters 0; AST-static scan clean; contracts in tables/). Selected
checkpoint = FINAL (preregistered; sha256 ca0e6533…). The previous
"SEAGLE-RT" (anchor a7c6ccc8 + D4P3 deploy) is renamed
**Original-interface scratch reference** and serves as the P1 control.

## Executive verdict — CASE D (strongest SEAGLE support), with a
## sharpened mechanism

1. **Stage A — basis adaptation: SOLVED, and then some.** SRT_FP16
   mean4 **3.9018** vs original-interface control 3.5141 (+0.388,
   all 4 datasets; P1). Native rotated-interface training is not a
   handicap — it is the best FP16-draft acceptance ever measured on
   this W4A4 target (beats SEAGLE-QAT 3.6108 and PTQ 3.4788), because
   the teacher matches the deployed target exactly.
2. **Stage B — low-bit robustness: NOT solved.** W8A8-RTN retains
   96.9% (3.7829). **W4A4-RTN collapses to 1.2540 (32.1%)** — inside
   the historical naive-collapse band (1.05–1.34). *Even full native
   from-scratch retraining does not remove the W4A4 projection
   bottleneck* (§33 claim now demonstrated under the strict contract).
3. **Rotation does NOT rescue.** Fixed Hadamard 1.2881 (33.0%);
   learned Cayley R_D (3 seeds, flat training loss) best 1.3819
   (35.4%). NOT Case C.
4. **The canonical SEAGLE correction DOES rescue.** Validation-only
   selection over the rescue grid picked α=32 + learned R_D (calib
   3.157 > α-alone 2.990 > α+HAD 2.931, each grid's own argmax —
   AH peaks at α=64; AH@α32 = 2.784): final **3.4487 (88.4%)** —
   +2.19 mean4 over RTN (P7 mean4 +2.07; MT-Bench +1.87 [1.79,1.96]; 4/4 Holm), statistically at
   the SEAGLE-PTQ level (3.4788). α alone restores 3.2454 (83.2%);
   the rotation helps ONLY on top of α. Transparency note: α-alone
   finals ran before the α+SQ validation comparison completed; the
   canonical arm follows the validation argmax only. **CASE D:
   SEAGLE's projection correction remains necessary and effective
   even after true native retraining.**
5. **Mechanism shift (new).** The native-basis fc input is
   OUTLIER-FREE (kurtosis 3.0-3.1, absmax 5.1, top-0.1% channel
   energy 0.31-0.43% — R1 already mixed the outliers away) but has a
   **59.5× e/h branch RMS imbalance** (0.0168 vs 1.0). Single-site
   audit (matched c4-calib pool): fc ALONE collapses to 1.2540 vs the
   full-W4A4 draft's 1.2067 — essentially the entire collapse from one
   site; every other site ≥3.24. A/W decomposition (fc): A-only tau 1.526 / W-only
   2.431; NMSE A 0.082 / W 0.761 / A+W 0.830; α=32 cuts A-NMSE to
   0.020. Native retraining *transforms* the projection pathology
   from outlier geometry into branch-scale imbalance — which is why
   rotations (norm-preserving) fail and α-balancing succeeds.
6. **Compute-quality (§31).** Strict RT core = **267.9 GPU-h
   measured** (trainer-t0 wall incl. ~3 min setup, ×8 GPUs; ISO-stamp
   derivation 267.4) + 4.64 cache generation + 16.4 evaluation +
   ~10 bench/gate/debug (rotation training 0.5, separate; study total
   ≈307 GPU-h incl. estimates) ≈ 33× SEAGLE-QAT's core (8.0) and 89× PTQ's (3.0) —
   yet its best W4A4 deployment (rescued 3.4487) still loses to
   SEAGLE-QAT (3.6108) and only matches PTQ (3.4788) built on the cheap
   original-interface anchor. Native RT is the right spend only when
   the draft can stay FP16/W8A8; for W4A4 drafts, post-training
   SEAGLE adaptation dominates on both cost and quality.

## Final 4-dataset table (single strict checkpoint, W4A4 SQ target)

| Method | Draft prec | Rotation | MT | GSM8K | ShareGPT | HumanEval | mean4 | ret. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original-interface control | FP16 (restored) | — | 3.2795 | 3.6586 | 3.2718 | 3.8466 | 3.5141 | — |
| **SEAGLE-RT-FP16** | FP16 | — | 3.6626 | 4.0225 | 3.7378 | 4.1842 | **3.9018** | 100% |
| SEAGLE-RT-W8A8-RTN | W8A8 | none | 3.5372 | 3.9151 | 3.6035 | 4.0759 | 3.7829 | 96.9% |
| SEAGLE-RT-W4A4-RTN | W4A4 | none | 1.2249 | 1.2683 | 1.2136 | 1.3092 | 1.2540 | 32.1% |
| SEAGLE-RT-W4A4-HAD | W4A4 | fixed | 1.2712 | 1.3158 | 1.2558 | 1.3097 | 1.2881 | 33.0% |
| SEAGLE-RT-W8A8-SQ | W8A8 | learned | 3.5009 | 3.8763 | 3.6370 | 4.0314 | 3.7614 | 96.4% |
| SEAGLE-RT-W4A4-SQ | W4A4 | learned | 1.3478 | 1.3902 | 1.3423 | 1.4473 | 1.3819 | 35.4% |
| SEAGLE-RT + rescue (α=32 only, diagnostic) | W4A4 | α | 3.0548 | 3.3484 | 3.0968 | 3.4817 | 3.2454 | 83.2% |
| **SEAGLE-RT + SEAGLE-PTQ rescue (α=32 + R_D, canonical)** | W4A4 | canonical | 3.2197 | 3.5478 | 3.2687 | 3.7587 | **3.4487** | 88.4% |

P3 (W8A8 RTN vs SQ): 0/4 Holm-sig — rotation gives no gain at 8-bit
(3.7614 vs 3.7829; spec §18's caution confirmed).

## Contracts, gates, provenance (all PASS)

Gates A-O of the spec: fresh init (official, seed 0, embedding sha
054c038d); fixed target (one build for every arm); native tap
(modeling_llama_kv.py:1074, RMS 1.0 measured); no-restore (runtime
counters 0 + static scan); basis-consistent labels (contract CSV all
MATCH); full official schedule (41,685 steps / 21 epochs, world 8);
no test leakage (c4-calib-only selection; FINAL-ckpt rule); single
checkpoint for all arms (sha ca0e6533); pure RTN baselines
(no α/GS/LS/D4P3/R2/R4/R5 — by construction in native_raw_draft.py);
canonical manifests (sha-identical); GPU-hours from timestamps;
rotation cost separated. Teacher cache: 495 GiB, 65.3% tokens,
BIT-EXACT vs online (128/128), one-step loss/grad diff exactly 0.0;
hybrid online remainder through the same deployed build.

Speed engineering (user amendment): offline cache + hybrid teacher,
single-visibility DDP (NCCL peer-buffer fix), validated-semantics
ckpt-on mode 2.802→2.89 s/step realized, training wall 33.48 h vs
32.5 h predicted; post-training stages fully parallel (autopilot).

## Statistics
Preregistered P1-P7, paired prompt-cluster bootstrap 10k reps, Holm
across the 4 datasets per family: stats/bootstrap_pairs_*.json,
stats/holm.json — Holm-sig counts: P1 4/4, P2 4/4, P4 4/4, P7 4/4;
P6 4/4 (small), P5 3/4 (tiny), P3 0/4 n.s. Effect sizes: mean4 deltas
P1 +0.39 / P2 −0.12 / P4 −2.65 / P7 +2.07 (per-dataset values incl.
the MT-Bench P7 +1.87 [1.79,1.96] in tables/bootstrap.csv, holm.csv;
bootstrap p floored at 1e-4). RCAL replay
(mtbench): tables/rcal.csv — AL_q/AL_0/RCAL/SAL/LAL/AFS per headline
arm; no deceptive-AL flag on any claimed improvement.

## Decision-rule statement (§32/§33)
- "Even native from-scratch draft retraining does not remove the
  W4A4 projection bottleneck." — **SUPPORTED (this strict study).**
- "SEAGLE achieves comparable quality at much lower compute." —
  **SUPPORTED with precision**: at W4A4 deployment, SEAGLE-QAT
  (8 GPU-h core) exceeds strict-RT+rescue (267.9 GPU-h core) by
  +0.16 mean4 and SEAGLE-PTQ (3 GPU-h) matches it (+0.03, n.s.
  scale); strict RT's real advantage exists only at FP16/W8A8 draft
  precision (3.90/3.78, both bests on this target).
- The pathology the correction fixes is now identified as
  branch-scale imbalance (α's exact mechanism), not outliers, in the
  native basis.
