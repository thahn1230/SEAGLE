# Experiment Contract — Precision × Method Grid (2026-08-05)

Frozen BEFORE any experiment. Deviations require a logged amendment.

## Evaluation contract (all arms)

- Evaluator: `scripts/eval_eagle_acceptance_length.py` (official
  micro tau = cycle-pooled sequence growth per cycle, includes the one
  verifier token). Macro-AL additionally computed by a new
  `scripts/compute_macro_al.py` from the same shards (per-prompt mean
  of acceptance_list, then unweighted prompt mean) — reported
  alongside, never conflated with official tau.
- Datasets (eval pools, identical to @3cb3fb2): mtbench 80
  (single-turn requests, turn 1), gsm8k 200, sharegpt 80,
  humaneval 164. Held-out selection pool: c4 calib 20 (offset
  pool). Prompt checksums recorded in dataset_checksums.txt; gate
  test asserts equality with the R_D study shards' prompt ids.
- Generation: greedy T=0.0, max_new_tokens 128, tree mc_sim_7b_63,
  max_steps 136, prompt truncation as in evaluator (1024), llama2
  chat template, KV reset per prompt, batch 1.
- Statistics: paired prompt-cluster bootstrap 3000 reps
  (bootstrap_eagle_tau seed 20260723; RCAL 20260731). p reported as
  p < 1/3000 when no replicate crosses. Holm correction via new
  adapter over the pre-registered comparison families (§16 of spec).
- Target precisions: fp16 = (rot none, quant none); w8a8 = (rot full,
  quant w8a8, KV16); w4a4 = (rot full, quant w4a4, KV16). KIND
  learned_chat_w4a4kv16 for all rotated targets. `--target w8a8`
  added to eval + capture scripts (same backfill/stash semantics as
  int4).
- Draft precisions: D16 = fp16 adapter path; D8 = fake_w8a8 on
  first/rec/AR (ar_r2r4 on); D4 = fake_w4a4 same sites. Interface:
  T16 row first_hidden_mode=identity; T8/T4 rows gamma_R1. Stock-draft
  cells on rotated targets use the folded gamma_R1 path (F0-equivalent
  to the restored interface — verified by gate test).

## Experiment A — 3×3 grid (precision effect only)

9 cells; D4 = **Naive W4A4 PTQ** (no P3, no R_D, no QAT), D8 = naive
fake_w8a8 (same policy class), D16 = fp16 draft. All interface
correctness invariants kept (gamma exactly once, basis-correct folds).

## Experiment B — 8 methods × 3 targets (24 primary arms)

| method | scales | rotation | training |
|---|---|---|---|
| B1 Naive PTQ | none (m=1) | R_T shared | none |
| B2 Generic QAT | none (m=1) | R_T shared | draft-core QAT |
| B3 EP3-G PTQ | single m (per-target calibrated) | R_T | none |
| B4 EP3-G QAT | B3 m frozen | R_T | draft-core QAT |
| B5 EP3-P PTQ | (m_f, m_r) per-target | R_T | none |
| B6 EP3-P QAT | B5 scales frozen | R_T | draft-core QAT |
| B7 EP3-P + R_D PTQ | B5 scales | target-matched R_D | R_D only (frozen weights) |
| B8 EP3-P + R_D QAT | B5 scales frozen | B7 R_D frozen | draft-core QAT |

- Canonical scale values (validated per target on calib AL, not
  proxy-only): T4 EP3-P (0.40, 0.45), EP3-G β 0.42 (m 32.900);
  T16 EP3-P (0.46, 0.46), EP3-G β 0.46 (m 45.887); T8 = NEW
  calibration (capture → NMSE grid → calib-AL top-k → select),
  protocol identical to the int4 one (calibrate_eagle1_p3exp.py +
  ep3sel phase).
- B7 R_D: T4 = RD_HYB_s2 (reuse, canonical finalist). T16/T8 =
  target-matched NEW training (residual Cayley, hybrid objective,
  3000 steps, lr 3e-4, seed 2 convention, teacher = that target's
  precision, corpus = target-generated 5×1000 windows per target).
  Transfer ablation (T4-trained R_D on T16/T8) reported separately.
- B2/B4/B6/B8 QAT (12 conditions × 3 seeds): canonical trainer,
  official EAGLE loss, anchor init, frozen rotations/scales, 3000
  steps, ckpt every 500, effective batch = OF protocol, best-val
  selection on c4 calib 20, mtbench+3 datasets on selected ckpt.
  LR: fresh validation-only pilot (1000 steps × {1e-6, 3e-6, 1e-5} ×
  {Generic, EP3-P} on T4) → ONE LR applied to ALL 12 conditions.
  3e-5 failure-control short run only. Same budget everywhere
  (Table C proves it).
- Teacher contract: target-matched (T16/T8/T4 teachers); C3-style
  identity interface for T16, C7-style gamma_R1 for T8/T4.

## RCAL safeguard

T4 × all 8 methods × mtbench 80 (n=80 captures, 2 GPUs each),
metrics + paired bootstrap vs B1 and vs each PTQ/QAT sibling.
More datasets only if wall-clock allows.

## Target quality safeguard

Per target precision: target-only greedy output checksum + FP16-target
token agreement on fixed prompts; NEW minimal gsm8k exact-match +
humaneval pass@1 harness on TARGET-ONLY generations (max_new_tokens
256; limitation: short budget understates absolute quality — used for
BETWEEN-target comparison only, never as absolute capability).

## Abort conditions

Spec §25 verbatim; especially: T16D16 stock reproduction within noise
(±0.03 of 3.5762/3.6276-class anchors after contract matching),
Gate-P R_D bitwise, budget-mismatch between Generic and P3 QAT.
