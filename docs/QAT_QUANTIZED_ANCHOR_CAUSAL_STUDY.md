# QAT Quantized-Anchor Causal Study — Contract & Live Log

Branch `exp/eagle1-qat-quantized-anchor-causal` (base df2e131).
Run root: `runs/eagle1_qat_qanchor_causal_20260814/`.
Status: PHASE A in progress. This document is the preregistered contract;
results sections are appended as phases complete.

## 1. Evaluation contract (identical to the canonical AA-QAT study)

- Evaluator: `scripts/eval_eagle_acceptance_length.py` — greedy
  (temperature 0.0), batch 1, official tree `mc_sim_7b_63` (25 nodes,
  depth 5), `--max-new-tokens 128`, prompt truncated to 1024 tokens,
  llama-2-chat template.
- Primary metric: cycle-pooled official micro-tau =
  sum(accepted+1)/n_cycles pooled over all cycles of all prompts;
  shards store per-cycle accepted+1 in `acceptance_list` (never
  converted). mean4 = arithmetic mean of the four dataset-level taus
  (never cycle-pooled across datasets). Macro-AL is never substituted.
- Datasets/pools (manifest-pinned, Gate F): MT-Bench 80 (turn-1 frozen
  pool) / GSM8K 200 (test rows 0-199) / ShareGPT 80 (Aeala first human
  turn 60-1200 chars, order-pinned) / HumanEval 164. Calib = c4:20
  offset-500 pool (selection only; never test sets).
- Target: canonical W4A4 KV16 (`learned_chat_w4a4kv16`, R.bin sha
  4b7e91d2 — ORIGINAL-server function, 17/17 exact baseline
  reproduction certified 2026-08-13). Draft deploy: D4P3 fake-quant
  (9 GEMM sites W4 after fold; embedding/head/norms/KV fp16),
  GS alpha 32.89964245299412, R5 = RD_HYB_s2 (sha 20c03f00) frozen.
- Paired stats: same-prompt cluster bootstrap, >=10,000 reps for final
  claims (pilots 3,000), two-sided p floored at 1/reps, Holm across the
  four datasets within each planned family. "n.s." is never treated as
  equivalence; equivalence claims require a preregistered-margin test.
- Diagnostics reported where applicable: proposal-only AL = tau-1,
  RCAL (existing capture/metric scripts), depth-wise teacher-forced and
  free-running acceptance, W4 code-flip rate H_Q, dequantized drift
  D_Q, FP master drift D_FP, module-wise flips (W_first, W_rec,
  q/k/v/o/gate/up/down).
- All quantization is FAKE-quant (quantize-dequantize, fp16 GEMMs).
  This repository contains real INT4 CUTLASS kernels
  (`kernels/`, RealInt4Linear) but they are NOT wired into this
  evaluation path; no real-INT4 deployment claims are made. The parity
  target for Phase A is: QAT fake-quant forward vs canonical W4A4
  fake-quant evaluator semantics.

## 2. Anchor definition (A2 contract)

- Deployed anchor = GS+R5 PTQ: public draft `yuhuili/EAGLE-llama2-chat-7B
  @44e37ec3` masters, folded into the deploy basis
  (W_first=[W_e/a | W_h*(gamma_f o R1_T)], W_rec=[W_e/a | W_h*R_D],
  q/k/v/o/gate/up/down per the runtime fold), then W4-quantized with
  the official per-out-channel symmetric RTN + MSE-clip quantizer.
- **Frozen anchor quantizer**: scales s_i (and maxq=7 clamp) captured
  ONCE from the anchor folds and FROZEN for every anchor experiment
  (training AND evaluation of constructed/anchored models). Activation
  quantization stays dynamic per-token asymmetric (deployment
  contract). c0_i = clamp(round(fold(W0)_i/s_i), -maxq, maxq) is then
  a fixed cell for the entire study (unit-tested).
- Cell space = FOLDED deploy basis (the deployment-visible codes).
  R5/GS/rotations frozen => folds are fixed linear maps => cells are
  well-defined. Coupling note: W_first and W_rec share masters
  (W_e, W_h); their e-halves share identical cells (same fold column
  scale); their h-halves differ by rotation. Soft cell penalties apply
  to all 9 sites through the fold graph. Hard-budget projection is
  enforced exactly on the 7 uncoupled sites + the shared e-half via
  the e-fold, and applied to W_h via the RECURRENT fold (the chain
  training path), with W_first h-half flips MEASURED and reported but
  not independently projected (preregistered limitation; the report
  will state it).

## 3. Reused artifacts (no reruns; §23)

- PTQ rows: CANON_B1/B3/B5/B7/B9 shards (canonical run
  `runs/eagle1_aaq_canonical_20260813_082500`).
- Plain-QAT LR frontier points that already exist (public-draft init,
  chain pipeline, canonical corpus `lkcorpus__can_all`):
  conv/hybrid @1e-5 (CAN_P2_[AB]_{conv,hybrid}_s0..2),
  conv/hybrid @1e-6 (CAN_T6_*), hybrid @3e-6,1e-6 seed0 (LRP pilots,
  pilot-track corpus — labeled where used). New frontier points needed:
  3e-7 (conv+LK), 3e-6 (conv; LK reuse pilot or rerun canonical).
- Aggressive checkpoint (B1/B2): PRIMARY = CAN_P2_B_conv_s0
  (lr 1e-5, same lineage as the anchor: public-draft init + canonical
  corpus). SECONDARY (labeled confound: anchor.pt init + ShareGPT
  data): CQH_gsr5 (historical 43% down_proj flips).
- Free-running/teacher-forced depth diagnostics: canonical
  freerun_diag.json (AA draft) + new A3 FP16/PTQ controls.

## 4. Phase gates (stop/go)

- Gate A: QAT-forward vs evaluator parity on identical states (32
  MT-Bench states x first/recurrent paths x depths; max|dlogit|, TV,
  top-1/top-k agreement). Material disagreement => STOP, fix parity.
- Gate B: flip-revert + interpolation verdict in {B-positive, B-mixed,
  B-negative}; B-negative => the anchor-loss paper direction stops.
- Gate C: code-frozen adapter dominance check.
- Gate D: anchor beta frontier vs FULL plain-LR frontier (Pareto), on
  calib/validation for selection, final contract once.

(Results appended below as phases complete.)

## RESULTS — Phase A (2026-08-14)

**Gate A: PASS.** Gate-D harness (`tables/gateD_qat_parity.json`):
QAT training forward vs canonical deployment adapter — all 9 quantized
weight tensors byte/value-equal and the K=4 teacher-forced chain equal
in hidden/logits/greedy tokens, in BOTH structural modes (identity,
gamma_R1); max|dlogit| = 0.0. The parity target is fake-quant-vs-
fake-quant (no real INT4 kernel in this path; stated in section 1).
A2 anchor semantics: frozen-scale module + 7 unit gates PASS; anchor
captured (sha 98c9ee85, 236M weights, self-flip 0).

A3 status: W4A4 depth curves measured (PTQ free-running gap at depths
2-4: 0.184/0.222/0.248; tuned-QAT: 0.161/0.173/0.253 — the tf-vs-free
gap is NOT reduced by tuned QAT). FP16 control: first measurement gave
implausibly low absolute alphas (tf 0.41 at depth 1); the identity-mode
Gate-D parity PASSES, so the chain implementation matches the runtime —
the fp16-CORPUS/interface pairing is under re-audit before the control
is used. The depth-collapse attribution claim is therefore DEFERRED
until a validated FP16 control exists (interpretation rule respected).

B-prep (`tables/b_prep_flip_stats.json`): same-lineage aggressive
checkpoint = CAN_P2_B_conv_s0@st3000, H_Q 13.90%, down_proj 31.02%,
D_Q 0.01435; monotone step trajectory captured (st100..st3000).
Tuned-hybrid selected ckpt: H_Q 1.00% (down 1.20%).

## RESULTS — Phase B / Gate B (2026-08-15 01:55)

**Gate B verdict: B-MIXED (causal, with structure).** Full curves:
`tables/gateB_curves.{json,csv}` (26 constructed-code configs + frozen
PTQ endpoint, each 4-dataset mean4 under code-verified frozen-scale
deployment).

- Validation: frozen-scale c0 deployment = 3.4788, EXACTLY the
  fresh-scale CANON_B9 PTQ value (anchor scales == W0's own scales).
- Aggressive (H_Q 13.90%, down 31%) = 3.4504 (-0.028 vs PTQ) — mild
  net damage at full deployment.
- NON-MONOTONIC revert curve with a PEAK: retaining a random ~10% of
  the aggressive flips (H_Q 1.39%) yields 3.509/3.525/3.534 across
  three revert seeds (+0.03..+0.055 over PTQ); interpolation alpha=0.1
  (H_Q 1.36%) gives 3.517 — two independent trajectories agree.
  => a small subset of code flips is BENEFICIAL; most are
  harmful-to-neutral. Direct empirical support for a small flip budget
  (~1-2%) and for utility-selected retention (Phase D hard budget).
- MODULE ENTANGLEMENT: keeping down_proj's 31% flips while reverting
  the other sites COLLAPSES the model (nondown_f100 = 2.314;
  f75 2.650; f50 3.079), while reverting ONLY down (down_f100) is mild
  (3.457). down's drift is tolerable only jointly with compensating
  flips elsewhere — partial reversion can be far worse than either
  endpoint. Interpretation rule respected: we do NOT claim monotonic
  "flips cause damage"; the causal statement is that the aggressive
  configuration's EXCESS flips beyond a small useful subset reduce tau,
  and cross-module coupling makes naive partial reversion dangerous.
