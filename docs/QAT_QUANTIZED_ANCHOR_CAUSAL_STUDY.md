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

## RESULTS — Phase C: code-frozen LoRA (Gate C, 2026-08-15)

**Gate C: NEGATIVE — the code-frozen adapter does NOT dominate.**
FP16 LoRA side-branches on the frozen anchor codes (D_Q = 0 by
construction; ranks 4/8/16 @1e-4, r8 @1e-3; calib-selected
r16 @1e-4 st1500, calib 3.2616 < PTQ 3.2794). Final 4-dataset
deployment: mean4 3.3914 vs PTQ 3.4788; Holm verdict 0W/2L
(gsm8k -0.117, humaneval -0.203, both p < 1e-4). Expressivity outside
the quantized cells (a fp16 bypass) is NOT the mechanism; moving the
codes themselves (a few, chosen well) is.

## RESULTS — Phase D: anchored QAT (2026-08-15)

All anchored deployments use the FROZEN anchor W4 scales
(--anchor-scales; A4 stays dynamic). Baselines: PTQ anchor
(CANON_B9) mean4 3.4788; plain aggressive chain-QAT at the same
LR 1e-5 (CFIN_P2conv_gsr5, 3 seeds) 3.4443/3.4510/3.4237 — Holm vs
PTQ 0W/1L (humaneval -0.0985, p < 1e-4), H_Q 13.90%, down flips 31%.

**D1. Soft cell loss, 3000-step horizon (preregistered grid) is
tau-NEUTRAL.** {conv,hybrid} x beta {0.1,1} x 3 seeds, cadence-selected
on calib: every family's best calib (<= 3.2509) fell below the PTQ
calib reference 3.2794; the best family (conv beta 1 @st600, median
seed) finals at mean4 3.4812 ~= PTQ, Holm 0W/0L. The cell loss at
these betas does not rescue the aggressive 1e-5 trajectory over a full
horizon: H_Q still 6.76% at st600 (schedule keeps LR high early).

**D2. Soft cell loss, 600-step horizon WINS.** The same recipe with
the LR schedule compressed to 600 steps (decayed LR at the cadence):
conv beta 1 -> mean4 3.5492/3.5538/3.5523 (3 seeds), hybrid beta 0.1 ->
3.5523/3.5634/3.5525. Holm (median seed) vs PTQ: conv 2W/0L, hybrid
3W/0L; vs plain QAT: conv 4W/0L. Drift at deployment (median seeds):
conv H_Q 4.42% (down 8.3%), hybrid H_Q 2.82% (down 3.5%) — an order
of magnitude closer to the anchor than plain QAT.

**D3. Hard flip-budget projection is the best method in the study.**
Trainer-side projection every 100 steps: utility
u = -g*(Q_new - Q_anchor) per flipped weight (g = task gradient via
the STE identity), deterministic top-K within budget
K = frac * N_sites, only positive-utility flips kept, all other
weights clamped into the anchor cell (|u - c0| <= 0.49; escapee
retry with tightening margin for the down FWHT round-trip). Site
policy per contract section 2: exact on the 7 uncoupled sites + W_rec
(e-half -> W_e, h-half -> W_h via the recurrent fold); W_first
measured-only. Budgets {0.25,0.5,1,2,5}% x {conv,hybrid}, LR 1e-5,
no cell loss, calib-selected cadence (9/10 chose st3000):

| model (st3000) | mean4 | vs PTQ (Holm) | H_Q | D_Q | down flip |
|---|---|---|---|---|---|
| hybrid b0.5% | **3.6276** | 3W/0L (humaneval n.s.) | 1.30% | 0.00066 | 0.002% |
| hybrid b1% | 3.6021 | 3W/0L | 1.28% | 0.00065 | 0.003% |
| hybrid b2% (= b5%) | 3.5991 | 3W/0L | 1.27% | 0.00065 | 0.003% |
| conv b2% | 3.5877 | **4W/0L** | 1.56% | 0.00080 | 0.055% |
| conv b0.5% | 3.5627 | — | 1.63% | 0.00084 | 0.040% |
| conv b1% | 3.5600 | — | 1.58% | 0.00081 | 0.064% |
| conv b5% | 3.5636 | — | 1.54% | 0.00079 | 0.070% |

conv b2% sweeps ALL FOUR datasets post-Holm; hybrid b0.5% has the
best mean4 (+0.149 over PTQ, +0.183 over plain QAT at the same LR).
Every anchored candidate beats plain QAT 4W/0L (10k paired cluster
bootstrap, p < 1e-4 on all four datasets for hybrid b0.5%).
hybrid b2% and b5% converged to the IDENTICAL model (the
positive-utility cap binds below both budgets; noted, deduplicated).
The final kept-flip count self-limits at H_Q 1.3-1.6% for every
budget — the converged models sit exactly in the Gate-B-predicted
beneficial band (~1-2%), and their kept sets contain essentially NO
down_proj flips (share <= 0.8% of kept vs 37% in the aggressive
drift), i.e. utility selection independently discovers the module
structure Gate B exposed.

**Frontier verdict (the preregistered success criterion).** In the
(tau, H_Q) and (tau, D_Q) planes the Pareto frontier is formed
entirely by anchored candidates: PTQ (3.4788 @ 0), hard-budget
(3.59-3.63 @ 1.3-1.6%), soft-cell-600 (3.55-3.56 @ 2.8-4.4%);
plain QAT (3.44 @ 13.9%) and code-frozen LoRA (3.39 @ 0) are
strictly dominated. Quantized-anchor preservation — not a specific
tau number — is established as an optimization principle for
speculative-draft QAT under this deployment contract.
Figures: figures/pareto_tau_vs_{HQ,DQ,downflip}.png (+ raw CSV),
figures/gateB_revert_curve.png; stats/final_stats_holm.json;
tables/{final_taus,final_drift_metrics,selectivity,qsel_selection}.json.

**Selectivity.** Kept-flip Jaccard across budgets within an objective:
0.62-0.72 (approximately nested); across objectives ~0.40 (the
utility ranking is objective-dependent but substantially shared).
Per-site kept shares concentrate in W_rec/q/k/v/o/gate/up and avoid
down (<= 0.8%).

**Limitations (binding).** (1) All results are FAKE-quant
(quantize-dequantize, fp16 GEMMs); no real-INT4 kernel claims.
(2) W_first's h-half is measured-only under the preregistered site
policy: its flip rate stays 10.4-12.2% in the HB finals (the first
fold sees W_h through gamma_f o R1_T, which the recurrent-fold
projection does not control) — tightening it is future work.
(3) humaneval is the weakest dataset (n.s. for the hybrid HB models;
only conv b2% sweeps it). (4) The FP16 A3 depth-collapse attribution
remains DEFERRED (no validated FP16 control). (5) The 600-step
soft-cell win is entangled with its LR schedule; we claim the
combination, not the loss alone. (6) Single seed for HB (s0);
soft-cell-600 has 3 seeds.

## 30-item final summary

1. Anchor = GS+R5 PTQ draft, captured from the ADAPTER's own folds
   (sha d290220c), frozen W4 scales + c0 for the entire study.
2. Gate A PASS: training-forward vs deployment parity max|dlogit|=0
   (both structural modes); 7 anchor unit gates PASS.
3. Frozen-c0 deployment reproduces PTQ EXACTLY (3.4788) — pipeline
   validated end-to-end.
4. Gate B verdict B-MIXED: drift is causal WITH structure.
5. Gate B peak: retaining ~10% of aggressive flips (H_Q 1.4%) BEATS
   both endpoints (3.509-3.534 vs 3.4788/3.4504); two independent
   trajectories (revert, interpolation) agree.
6. Gate B entanglement: down-only retention collapses to 2.314 —
   partial reversion can be far worse than either endpoint.
7. Aggressive plain QAT (1e-5): H_Q 13.9%, down 31%, finals 3.4443
   (median), Holm 0W/1L vs PTQ — net harmful.
8. Plain QAT loses to PTQ on humaneval (-0.0985, p<1e-4).
9. Phase C code-frozen LoRA (D_Q=0): 3.3914, Holm 0W/2L — bypass
   expressivity is NOT the mechanism; Gate C negative.
10. Phase D soft cell loss @3000-step horizon: tau-neutral (0W/0L);
    calib never beats PTQ; H_Q still 6.8% at best cadence.
11. Phase D soft cell loss @600-step horizon: mean4 3.549-3.563
    across 6 runs (2 objectives x 3 seeds) — all above PTQ.
12. Holm: soft-cell-600 hybrid beta 0.1 3W/0L vs PTQ; conv beta 1
    2W/0L; conv 4W/0L vs plain QAT.
13. LR-schedule horizon is decisive for the soft penalty (600 vs 3000
    at the same beta flips the verdict).
14. Hard-budget projection: utility-ranked top-K flip retention +
    in-cell clamping, exact on 8 folds, verified post-refold every
    projection (zero non-kept flips, hard-gated).
15. HB best mean4 3.6276 (hybrid b0.5%): +0.149 vs PTQ, +0.183 vs
    plain QAT; p<1e-4 on 3 datasets (humaneval n.s.).
16. HB conv b2% (3.5877) beats PTQ on ALL FOUR datasets post-Holm —
    the only 4W/0L candidate vs PTQ.
17. Every anchored candidate beats plain QAT 4W/0L.
18. HB kept flips self-limit at H_Q 1.27-1.63% for ALL budgets
    (positive-utility cap) — inside the Gate-B beneficial band.
19. hybrid b2% == b5% bitwise (cap binds below both) — deduplicated.
20. HB kept sets avoid down_proj (<=0.8% of kept vs 37% aggressive
    share): utility selection rediscovers Gate B's module structure.
21. HB drift: D_Q 0.00065-0.00084 (17-22x smaller than aggressive
    0.0144); D_FP 0.002-0.0045.
22. Selectivity: Jaccard 0.62-0.72 across budgets within objective
    (nested-ish); ~0.40 across objectives.
23. Pareto frontier (tau vs H_Q and tau vs D_Q) is formed entirely by
    anchored candidates; plain QAT and LoRA strictly dominated —
    preregistered success criterion (frontier dominance) MET.
24. Cadence selection on calib c4:20 offset-500 only; finals on the
    4 frozen pools, official micro-tau, greedy, mc_sim_7b_63.
25. Statistics: 10k-rep paired same-prompt cluster bootstrap, p
    floored at 1e-4, Holm across the 4 datasets within each family.
26. All anchored deployments use frozen anchor W4 scales; A4 stays
    per-token dynamic (deployment contract).
27. Fake-quant only; no real-INT4 deployment claims.
28. W_first h-half measured-only: 10.4-12.2% flips remain in HB
    finals (preregistered limitation; first-fold control is future
    work).
29. FP16 A3 depth-collapse attribution stays DEFERRED (no validated
    FP16 control); no claim made.
30. Novelty claim limited to: quantized-anchor preservation as an
    optimization principle for speculative-draft QAT (utility-selected
    small flip budgets + frozen-scale deployment), demonstrated by
    frontier dominance on one model/deployment contract.
