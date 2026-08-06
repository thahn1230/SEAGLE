# EAGLE-1 Target/Draft Precision × W4A4-Draft Method Grid (2026-08-05→07)

Run: `runs/eagle1_target_draft_precision_method_grid_20260805_184316`
(pointer `runs/PMG_RUN_DIR`) · Branch
`exp/eagle1-target-draft-precision-method-grid` (base @3cb3fb2) ·
8× RTX 4090 · ~250 scheduler jobs, 0 unresolved failures.
Metric: official micro tau (cycle-pooled accepted+1), greedy,
mc_sim_7b_63, 128 new tokens; datasets mtbench 80 / gsm8k 200 /
sharegpt 80 / humaneval 164 (identical pools to @3cb3fb2, checksummed).
Paired prompt-cluster bootstrap 3000; Holm per dataset-family
(184 comparisons, 126 rejected at 0.05). Macro-AL reported in
`tables/macro_al_summary.json` (never conflated with micro tau).
All quantization FAKE-quant (real-kernel claims out of scope).

## 1. Research questions
RQ1: how does AL move over Target×Draft ∈ {FP16,W8A8,W4A4}²?
RQ2: which of 8 W4A4-draft methods (naive/EP3-G/EP3-P/±R_D × PTQ/QAT)
is best, per target precision — with runtime/foldability accounting.

## 2. Quantization contract
Targets: fp16 stock; w8a8/w4a4 = SpinQuant full (R1+R2+R4, learned
`learned_chat_w4a4kv16` R_T sha 4b7e91d2), RTN weights + clip,
per-token asym acts, KV16. Draft W4A4/W8A8 = concat-projection
first/rec + AR q/k/v/o/gate/up/down quantized (per-out-channel SYM
RTN+MSE-clip weights, per-token ASYM acts); PostProjectionR1 fp32,
embedding/head/norms/KV fp16 — "draft W4A4" is NOT whole-draft-4bit.

## 3-4. Interface structures
T16 row: identity interface (draft consumes h_t). Rotated targets:
draft consumes a_t = n·R_T; γ_f folded exactly once into
W_first = [W_e | W_h·D_γ·R_T] (F0). Stock-draft-on-rotated-target uses
the restored interface (explicit fp32 (a_t R_Tᵀ)⊙γ_f) — F0-equivalent
fold exists and is verified. **Trap fixed during the study**: the
evaluator applied the restore only for int4 targets; W8A8 stock fell
through unadapted → A_T8D16 initially collapsed to 1.14; after the fix
it reproduces TLDR (3.5384 vs 3.5396). Invalid artifacts quarantined.

## 5. Experiment A — 3×3 grid (naive drafts), official tau

mtbench (4-dataset mean in brackets):

| T\D | FP16 | W8A8 | W4A4-naive |
|---|---|---|---|
| FP16 | 3.5762 [3.8035] | 2.1099 [2.1739] | 1.0444 [1.0463] |
| W8A8 | 3.5384 [3.7764] | 3.3671 [3.5264] | 1.2690 [1.2794] |
| W4A4 | 3.2744 [3.5775] | 2.8953 [3.1780] | 1.2521 [1.2667] |

[확보] Draft axis dominates: naive D4 collapses on every target
(1.05-1.28). [확보] W8A8 target is near-free (T8D16 −0.027 vs T16D16
mtbench, within old BWAL noise band). [확보] Quantized drafts are FAR
better under rotated targets than under FP16 (D8: 3.37@T8 vs
2.11@T16) — the a_t/gamma_R1 interface, not precision matching per se
(§11). BWAL/LRAS grids reproduce within noise as regression baselines.

## 6-9. Experiment B — 8 methods × 3 targets (official tau, 4-ds mean; per-dataset in tables/)

| method | T16(fp16) | T8(w8a8) | T4(int4) |
|---|---|---|---|
| B1 naive PTQ | 1.046 | 1.279 | 1.267 |
| B2 generic QAT | 1.573 | 1.828 | 1.801 |
| B3 EP3-G PTQ | 3.108 | 3.575 | 3.326 |
| B4 EP3-G QAT | 3.323 | 3.431 | 3.448 |
| B5 EP3-P PTQ | 3.108 | 3.582 | 3.343 |
| B6 EP3-P QAT | 3.302 | 3.439 | 3.445 |
| B7 EP3-P+R_D PTQ | 3.305 | **3.692** | 3.527 |
| B8 EP3-P+R_D QAT | 3.305 | 3.454 | 3.472 |

Scales (validation-AL-selected per target): T16 (0.46,0.46)/EP3-G
0.46 — at T16 calibrated EP3-P **collapses to EP3-G** (m_f=m_r), so
B3=B5 identically; T8 NEW calibration (0.39,0.45)/0.42; T4 canonical
(0.40,0.45)/0.42. R_D: T4 = RD_HYB_s2 (@3cb3fb2); T16/T8 =
target-matched retrained (T16 required a NEW identity-interface
training mode; the gamma_R1-fold trainer produced a collapsed rotation
first — fixed, v2). Transfer ablation (T4-R_D transplanted): T16
3.249, T8 3.656 — [확보] slightly below target-matched on the
4-dataset mean (T8 3.656 vs 3.692), though it WINS on mtbench alone
(3.445 vs 3.419): target-matched training is the safer default, but
the T4 rotation transfers remarkably well.

QAT fairness (Table C, tables/qat_selection_*.json + lr_selection):
shared LR 1e-5 (rank-sum over {1e-6,3e-6,1e-5} pilots; gen prefers
larger — 1.48→1.67 — ep3p smaller, 3.297 best at 1e-6, −0.012 at
1e-5), anchor init, 3000 steps, batch 1×4, ckpt/500→1500, identical
selection ({step1500,best,last} → calib-AL argmax), 3 seeds; multi-
dataset on the PRE-REGISTERED median-mtbench seed, never the best.
Seed spread ≤0.06 for all P3 arms. One QAT bug found+fixed mid-run:
the C3-branch teacher ignored R_D (B8@T16 collapsed to 1.0; corrected
rdv2 runs: 3.098/3.130/3.129 — quarantined artifacts logged).

## 10-11. Findings across datasets (Holm-adjusted)
- [확보] Generic QAT never escapes the naive basin (1.57-1.83; e.g.
  T4 humaneval naive→genQAT +0.60 SIG but ×2 below any P3 arm).
- [확보] P3-class scaling is the first-order lever (+2.0-2.5 over
  naive, p<1/3000 everywhere).
- [확보] EP3-G vs EP3-P: statistically indistinguishable at every
  target (|Δ|≤0.02, all n.s. after Holm) — pathwise splitting adds
  nothing once the global scale is calibrated (T16 literally equal).
- [확보] R_D on top of EP3-P PTQ: +0.13-0.20 SIG at T4/T8/T16
  (e.g. T4 humaneval +0.127 [0.076,0.181]).
- [확보] QAT gains are target-dependent: helps at T16 (+0.2) and on
  naive; at T8 it HURTS every P3 arm (PTQ 3.58-3.69 > QAT 3.43-3.45);
  at T4 it helps ep3g/ep3p (+0.10-0.12) but NOT rd (B7 3.527 > B8
  3.472; humaneval bestPTQ vs bestQAT −0.156 [−0.209,−0.104]).
- [확보] R_D's edge does NOT survive draft-weight QAT (ep3p-QAT vs
  rd-QAT n.s. at T4/T8) — rotation and weight-training reach the same
  basin; pick ONE: R_D-PTQ (training-free deploy) or plain EP3-P QAT.

## 12. RCAL safeguard (T4 × 8 methods, mtbench 80, proposal-only)

| method | AL_q | RCAL | SAL | AFS |
|---|---|---|---|---|
| B1 naive | 0.252 | 0.229 | 0.023 | 0.914 |
| QF gen | 0.718 | 0.635 | 0.084 | 0.901 |
| B3 EP3-G | 1.996 | 1.648 | 0.347 | 0.850 |
| B5 EP3-P | 2.073 | 1.731 | 0.342 | 0.856 |
| B7 +R_D | 2.200 | 1.777 | 0.423 | 0.832 |
| QF ep3g | 2.192 | 1.755 | 0.437 | 0.827 |
| QF ep3p | 2.176 | 1.759 | 0.417 | 0.835 |
| QF rd | **2.247** | **1.816** | 0.431 | 0.837 |

[확보] Ranking by RCAL matches ranking by AL_q (no method wins by
verifier drift alone; no deceptive-AL flag fires). [확보] ~30-40% of
every P3-arm's AL sits in SAL (W4A4-verifier drift) — consistent with
the R_D study; AFS 0.83-0.86 for all competitive arms.

## 13-15. Runtime, folding, memory (tables/folding_audit.json)
Measured (mtbench decode, CUDA events): draft cycle cost is METHOD-
INDEPENDENT — 25.9-27.5 ms/cycle for B1/B3/B5/B7 alike;
PostProjectionR1 0.60-0.63 ms/cycle for ALL of them (same 4096² fp32
GEMM whether R_T or R_D). [확보] EP3-G/EP3-P/R_D add ≈0 runtime: m
folds into the embedding table + W_e views (F0); m_rec needs one fused
e-slice multiply per recurrent call (F2; dual-table F1 option +250MiB
unused); R_D refolds offline (F0) — no explicit R_D matmul beyond the
PostProjectionR1 that every arm already pays. R4 online Hadamard =
separate op (F2/F3), visible in profiler for all arms. QAT adds no
runtime operator (identical export). ms/token differences between
methods are pure tau differences. ABSOLUTE ms/token here is fake-quant
dominated (T16D16 9.1 vs T4 41-114 ms/tok) — no real-kernel claim.
[provisional] PostProjectionR1 is mathematically absorbable into the
consumer AR weights; left explicit in the current implementation (F1).

## 16. Statistics
stats/bootstrap_pairs_<ds>.json (point, Δ, 95% CI, p with 1/3000
floor), stats/holm_adjusted.json (per-dataset families), seed tables
in tables/median_seeds.json + qat_selection_*.

## 17. Target quality
[미확인] The minimal gsm8k-EM/humaneval-pass@1 harness produced
unusable absolute numbers under the 256-token chat-format budget
(pass@1 0.0 everywhere; EM 0.06-0.12 noise-level, int4>fp16 nonsense
ordering) — retained only as generation checksums
(tables/target_quality_*). Between-target quality comparison remains
OPEN; AL comparisons in this report are never quality claims.

## 18. Known limitations
Fake-quant only; RCAL at T4×mtbench only; QAT budget single
(3000 steps, one shared LR); T8 calibration used the int4-protocol
analogue without the AFS tiebreak; target-quality open (§17);
mid-study bug fixes (restored-T8 interface, C3+R_D teacher) with
quarantined artifacts (logs/known_issues.md, quarantine/).

## 19. Safe claims
Draft precision dominates the grid; W8A8 target ~free; naive W4A4
draft unusable; calibrated global scale (EP3-G) captures ~all of P3;
R_D is the best PTQ overhead-free upgrade; QAT does not beat the best
PTQ at rotated targets; all method runtime deltas ≈0 (fold/fuse).

## 20. Claims NOT to make
Old-grid numbers ≠ this contract (different runners/metrics); micro
tau ≠ proposal-only AL (RCAL AL_q = tau−1); EP3-P activation scales do
NOT co-fold into ONE shared table (two views or fused mul); higher AL
≠ better model quality; single-seed QAT generalization; real-kernel
latency from these fake-quant timings; "draft entirely 4-bit".

## §23 final answers
Q1 highest-AL cell: T16D16 3.576/3.804; ex-baseline: **T8D16**
(3.538/3.776); with methods: **T8 + EP3-P+R_D PTQ draft = 3.419 mtb /
3.692 4-ds** — best quantized-draft combination [확보].
Q2 matched precision NOT specially favored: T8D8 good (3.53) but
T8D16>T8D8; T4D4<T4D8<T4D16 — no diagonal advantage [확보].
Q3 T16+quantized-draft weakness is the INTERFACE, not quantization
alone: same D8 draft: 2.17@T16 vs 3.53@T8 [확보] (§5).
Q4 Generic QAT stays far below P3 even at matched budget/LR (its best
LR direction was toward 3e-5 failure-control; still ×2 gap) [확보].
Q5 EP3-G↔EP3-P: n.s. everywhere after Holm [확보].
Q6 P3-QAT ≫ generic QAT (+1.6-1.8, p<1/3000) [확보].
Q7 R_D improves EP3-P PTQ at every target (+0.13-0.20 SIG) [확보].
Q8 R_D gain does NOT survive QAT (rd-QAT ≈ ep3p-QAT n.s.) [확보].
Q9 Reference-consistent share: RCAL/AL_q ≈ 0.81-0.84 for P3 arms;
SAL ≈ 0.34-0.44 [확보].
Q10 Extra runtime per method: ≈0 measured; shared costs =
PostProjectionR1 0.6 ms/cyc + R4 Hadamard [확보].
Q11 Fully weight-foldable: R_T/γ_f first fold, EP3-G m, EP3-P weight
sides, R_D, R2 [확보].
Q12 Not foldable into ONE shared embedding table: EP3-P (m_f≠m_r)
activation side — dual views or fused multiply [확보].
Q13 Explicit ops remaining in profiler: PostProjectionR1 GEMM, m_rec
e-slice mul, R4 Hadamard, restored-interface restore (stock arms);
no other rotation/scale op [확보].
Q14 Recommendation: **W8A8 target + W4A4 draft with EP3-P+R_D PTQ**
(3.69 mean AL, zero method-runtime overhead, no training) — if the
target must be W4A4: same method (3.53); QAT only worth it for T16
targets or when P3 calibration is impossible [확보/provisional mix —
latency ranking real-kernel-unverified].
