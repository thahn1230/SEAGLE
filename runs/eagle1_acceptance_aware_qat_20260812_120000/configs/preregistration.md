# Preregistration — Acceptance-Aware QAT for W4A4 EAGLE-1 (2026-08-12)

Run: `runs/eagle1_acceptance_aware_qat_20260812_120000`, host `gpusystem`,
branch `exp/eagle1-draft-aware-r1-r2-gs` (+ this study's extensions).
Deployed contract: target = SpinQuant W4A4 KV16 (`learned_chat_w4a4kv16`,
R.bin sha 0a2d1997…, re-optimized on THIS host), draft = fake-W4A4 D4P3
(9 GEMM sites quantized after fold; embedding/head/norms/KV fp16), greedy,
official tree mc_sim_7b_63, official cycle-pooled micro-tau, 128 new
tokens, pools mtbench 80 / gsm8k 200 / sharegpt 80 / humaneval 164
(manifests pinned in `manifests/`, Gate F passed).

## 1. Question

Should draft-weight QAT for a FIXED deployed W4A4 verifier optimize
verifier acceptance directly (LK-style / survival / greedy-token
surrogates against the W4A4 target's own distributions) rather than
target-hidden reconstruction (SmoothL1 + 0.1 SoftCE)? Most important
cell: does GS+R5, where conventional QAT adds nothing over PTQ, finally
benefit from QAT when the objective is acceptance?

## 2. Reference baselines (recovered exactly; NOT comparable across hosts)

Previous-server lineage (grid 20260805 + r6-branch constants; context
only): Naive PTQ 1.267 / Naive convQAT 1.801 / GS PTQ 3.326 / GS convQAT
3.448 / LS PTQ 3.3429 / LS convQAT 3.445 / GS+R5 PTQ (B9_T4) 3.4788 /
GS+R5 convQAT (QF_rdg_T4_s2) 3.4541 / LS+R5 PTQ 3.5267 / LS+R5 convQAT
3.472. The user's "GS convQAT+R5 ~3.43" resolves to QF_rdg_T4_s2 =
3.4541; 3.43 matches A4_COMPOSED 3.4337 = GS+R5+R6 composed, NO QAT —
an R6 artifact explicitly excluded here.

Current-host anchors (R1R2GS study, artifacts on disk): GS PTQ (A0)
mean4 **3.30876**; GS+R5 PTQ (A1: s1001 3.46441, s1002 3.48900, s1003
3.47244; median-val seed s1001). ALL new comparisons are within
current-host arms only; prev-server absolutes are never mixed in.

## 3. Structural arms and frozen inputs

| Arm | alpha (frozen) | R_D basis (frozen) | Init draft weights |
|---|---|---|---|
| Naive | 1.0 | R1_T (identity fold semantics of grid B1: rotated pipeline, alpha=1) | public yuhuili draft |
| GS | 4096^0.42 = 32.89964245299412 | R1_T | public draft |
| LS | (4096^0.40, 4096^0.45) = (27.85761802547598, 42.22425314473262) | R1_T | public draft |
| GS+R5 | GS | `RD_GS_A1_s1001.pt` R_D (sha 25068f60…, R5-only, R6-free per param audit) | public draft |
| LS+R5 | LS | `RD_LS_s{1001..1003}.pt` (this run; A1 protocol under EP3-P scales, same corpus, seeds 1001-3, median-val seed) | public draft |

R5 frozen in every QAT arm (`--rot-fixed-ckpt`); no R5 relearning, no
joint W+R5, no R6, no target training. W4A4 target completely frozen.

**Initialization deviation (declared):** the 20260805 grid initialized
QAT from the from-scratch FP16 anchor (absent from this host, sha
a7c6ccc8 unrecoverable). Here EVERY QAT variant starts from the public
draft masters == the corresponding PTQ deployment state, per study
section 19 (removes the grid's anchor-vs-public asymmetry; conv-QAT
rows are therefore current-host re-anchors, not grid reproductions).

## 4. Objectives (exact executable formulas)

All chain objectives use the K=4 teacher-forced recurrent chain
(`forward_chain`), corpus teacher_logits (fp16-stored) as z_T, draft
chain logits as z_D, full 32000 vocab, temperature 1. Depth weights
w_k = 0.8^(k-1) **UNNORMALIZED** (code+test canonical; the R1R2GS
doc/prereg normalized formula was a documentation error).

- **Q0h (conventional, historical contract)** — `train_eagle_draft_int4_qat.py`
  arm C7: single-step first-path, ShareGPT-12k pinned corpus, loss
  1.0·SmoothL1(h,t) + 0.1·SoftCE, t = ((a R1ᵀ)·γ_f)·R_int, LR 1e-5,
  3000 steps, bs 1×4, warmup 200, AdamW(0.9,0.95), value-clip 0.5.
  This column IS "Conventional QAT" in the final table.
- **Q0c (conv-in-chain, isolation control)** — same loss computed at the
  K chain positions from corpus `a_chain`, uniform depth mean.
- **Q1 (LK)** — canonical hybrid: per depth, lam_k = exp(-3·sg(mean
  alpha_k)); L_k = lam_k·KL(p‖q) + (1-lam_k)·TV; L = Σ w_k L_k.
- **Q2 (survival)** — L = −E[expected_tau] = −E[1 + Σ_k Π_{j≤k} α_j]
  (linear form; `exptau` = 0.1·KL + −log(expected_tau/(K+1)) runs as a
  pilot variant Q2b).
- **Q3 (greedy)** — L = Σ w_k · E[−log q(argmax p_k)]; variant Q3b
  `accsurv` = −E[Σ_k Π_{j≤k} q_j(t_j)] (soft greedy prefix survival).

Trainable: 11 core masters (W_e,W_h,q,k,v,o,gate,up,down + b_fc + gl64)
through STE fake-W4A4 (per-out-channel sym RTN + MSE clip re-estimated
every step; per-token asym dynamic A4). Gate E-core passed for every
objective (gradchecks/gate_core_chain.json).

## 5. Matched objective-isolation contract (Phase 1/2)

Anchors: GS and GS+R5. All chain arms identical: corpus (regenerated
GS-teacher t4 corpus with a_chain, same prompts/seeds as R1R2GS,
bit-repro checked), split (last-10% val), batch 32×1, K=4, kv-bits 16,
3000 steps, warmup 100, cosine, grad-norm clip 0.5, AdamW(0.9,0.95),
**core-lr 1e-5** (canonical QAT LR), seeds 0/1/2, eval-every 100,
core ckpts at {0,100,250,500,1000,1500,2000,2500,3000}.
Pilots first: 600 steps, seed 0 only, both anchors × {Q0c,Q1,Q2,Q2b,
Q3,Q3b}; an objective is dropped for instability iff val expected_tau
falls >0.15 below its own step-0 value on two consecutive evals, or
NaN. Survivors run the full matched contract.

## 6. Checkpoint selection (identical for every arm/objective)

AMENDED 2026-08-12 (before any Phase-2 confirmatory run; original
"best-val-by-own-objective" candidate is unavailable for core exports —
the trainer's best-val snapshot covers rotations only): candidates =
cadence checkpoints at steps {1000, 1500, 2000, 2500, 3000}; winner =
argmax c4-calib 20-prompt micro-tau (held-out pool, offset 500 — the
grid's fairness gate; identical rule for every arm/objective).
No selection ever touches the 4 eval datasets. Phase-2 checkpoint
exports live under /data/thahn1230/aaq_ckpts/ (disk budget), pruned to
the selected + final ckpts after selection.

## 6b. Budget-sensitivity analysis (EXPLORATORY; added 2026-08-12 22:20
after Phase-2 selection, before Phase-4 launch)

Phase-2 outcome at the pre-registered ≥1000-step candidates: every
objective's selected calib tau at anchor B sits BELOW B's PTQ (best
hybrid 3.1119 vs 3.2388), selections concentrate on the earliest
candidate (st1000 in 9/24 runs), while 600-step pilots beat all full-run
candidates — training alpha rises to 0.85-0.95 as deployed tau falls
(teacher-forced overfit signature). MOTIVATED ADDITION: evaluate the
already-saved cadence ckpts {100, 250, 500} on the same c4-calib pool
for all 24 runs. The ≥1000 rule REMAINS the pre-registered primary;
the <1000 sweep is reported separately as exploratory budget-sensitivity
(it answers "is the binding constraint the objective or the budget?").
The pre-specified LR pilot (section 7) doubles as the confirmatory
version of the same question at fixed 3000 steps.

## 7. Phase 3 — objective lock + LR pilot

Lock ONE acceptance-aware objective by (a) matched-LR val expected_tau
+ c4-calib tau at selected ckpt, (b) stability. Then LR pilot {1e-6,
3e-6, 1e-5} on the winner (both anchors, 1 seed, val-only selection).
Report matched-LR comparison (A) and tuned result (B) separately.

## 8. Phase 4 — final arms (current host)

PTQ rows: Naive, LS eval'd fresh; GS=A0, GS+R5=A1 reused; LS+R5 eval'd
with RD_LS median-val seed. QAT rows: 5 arms × {Q0h conv, locked AA} ×
3 seeds. mtbench for all seeds; the pre-registered **median-mtbench
seed** gets gsm8k/sharegpt/humaneval (grid convention). All seeds
reported; no best-seed selection.

## 9. Statistics

Paired prompt-cluster bootstrap, 3000 reps, seed 20260723
(`bootstrap_eagle_tau.py`), Holm within each pre-registered family
across 4 datasets (`holm_adjust_bootstrap.py`). Families:
F1: GS+R5 AA-QAT vs GS+R5 PTQ (primary)
F2: GS+R5 AA-QAT vs GS+R5 convQAT(Q0h)
F3: LS+R5 AA-QAT vs LS+R5 PTQ
F4: GS AA-QAT vs GS convQAT(Q0h)
F5: LS+R5 AA-QAT vs LS+R5 convQAT(Q0h); exploratory: everything else.
n.s. is reported as n.s., never as equivalence.

## 10. Verifier-shape + SD-correctness policy

Teacher p_k is the corpus sequential-incremental W4A4 execution (same
as every prior arm). Before Phase 2: measure distribution-level
tree-vs-sequential divergence on this host
(`measure_tree_vs_seq_teacher.py`; TV bounds the alpha error for any
draft). If median TV is NOT ≪ per-depth (1−α)≈0.3–0.5, switch the
teacher to tree-shaped extraction before training. SD parity
(`check_verifier_correctness.py`): measured for baseline PTQ draft AND
final AA-QAT drafts. Known from prior-host studies: W4A4 tree-vs-AR
parity is ~0.2 prefix-match ALREADY AT BASELINE (A4 bin-flips,
draft-independent); the pass criterion here is therefore **parity
INVARIANCE**: AA-QAT must not degrade exact-match/prefix metrics
relative to the same-day baseline measurement (paired bootstrap CI
including 0 or favoring AA). Any DRAFT-DEPENDENT degradation ⇒ STOP.

## 11. Decision rule (study sections 28/32)

Per-case verdicts as pre-specified: A (AA>PTQ>conv at GS+R5) supports
the hypothesis; B (AA≈PTQ>conv) = QAT unnecessary but objective
misalignment confirmed; C (AA≈conv) = objective not the cause; D (LK
surrogate improves, tau doesn't) → weigh Q3/Q3b evidence; E (greedy >
LK on tau) → greedy surrogate finding. Deployment recipe changes only
on a significant GS+R5 AA-QAT > GS+R5 PTQ result (Holm-surviving on
≥2 datasets and positive mean4).

## 12. Diagnostics (logged, never optimized)

Panel at every eval step (all arms): per-depth alpha/KL/TV/top-1/
logq(t)/rank(t)/hybrid/greedy-CE + SmoothL1/cos/NMSE vs a_chain
targets + expected_tau + greedy survival. Post-hoc from cadence ckpts:
W4 code-flip fraction, per-module ||ΔW||_F/||W||_F, clip-scale drift
(vs step-0), esp. W_first/W_rec/o/down. RCAL on final arms (mtbench,
FP16 reference lockstep). Free-running vs teacher-forced diagnostic
(section 17) on the locked objective only.
