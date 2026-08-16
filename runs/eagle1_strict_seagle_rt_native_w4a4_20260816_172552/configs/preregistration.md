# STRICT SEAGLE-RT — Preregistration (binding)

Branch `exp/eagle1-strict-seagle-rt-native-w4a4-target` @ base b2eb71e.
Run root `runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552`.
Written BEFORE training launch (2026-08-16). Supersedes the previous
ambiguous "SEAGLE-RT" mapping: the old artifact (original-interface
from-scratch anchor a7c6ccc8 + D4P3 deploy) is renamed
**"Original-interface scratch reference"** in all new documents.

## Definition (strict)
SEAGLE-RT = EAGLE-1 draft initialized from scratch (official recipe,
PyTorch-default init, target embedding copied+frozen) and fully trained
while directly consuming h_RT_native = a_t of the DEPLOYED W4A4
SpinQuant target (R.bin 4b7e91d2, RTN W4 sym MSE-clip, dyn per-token
asym A4, KV16). No restored/original-basis target feature anywhere in
training or eval (Gates C/D/E; runtime+static enforcement).
Contracts: tables/seagle_rt_interface_contract.csv,
tables/seagle_rt_training_basis_contract.csv.

## Training
scripts/strict_rt/train_eagle1_strict_rt.py — line-identical fork of
the validated scripts/train_eagle1_official_fp16.py except (1) teacher
= native W4A4 cache/online-hybrid (bit-parity gated), (2) ploss head =
deployed fused head W_lm·D_γ·R1 (native_head.pt, sha-pinned), (3)
speed-only, math-identical changes (offline teacher, grad-ckpt off if
benchmark-validated, ckpts on /data). Official constants TC asserted
identical. Corpus: official 68k tokenize (66,890 usable, 63,545/3,345
— identical counts to the validated run). 21 epochs, world 8,
bs1×accum4/rank. Seed 0 canonical (more seeds only if budget permits).

## Checkpoint selection (preregistered)
The canonical checkpoint is the **FINAL checkpoint** — the exact rule
of the validated official-recipe reproduction (official v1 has no
best-ckpt selection; _of_gate_chain.sh used fp16_fresh_final.pt).
Best-val (min val vloss, 200 test convs) is saved for reference only
and is NOT the headline unless the final catastrophically regresses
(divergence), in which case the deviation is reported per spec §13.
No final-test-dataset numbers are computed before the checkpoint is
frozen (Gate I/§15 of the amendment).

## Arms (Stage A + Stage B ladder)
All arms derive from the SAME selected SEAGLE-RT checkpoint (Gate K).
Target fixed: W4A4 SpinQuant (--target int4 eval build) for every arm.
- SRT_FP16       draft fp16, native interface, no adapter/fold/alpha
- SRT_W8A8_RTN   draft W8A8 RTN (per-ch sym MSE-clip W, per-token asym
                 A), stock modules, NO rotations, NO alpha (Gate L)
- SRT_W8A8_SQ    + learned draft rotation (Cayley R_D = R_init·C(A),
                 validated parameterization; FullRotation+QR forbidden)
- SRT_W4A4_RTN   draft W4A4 RTN, stock modules, nothing else (Gate L)
- SRT_W4A4_HAD   + fixed Hadamard rotation, FP-preserving fold, no
                 optimization
- SRT_W4A4_SQ    + learned draft rotation (as W8A8_SQ)
- SRT_RESCUE     (only if triggered per spec §26) + canonical SEAGLE
                 projection correction, validation-only selection
Component sensitivity (spec §20) and projection stats (§21) run on
fresh strict-RT tensors only if W4A4_RTN drops substantially.

## Evaluation
Canonical evaluator + manifests: mtbench:80, gsm8k:200, sharegpt:80,
humaneval:164, greedy, mc_sim_7b_63, max-new 128; micro-tau per
dataset; mean4 = arithmetic mean of the 4 dataset taus. Manifests
materialized via write_or_verify_manifest + Gate F no-overlap check.
Validation-only selection pool: c4:20 --pool calib (offset-500).
RT-CONTROL-original-interface = the validated original-interface
scratch anchor (a7c6ccc8): its restored-interface deployment on the
same W4A4 target is P1's comparator.

## Preregistered comparisons (paired prompt-cluster bootstrap,
10,000 reps final (3,000 pilots), two-sided p floored at 1/reps,
Holm across the 4 datasets within each family):
- P1 RT-CONTROL-original-interface (FP16 draft, restored interface)
     vs SRT_FP16
- P2 SRT_FP16 vs SRT_W8A8_RTN
- P3 SRT_W8A8_RTN vs SRT_W8A8_SQ
- P4 SRT_FP16 vs SRT_W4A4_RTN
- P5 SRT_W4A4_RTN vs SRT_W4A4_HAD
- P6 SRT_W4A4_HAD vs SRT_W4A4_SQ
- P7 SRT_W4A4_SQ vs SRT_RESCUE (only if rescue triggered)
Retention table: tau_method / tau_SRT_FP16.

## RCAL safeguard
capture_eagle_proposal_cycles + compute_eagle_rcal_metrics +
bootstrap_eagle_rcal (comma pairs!) for SRT_FP16, SRT_W8A8_RTN,
SRT_W4A4_RTN, SRT_W4A4_SQ, rescue if any. Any quantized arm that
appears to beat FP16 AL must pass the RCAL check before being claimed.

## Decision cases (spec §32)
- CASE A: SRT_FP16 high AND SRT_W4A4 high AND rescue adds nothing
  → SEAGLE projection-correction novelty weakened (state honestly).
- CASE B: SRT_FP16 high, W8A8 high, W4A4 collapses → native training
  solves basis semantics, not low-bit geometry.
- CASE C: W4A4 RTN collapses but HAD/SQ restores ≈all → remaining
  pathology is outlier geometry; alpha/scaling less necessary.
- CASE D: W4A4 stays materially below FP16 AND canonical SEAGLE
  correction significantly restores → strongest: SEAGLE complementary
  even after true RT.
- CASE E: RT-W4A4 > SEAGLE-PTQ/QAT → SEAGLE positioned as
  plug-and-play/low-compute; quantify quality-compute tradeoff.
"SRT_FP16 comparable to control" = P1 Holm-n.s. on ≥3 of 4 datasets
AND |Δmean4| ≤ 0.06 (the reproducibility band of the validated
original-interface reproduction, |Δ|=0.042 n.s.). "Collapses" =
retention < 60%. "Materially below" = retention < 90% with P4 Holm-sig.

## Compute accounting (spec §11/§24/§31)
GPU-hours = wall × GPUs from actual timestamps (manifests/
gpuhours_train.jsonl, cachegen logs, eval gen_seconds + 240s load).
Separate columns: (A) draft from-scratch training; (B) target rotation
(shared, existing ~10 GPU-h EST, reported separately); (C) evaluation;
(D) failed/debug; (E) draft rotation training (separate); cache
generation reported under data-generation, separate from (A).

## Speed amendment compliance
Offline native-feature cache 495 GiB budget (validation-first
admission, ~65% token coverage; §17 analytics: full=757.63 GiB >
free=528.73 GiB → budgeted hybrid). Bit-parity gate P1, one-step
loss/grad gate P2, no-restore gate NR must all PASS before launch.
Mode chosen by 200-step benchmark among {online, hybrid ckpt-on,
hybrid ckpt-off} + DDP world scaling; forbidden speedups list (§16)
honored — 21 epochs / 43.9k-step-equivalent schedule unchanged.
