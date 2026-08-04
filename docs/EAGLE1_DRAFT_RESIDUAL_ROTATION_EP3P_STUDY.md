# Draft-Aware Residual Rotation R_D under the EP3-P Frontier (RQ1/RQ2)

**Run**: `runs/eagle1_draft_residual_rotation_ep3p_20260804_165317`
(pointer `runs/RDROT_RUN_DIR`) · Branch
`exp/eagle1-draft-residual-rotation` · Date 2026-08-04/05 · 8× RTX 4090.

## 0. Research questions and verdicts

- **RQ1 — Is the draft-aware learned residual rotation R_D actually
  different from the target SpinQuant rotation R_T?**
  **YES, massively — and the difference is functional, not
  statistical.** The learned R_D rotates ~3,900 of 4,096 eigenplanes
  of R_Tᵀ·R_D by more than 0.1 rad (median angle 1.32 rad, max 3.02;
  Frobenius ‖R_D−R_T‖_F ≈ 81, ‖A‖_F ≈ 195), yet every classical
  quantization statistic is unchanged (per-site W4 NMSE, kurtosis,
  absmax, activation absmax/kurtosis — §5). Random rotations of the
  SAME magnitude gain nothing (§4, RANDPERT). The difference that
  matters is the *learned direction*, which aligns the draft's
  quantization error with directions the verifier forgives. The good
  R_D is also **not unique**: independently trained seeds are as far
  from each other (median plane angle 0.72–1.16 rad) as from R_T,
  with equal gains — a degenerate family, not a single optimum.

- **RQ2 — Does learned R_D beat reusing R_T on AL and RCAL?**
  **AL: YES, significant everywhere.** Retrained-under-EP3-P R_D
  beats the shared-R_T EP3-P baseline on all four confirmatory
  datasets with 95% CIs excluding 0 (paired prompt-cluster bootstrap
  3000): mtbench **+0.128** [+0.055, +0.198] p=0.0007, gsm8k
  **+0.263** [+0.220, +0.305], sharegpt **+0.217** [+0.146, +0.298],
  humaneval **+0.127** [+0.076, +0.180] (all p≈0). Seed-consistent
  on mtbench: +0.128 / +0.135 / +0.142 (all SIG).
  **RCAL: positive but not significant at n=80** — dRCAL +0.046
  [−0.024, +0.113]. The deceptive-AL flag is NOT triggered, but the
  honest decomposition is: of the +0.128 mtbench AL gain, +0.046 is
  reference-consistent and +0.082 is spurious acceptance (the W4A4
  verifier accepting tokens the FP16 verifier would not); AFS drops
  0.8555 → 0.8324. Lockstep validity confirmed by offline replay
  (377/377 cycles equivalent).

## 1. Setting (operator spec §0, all enforced by asserts)

Frozen: target weights, R_T (`outputs/rotations/learned_chat_
w4a4kv16/R.bin`, key R1), R2 (V/O pair), R4 (down_proj Hadamard),
draft weights, EP3-P pathwise migration scales (m_first = D^0.40,
m_rec = D^0.45, D=4096). Trainable: **R_D only** — ResidualRotation
`R_D = R_T · Cayley(A)`, A = W−Wᵀ dense 4096², zero-init (step 0 ⇒
R_D ≡ R_T). W4/A4 official SpinQuant fake quant (per-out-channel SYM
W4 + MSE clip search; per-token ASYM A4 clip 1.0); draft KV16
(frontier contract). Cross-branch projection rotation Q: excluded.
The T→D interface fold stays pinned at R_T (`first_fold_R`); the
draft's return to the vocab basis is absorbed in the lm_head fold —
both inherited from the validated LK construction.

## 2. What is new vs the LK study (7/21)

LK proved independent R_D > shared R_T under the LEGACY fold (single
global alpha=32, t4kv4, micro-AL, no RCAL). This study closes the
gaps: (1) trainer extended with the **EP3-P pathwise fold**
(`ExactQuantizedRotationForward(alpha_rec_init=…)`) matching the
deploy adapter **bitwise** — Gate-P: weight max|Δ|=0.0 and
4-depth chain logits Δ=0.0, greedy tokens equal, for BOTH R_T and a
perturbed R_D leg (`gradchecks/gateP_ep3p_parity.json`); (2) teacher
and corpus regenerated on the frontier target **t4** (W4A4 KV16),
5,000 target-generated windows, 5 domains, train pools disjoint from
all eval pools; (3) evaluation with the official-tau + RCAL +
paired-bootstrap protocol (GQ-era); (4) a new **accsurv** objective
(LRGF soft prefix survival) added to the LK objective family;
(5) matched-norm random-rotation control.

## 3. Arms and screening (held-out c4 calib 20, official tau)

| arm | tau | | arm | tau |
|---|---|---|---|---|
| **RD_HYB_s2** | **3.3359** | | LKXFER | 3.2102 |
| RD_HYB_s0 | 3.2731 | | RD_ACCS_s0 | 3.1990 |
| RD_AUXG_s0 | 3.2702 | | RD_ACCS_s2 | 3.1794 |
| RD_EXPT_s0 | 3.2363 | | RD_ACCS_s1 | 3.1608 |
| RD_HYB_s1 | 3.2306 | | RANDPERT2 | 3.0627 |
| | | | RANDPERT | 3.0234 |
| | | | BASE_EP3P | 3.0187 |

Every trained arm (+0.14…+0.32) > legacy transfer (+0.19) > random
controls (≈0) > baseline. Objective ranking hybrid > exptau ≈ auxg >
accsurv reproduces LK's ordering (full-vocabulary distribution
matching first-order; the greedy-token survival surrogate is weaker
for the residual-stream rotation than it was for the projection
rotation in LRGF). Finalist selection: held-out tau only.

Training: 3,000 steps, batch 32, AdamW+cosine+warmup 100+clip 0.5,
lr 3e-4, K=4 teacher-forced, Gate-G orthogonality assert every step
(~1.5e-4 = R_T's own storage error). ~4.3 h/arm on one 4090.

## 4. Confirmatory results (official tau, paired bootstrap 3000)

| dataset (n) | BASE_EP3P | RD_HYB_s2 | Δ [95% CI] | p |
|---|---|---|---|---|
| mtbench (80) | 3.0728 | 3.2004 | **+0.1276** [+0.0554, +0.1980] | 0.0007 |
| gsm8k (200) | 3.4759 | 3.7390 | **+0.2631** [+0.2204, +0.3050] | 0.0000 |
| sharegpt (80) | 3.0942 | 3.3115 | **+0.2173** [+0.1457, +0.2976] | 0.0000 |
| humaneval (164) | 3.7286 | 3.8557 | **+0.1270** [+0.0757, +0.1799] | 0.0000 |

mtbench controls and seeds (vs BASE, same protocol): RD_HYB_s0
**+0.1421** [+0.0785, +0.2051]; RD_HYB_s1 **+0.1352** [+0.0684,
+0.1966]; LKXFER (legacy LK2_HYBRID_s2 transplanted onto the EP3-P
fold) **+0.0914** [+0.0303, +0.1513] p=0.002; RANDPERT −0.0324
[−0.0911, +0.0236] n.s.

Three findings: (a) retraining under the deployed EP3-P fold beats
transplanting the legacy rotation (+0.128 vs +0.091 on mtbench,
+0.126 on held-out); (b) the gain needs the learned direction —
matched-‖A‖ random rotations are exactly baseline; (c) BASE_EP3P
mtbench 3.0728 reproduces the GQ study's EP3-P FULL-deploy number to
4 decimals, validating protocol continuity.

## 5. RQ1 analysis (geometry/rq1_geometry.json)

Eigenangle spectrum of R_Tᵀ·R_D (finalist): median 1.32 rad,
n(>0.1)=3936, n(>0.5)=3308 of 4096 — a global, distributed re-basis,
max element change only ~0.10 (the local-residual regime LK showed
is required; unrestricted rotations collapse). Quantization-relevant
statistics in the R_T vs R_D gauge (official W4 quantizer):

| site | NMSE (R_T) | NMSE (R_D fin) | kurtosis | activation absmax/kurt |
|---|---|---|---|---|
| W_rec | 0.01879 | 0.01919 | 1.87 → 1.92 | 7.73/0.32 → 7.15/0.35 |
| W_first | 0.01931 | 0.01931 (pinned) | — | — |
| down | 0.01214 | 0.01218 | — | — |

Nothing moves (weight-quant NMSE even ticks up slightly), yet AL
improves — reproducing LK's central mechanism finding under EP3-P:
**the rotation aligns the draft's quantization error with the
acceptance metric, invisibly to NMSE/kurtosis proxies.** Learned
solutions are mutually distant (RD_HYB_s2 vs LK2 median angle 1.16;
vs RD_HYB_s0 0.72) — a flat basin of acceptance-aligned rotations.
fp16-gauge framing: with quantization off R_D is a pure gauge, so
all differences above live at the quantization boundary by
construction.

## 6. RCAL (mtbench, lockstep FP16 reference, n=80/80/40)

| method | AL_q | AL_0 | RCAL | SAL | LAL | AFS |
|---|---|---|---|---|---|---|
| VBASE_EP3P (80) | 2.0728 | 1.9744 | 1.7313 | 0.3416 | 0.2431 | 0.8555 |
| VRD_HYB_s2 (80) | 2.2004 | 2.0689 | 1.7770 | 0.4234 | 0.2920 | 0.8324 |
| VLKXFER (40) | 2.0698 | 1.9657 | 1.7028 | 0.3670 | 0.2629 | 0.8439 |

Paired (VBASE:VRD_HYB_s2): dAL_q **+0.1276 [+0.0558, +0.1959] SIG**;
dRCAL +0.0457 [−0.0242, +0.1134] (positive, n.s.); no deceptive-AL
flag. Honest split of the AL gain: ~36% reference-consistent, ~64%
spurious acceptance (quantized-verifier drift) — R_D helps the draft
match the DEPLOYED W4A4 verifier (which is what ships and what
latency sees), more than it recovers the FP16 verifier's decisions.
LKXFER's smaller AL gain is more reference-consistent in proportion
(dRCAL +0.0615 of dAL +0.0673) but neither is significant.
Reference-replay control: `tables/replay_equiv_VRD_HYB_s2.json`,
377/377 cycles equivalent=True. (RCAL proposal-only convention:
AL_q here ≈ tau − 1.)

## 7. Limitations

1. RCAL significance not reached at n=80 (CI half-width ~0.069 vs
   effect +0.046; ~5× prompts needed — left open).
2. Finalist single-seed at multi-dataset scale; 3-seed consistency
   shown on mtbench only. LKXFER RCAL still n=40.
3. Fake-quant AL only — no real-kernel/latency claim (real-INT4 RTN
   contract differs; see SEAGLE INT4 study).
4. accsurv underperformance is one configuration (lr/steps shared
   with hybrid); not a refutation of acceptance surrogates —
   exptau (2nd place) is also acceptance-aware.
5. EP3-P scales kept frozen per spec ("initially frozen"); joint
   R_D+scale learning untested.

## 8. Repro

```bash
# Gate-P parity (bitwise trainer==runtime under EP3-P fold)
CUDA_VISIBLE_DEVICES=6 python scripts/check_rd_ep3p_parity.py --run-dir $RD
# corpus (5 domains x 1000 windows, teacher t4, GPUs 1-5)
python scripts/build_target_generated_lk_corpus.py --teacher t4 --mode greedy \
  --n-windows 1000 --domains <dom> --tag rd_t4_greedy_<dom> --run-dir $RD --seed <1-5>
python scripts/_rd_merge_corpus.py $RD
# train (per arm)
python scripts/train_eagle_lk_rotation.py --run-dir $RD \
  --corpus $RD/manifests/lkcorpus__rd_t4_all.json --batch 32 --accum 1 \
  --steps 3000 --eval-every 500 --teacher t4 --rot residual --kv-bits 16 \
  --alpha-init $(python -c "print(4096**0.40)") \
  --alpha-rec-init $(python -c "print(4096**0.45)") \
  --objective hybrid --seed 2 --out $RD/rotations/RD_HYB_s2.pt
# eval / capture / bootstrap
python scripts/eval_eagle_acceptance_length.py --target int4 --draft-cfg rot_ep3p \
  --ckpt $RD/rotations/RD_HYB_s2.pt --alpha … --alpha-rec … --tag RD_HYB_s2 \
  --datasets mtbench --n-prompts 80 --run-dir $RD
python scripts/capture_eagle_proposal_cycles.py … --draft-cfg rot_ep3p …
python scripts/compute_eagle_rcal_metrics.py --run-dir $RD
python scripts/bootstrap_eagle_rcal.py --run-dir $RD --tags … --pairs …
python scripts/analyze_rd_rotation_geometry.py --run-dir $RD \
  --extra-ckpts <LK>/rotations/LK2_HYBRID_s2.pt
```

Artifacts: `tables/final_summary.json`, `tables/screen_heldout.json`,
`tables/rcal_metrics.json`, `stats/bootstrap_pairs_*.json`,
`stats/rcal_bootstrap_mtbench.json`, `geometry/rq1_geometry.json`,
`gradchecks/gateP_ep3p_parity.json`, `rotations/RD_*.pt` (+sha256),
`manifests/lkcorpus__rd_t4_*.json`, `logs/`.

## 9. Next steps

- RCAL power: n≈400 captures for a significant dRCAL, or accept the
  deployed-verifier framing (AL is the shipping metric).
- Joint R_D + EP3-P-scale learning (unfreeze m under trust region).
- Stack with LP3-QAT: LK showed rotation+frozen-weights and QAT are
  separate levers; R_D-then-QAT composition is the natural frontier.
- Real-kernel transfer: fold R_D into the INT4 TensorCore path
  (the RTN kernel quantizer contract differs — needs its own
  validation).
