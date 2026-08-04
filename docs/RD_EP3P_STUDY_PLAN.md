# Draft Residual Rotation R_D under the EP3-P Frontier — Study Plan

Date 2026-08-04. Branch `exp/eagle1-draft-residual-rotation`.
Run dir: see `runs/RDROT_RUN_DIR`.

## Research questions (operator spec §0)

- **RQ1**: Is the draft-aware learned residual rotation R_D actually
  different from the target SpinQuant rotation R_T — and in what
  (quantization-relevant) sense?
- **RQ2**: Does R_D improve Acceptance Length AND RCAL over reusing
  R_T, under the current frontier contract?

## Primary setting (frozen contract)

Target weights, R_T, R2, R4, draft weights, EP3-P migration scales
(m_first = D^0.40, m_rec = D^0.45): all FROZEN. Trainable: R_D only
(ResidualRotation, R_D = R_T·Cayley(A), A skew, zero-init ⇒ R_D=R_T at
step 0). W4/A4 official fake quant on; draft KV16 (frontier contract,
GQ/LRGF); cross-branch projection rotation Q NOT used (spec exclusion).

## What LK already answered, and the gap this study fills

LK (7/21) showed R_D beats shared R_T (+0.1634 mean micro-AL, 5
datasets, CIs>0) — but under the LEGACY fold: single global alpha=32
(no EP3-P), t4kv4 teacher/deploy, micro-AL only, no RCAL (protocol
postdates LK), no acceptance-surrogate objective. GQ's LP3-RD arm
reused the LK rotation on the LP3 fold (T1 tau 3.1594 vs EP3-P
3.0517) but ran NO EP3-P+R_D arm and no paired bootstrap vs EP3-P.
This study retrains R_D inside the EP3-P exact path and evaluates with
the official-tau + RCAL + paired-bootstrap protocol.

## Infrastructure changes (this study)

1. `ExactQuantizedRotationForward`: `alpha_rec_init` — EP3-P pathwise
   fold matching the deploy adapter bitwise (e-slice divided in fold
   dtype BEFORE fp16 cast; recurrent e-slice × alpha_rec/alpha_first).
2. `lk_losses`: `teacher_token_logp` + `prefix_survival` (LRGF ACC
   surrogate, soft prefix survival).
3. `train_eagle_lk_rotation.py`: `--alpha-rec-init`, `--objective
   accsurv`.
4. `eval_eagle_acceptance_length.py` / `capture_eagle_proposal_cycles
   .py`: `--draft-cfg rot_ep3p` (R_D gauge + first_fold_R=R_T + EP3-P
   alphas).
5. `check_rd_ep3p_parity.py`: Gate P — trainer vs runtime bitwise
   parity under the EP3-P fold (RT + perturbed-R_D legs).

## Corpus

Target-generated (build_target_generated_lk_corpus), teacher **t4**
(rot full, w4a4, KV16 — frontier deploy target), greedy, 5 domains ×
1000 windows (wiki/c4/sharegpt/gsm8k/code), seeds 1-5, built in
parallel on GPUs 1-5; merged manifest `lkcorpus__rd_t4_all.json`.
Train pools disjoint from all eval/calib pools (Gate F, builder
guarantee).

## Arms

Training (all: residual Cayley, EP3-P alphas D^0.40/D^0.45, kv-bits
16, teacher t4, batch 32 accum 1, steps 3000, eval-every 500, AdamW +
cosine + warmup 100 + clip 0.5, lr 3e-4 — LK stage-2 protocol):

| arm | objective | seeds |
|---|---|---|
| RD_HYB | hybrid (adaptive KL/TV, LK primary) | 0,1,2 |
| RD_ACCS | accsurv (LRGF soft prefix survival) | 0,1,2 |
| RD_AUXG | hybrid + aux-greedy 0.3 (LK confirmatory best) | 0 |
| RD_EXPT | exptau | 0 |

No-training arms:
- **BASE**: shared R_T + EP3-P (d4p3_deploy, alphas D^0.40/0.45) —
  the baseline RQ2 compares against.
- **LKXFER**: LK2_HYBRID_s2 R_D (legacy-trained) deployed on the
  EP3-P fold (rot_ep3p) — transfer probe.
- **RANDPERT**: random skew A with ||A||_F matched to the finalist —
  control that gains need LEARNED direction, not any perturbation.

## Protocol

1. Gate P parity PASS required before training counts.
2. Screen: held-out AL (official tau), c4 calib pool 20 prompts, all
   trained ckpts + BASE + LKXFER. Selection by held-out tau only
   (never mtbench).
3. Confirm: top arm (+ ties) + BASE + LKXFER + RANDPERT on mtbench
   n=80 eval pool: official AL + RCAL captures (lockstep FP16
   reference, 2 GPUs) + `bootstrap_eagle_rcal.py` paired
   prompt-cluster 3000 reps: dAL_q, dRCAL, dAFS vs BASE, plus the
   deceptive-AL flag. Secondary datasets (gsm8k 200 / sharegpt 80) if
   wall-clock permits.
4. RQ1 analysis: rotation_geometry (geodesic/Frobenius/max-elem/orth),
   eigenangle spectrum of R_Tᵀ·R_D, per-weight W4 NMSE / kurtosis /
   clip-frac in R_T vs R_D gauge, draft activation absmax/kurtosis per
   channel in both gauges — plus the functional statement from RQ2
   (paired deltas). Gauge framing: in fp16 R_D is a gauge; every
   difference metric is reported at a quantization boundary.

## Success criteria

- RQ2 positive: finalist dAL_q > 0 AND dRCAL > 0 vs BASE with 95% CIs
  excluding 0 on mtbench, no deceptive-AL flag.
- RQ2 negative/null is also a publishable verdict (EP3-P may already
  capture what LK's R_D exploited in the legacy fold).
- RQ1: quantitative difference report + whether the LEARNED direction
  matters (RANDPERT control).

## GPU policy

GPU 0 excluded (operator). Corpus GPUs 1-5, gates GPU 6, smoke GPU 7;
training arms scheduled over 1-7. All long jobs `setsid nohup`
detached with logs under the run dir.
