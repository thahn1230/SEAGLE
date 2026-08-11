# FIDI — Full-Interface Distribution & Intervention study (preregistration)

Registered: 2026-08-11 (before any FIDI measurement was run).
Run root: runs/dflash_full_interface_distribution_intervention_20260811_074001
Amendment to: VSQ study (runs/dflash_vanilla_spinquant_novelty_20260810_104957) — MERGED, not
a restart. No VSQ/DKVA/DFST canonical artifact is modified or overwritten. All FIDI outputs
live under this run root only.

## Central question (verbatim intent)

Because the DFlash draft was trained while repeatedly receiving target-derived context H_t,
did the draft weights themselves adapt to that unusual distribution in a way that becomes
quantization-unfriendly? And across the complete conditioning path
[H1..H29] -> W_c -> H_t -> K/V projection -> K/V cache, where exactly should rotation or
scale correction be inserted?

Outcomes A–E all remain admissible; no positive SEAGLE conclusion is presupposed.
CASE A/B/C/D classification (§31 of the directive) is decided only by the final tables.

## Four stages

- STAGE-1: target hiddens [H1,H8,H15,H22,H29] (tap indices confirmed from code before capture)
- STAGE-2: W_c / fc fusion weight (4096 x 20480) incl. per-source blocks
- STAGE-3: Z_t = fc(concat) pre-norm; H_t = hidden_norm(Z_t)
- STAGE-4: K/V representations actually written to the draft cache (write points confirmed
  from code audit; "cache" label used only for tensors actually stored)

## Configuration axis (paired trajectories, identical prompts, replay mode)

- R0: no rotation (FP reference + RTN quant reference)
- R1: target vanilla SpinQuant only — R1_T/R2_T = outputs/rotations/llama31_w4a4kv16_s1/R.bin
  (chain seed s1, frozen by VSQ Gate C); only mathematically mandatory interface folds;
  NO R_C / P2 / MP3 / new scales
- R2: R1 + draft vanilla SpinQuant — R1_D/R2_D = VSQ run rotations/draft/R1D_s1r1.pt.best;
  basis-correct boundary handling; still no R_C
- R3: DFlash-aware SEAGLE = current best validated config (exact W_c source-interface
  handling + R_C = R1_T reuse + ctx-specific folded K/V views + P2 iff validated)
- R4 (diagnostic only): learned R_C checkpoint if available

Rotation comparisons are labeled A (raw coordinate change) vs B (FP-function-preserving
reparameterization). Headline mechanistic claims use B only; FP-output parity is verified
before/after each fold and recorded (rel err gate <= 1e-3 in bf16, <=1e-5 in fp32 capture).

## Frozen analytic result — global scalar gauge invariance (I5)

Claim (to be verified empirically once, then the scalar-search arm is CLOSED):
under per-token absmax A4 activation quantization and per-row (per-output-channel) absmax W4
weight quantization, the transform H_t' = s·H_t, W_K' = W_K/s, W_V' = W_V/s (s>0) leaves all
integer codes bit-identical: per-token scale s_t -> s·s_t so codes(H_t') == codes(H_t);
per-row weight scale w_r -> w_r/s so codes(W') == codes(W). Relative NMSE and SQNR are
invariant; absolute MSE scales by s². Hence a uniform scalar is a pure quantization gauge
and MUST NOT consume GPU search. Empirical check: s in {0.25,0.5,1,2,4} -> code equality +
NMSE table (tables/scale_invariance_test.csv). NOT gauge-invariant (remain admissible):
branch-wise scalar (I6, changes intra-token relative magnitudes before shared per-token
scale), per-channel diagonal S (I7).

## Intervention ladder at each pathological point

I0 none -> I1 reuse R1_T -> I2 reuse R1_D -> I3 random/Hadamard (diagnostic) ->
I4 learned local rotation (only if I1/I2 fail) -> I5 (closed if gauge, see above) ->
I6 branch-wise scalar -> I7 per-channel diagonal (SmoothQuant-like).
Every intervention ships with its paired inverse/fold; FP parity verified.

## Selection & statistics rules (same as VSQ)

- Selection ONLY on gsm8kvalid validation AL / proxy NMSE. The four test sets
  (mtbench80/gsm8k200/humaneval164/sharegpt80, frozen checksums of the VSQ manifests) never
  select rotation/scale/checkpoint/method.
- 4-ds mean = arithmetic mean of dataset-level micro-tau; no cycle pooling.
- Paired bootstrap >= 3000 resamples + Holm; never report p=0.
- KV8/KV4 cache quantization is DIAGNOSTIC ONLY; deployment cache precision unchanged.
- Causal language rule (§27): with only Tier-1/2 evidence say "trained weights exhibit
  compensatory alignment", not "training caused it". Tier-3 requires the actual DFlash
  initialization checkpoint (draft was initialized from target weights per DFlash recipe —
  check exact init source before using causal phrasing).

## AL ladder (deployment-policy end-to-end)

A0 FP16 / A1 naive RTN W4A4 / A2 learned vanilla SpinQuant W4A4 (mandatory folds only)
/ A3 +best W_c-side correction / A4 +best H_t/context correction / A5 +draft-side Q/K/V/O
correction ONLY IF an independent draft-side pathology is confirmed. Location ablations
L0..L6 on gsm8kvalid.

Note: VSQ M-arms already realize part of this ladder (M0=A0, M1=A1, M3=A2, M5 variants ~ A4).
FIDI reuses those canonical numbers (no re-runs, no overwrites) and runs only the missing arms.

## Required outputs

tables/: four_stage_activation_stats.csv, target_hidden_stats.csv, wc_weight_stats.csv,
wc_aw_decomposition.csv, ht_stats.csv, ht_qparam_stats.csv, draft_weight_stats.csv,
draft_weight_quant_stats.csv, activation_weight_alignment.csv, kv_projection_stats.csv,
kv_cache_stats.csv, hypothetical_kv_quant.csv, qkvo_component_sensitivity.csv,
qkvo_aw_decomposition.csv, intervention_map.csv, scale_invariance_test.csv,
validation_al_ablation.csv, final_4dataset_al.csv, bootstrap.csv, holm.csv
figs/: FIG-A (H_i 3D), FIG-B (W_c), FIG-C (H_t pre/post R_C), FIG-D (K/V cache per layer),
activation_vs_weight_column_scatter (per draft layer K/V).

## Ops constraints

- Do not interrupt or restart currently valid VSQ jobs; reuse selected R1_T/R2_T/R1_D/R2_D.
- No duplicate rotation optimization.
- Capture batches sized to share GPUs with the VSQ queue via the same scheduler.
