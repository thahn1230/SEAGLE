# PAPER_CLAIMS — what this study licenses us to say, and what it does not

Run: `runs/eagle1_draft_aware_r1_r2_gs_20260810_065036`. Every claim below is
tied to an artifact in that directory. Terminology: GS = Global Scaling
(legacy EP3-G); R_{1,T}/R_{1,D} = target/draft-aware residual rotation;
R_{2,B}/R_{2,D} = baseline/learned draft attention rotation.

## Claims we CAN make

**C1. Draft-aware residual rotation R_{1,D} transfers to the GS scaling
policy.** Under Global Scaling with the canonical T4 beta_GS = 0.42 and a
freshly re-optimized R_{1,T}, learning a draft-specific residual basis with
all model weights frozen raises official micro-tau on all four confirmatory
datasets: +0.163 / +0.218 / +0.114 / +0.128 (MT-Bench / GSM8K / ShareGPT /
HumanEval), every 95% CI excluding zero and every test surviving Holm
correction within its pre-registered family (F1). Evidence:
`stats/bootstrap_pairs_*.json`, `stats/holm_adjusted.json`.
*Why this is new:* the prior grid measured R_D only under Local Scaling, so
the GS + R_D cell had never been run.

**C2. Learning the attention-local R_{2,D} adds no measurable benefit once
R_{1,D} is present.** Family F3 (A3 vs A1) is non-significant on all four
datasets after Holm (+0.071 / -0.017 / +0.050 / +0.004; best raw p 0.041 →
Holm 0.163). A pre-declared seed-matched secondary analysis agrees and is
stronger: averaging A3_sX - A1_sX within seeds gives +0.014 / +0.002 /
-0.006 / +0.023, all CIs including zero, with the per-seed sign flipping.

**C3. R_{2,D} alone produces at most a small effect.** Family F2 (A2 vs A0)
is significant on 1 of 4 datasets after Holm (GSM8K +0.066, Holm p 0.011);
MT-Bench, ShareGPT and HumanEval are non-significant. The mean4 gain is
+0.036, roughly one fifth of R_{1,D}'s +0.156.

**C4. The residual-stream basis dominates the attention-local basis.** The
converse contrast (A3 vs A2: adding R_{1,D} on top of a learned R_{2,D}) is
significant on all four datasets (+0.196 / +0.136 / +0.131 / +0.124, all
p < 1/3000, exploratory family).

**C5. The two rotations address overlapping degrees of freedom — with
direct geometric evidence.** Trained alone, the R2 generator norm is
16.3-23.2 and the learned R_{2,D} rotates 124 of 128 planes by a median 79°.
Trained jointly with R_{1,D}, the same parameterization settles at 5.62-5.78
(median eigenangle 21°) across all three seeds. Once the residual basis can
adapt, four times less attention-local correction is requested. Evidence:
`geometry/r2_mechanism.json`, `tables/rotation_geometry.csv`.

**C6. Independently-optimized R_{1,D} and R_{2,D} do not compose.** Bolting
the median A2 R_{2,D} onto the median A1 R_{1,D} without joint fine-tuning
(arm A4) *lowers* mean4 to 3.4337 from A1's 3.4644 and degrades per-depth
full-vocabulary overlap markedly (0.396/0.396/0.326/0.277 vs
0.498/0.523/0.484/0.437). Each was optimized assuming the other sat at
baseline. Diagnostic arm, reported as such.

**C7. Neither rotation adds an inference operator.** The CUDA kernel sets
recorded by the profiler are identical across A0/A1/A2/A3 — the same 10
distinct kernels, with residual call-count differences (max 34 of ~3000)
tracking the number of generation cycles rather than any added op. Both
rotations are folded offline into existing weight buffers (F0). Evidence:
`runtime/operator_audit.json`, `tables/profiler__*.txt`.

**C8. The AL gains are largely reference-consistent, not verifier-specific
artifacts.** Same-proposal RCAL replay against an FP16 reference verifier
gives, for R_{1,D}, dAL_q +0.163 [+0.109, +0.219] with dRCAL +0.111
[+0.056, +0.164] — about 69% of the gain matched by increased agreement with
the FP16 reference, the remainder being verifier-specific acceptance. No arm
triggers the deceptive-AL flag. Evidence: `tables/rcal_metrics.json`,
`stats/rcal_bootstrap_mtbench.json`.

**C9. The gains are invisible to conventional quantization proxies.** W4
NMSE is unchanged to the fourth decimal at every quantized site across all
arms (e.g. v_proj 0.012085 → 0.012081 / 0.012103). A2 slightly improves the
A4 activation error at its own boundary (-3%) while A3 worsens it yet scores
higher AL. Acceptance does not track NMSE.


**C10. Our joint arm IS SpinQuant's own procedure, and its R2 component is
inert for the draft.** SpinQuant optimizes R1 and all 32 per-layer R2
matrices jointly in one optimizer against one loss
(`optimize_rotation.py:103-108`); there is no staged or alternating
schedule. Arm A3 is therefore SpinQuant's procedure with the objective
swapped from target cross-entropy to the draft's acceptance-aware
surrogate. Under that procedure the optimizer, free to move both rotations,
largely declines to move R2 (generator norm 5.62-5.78 jointly vs 16.3-23.2
alone) and A3 does not beat A1 on any dataset. Separately, the A4 failure to
compose is an independent confirmation that SpinQuant's joint-optimization
default is the correct one.

## Claims we must NOT make

**N1. NOT "R2 is unnecessary."** R_{2,D} was implemented correctly (six
executable gates, including bitwise trainer/runtime parity on four legs),
trained on a tuned learning rate selected on held-out validation loss only,
and evaluated at full confirmatory scale. It shows a small effect alone
(significant on GSM8K). The correct statement is narrower: *in a deployment
that already uses a draft-aware R1, learning R2 buys no measurable
additional acceptance.*

**N2. NOT "R1_D and R2_D lie in the same basin."** No landscape,
mode-connectivity, or interpolation evidence was collected. The redundancy
claim (C5) rests on generator norms and the null F3 result, which support
"overlapping degrees of freedom", not a basin statement.

**N3. NOT "the LK loss optimizes greedy AL."** It is an acceptance-aware
*surrogate* under our greedy evaluation; full-vocabulary overlap is not
mathematically identical to greedy acceptance length. The objective is not
novel — it is an LK-Loss-inspired hybrid reused for quantization-basis
optimization. The novelty under test is acceptance-aware optimization of the
draft quantization basis with all weights frozen.

**N4. NOT "quantization NMSE explains the gain."** Measured NMSE is flat
(C9); the mechanism is not resolved by these proxies.

**N5. NOT "zero runtime overhead" from timing similarity.** The operator
audit (C7) is what licenses the no-added-operator claim. Wall-clock numbers
in this study are fake-quant simulation on a shared 8-GPU host and were
partly contaminated by concurrent jobs; only the quiesced re-measurement is
quoted, and even that must not be read as INT4 deployment throughput.

**N6. NOT a deployment throughput claim of any kind.** All quantization here
is fake (quantize-dequantize, FP16 matmuls). No INT4 kernels were used.

**N7. NOT "the learned R2 direction is meaningfully better than a random
one."** Three random residual Cayley perturbations with generator Frobenius
norm matched *exactly* to the learned R_{2,D} give +0.014/+0.018/+0.027 on
MT-Bench versus the learned +0.080/+0.035/+0.038. The learned direction is
2.6x better on average, but the distributions overlap and the learned effect
is itself non-significant on MT-Bench. Note also that geodesic magnitude was
NOT matched by construction (only the generator norm was), although the
random draws happened to land within 0.7% of the learned geodesic.

**N8. NOT cross-server comparability.** R_{1,T} was re-optimized on this
host (PPL: FP16 6.9436, W4A4 7.1062 vs the previous server's 6.9452 /
6.9629), so absolute AL values are comparable *within* this study only.

## Scientific conclusion (one of the four pre-specified options)

> **"Draft-awareness is primarily required for R1; R2 can remain fixed."**

Chosen on the evidence: F1 significant 4/4, F2 significant 1/4, F3
significant 0/4 (and 0/4 under seed matching), while the converse contrast
A3 vs A2 is significant 4/4. The mechanism supports it: with R_{1,D} free,
the requested R2 correction shrinks four-fold.

This is the prompt's CASE 2 and is a publishable negative result: the
attention-local V/O basis transfers adequately from the target-style
construction, while the residual-stream basis governing the target-draft
interface and the recurrent trajectory does not.

**N9. NOT a claim about multi-layer drafts.** The target has 32 independent
per-layer R2 matrices; this draft has one decoder layer and a single R2
shared across all 32 heads. The structural argument for why R1 can absorb
R2's role here is interpretation, not measurement, and the result may not
transfer to a draft with more layers.
