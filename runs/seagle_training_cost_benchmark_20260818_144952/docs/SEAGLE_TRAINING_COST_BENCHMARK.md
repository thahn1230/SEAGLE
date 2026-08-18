# SEAGLE Training-Cost Benchmark: PTQ vs QAT vs RT

Run `runs/seagle_training_cost_benchmark_20260818_144952` ·
2026-08-18 · ONE RTX 4090 (GPU 7, contention-free, guarded) · repo
@f34b4a7 · identical env for all methods (no TF32/compile/fused-opt
anywhere; each method's canonical dtype policy; synchronized
wall-clock over complete optimizer steps; warmup 60 dropped; LK-family
300 measured steps (20-step integer-second markers, ±0.05 s/step
granularity), RT 500 per-step samples; 3 interleaved rounds; 16/16
jobs rc=0). Canonical contracts + exact commands:
tables/canonical_method_contract.csv, scripts/ in this run dir.

## Headline (VIEW A) — normalized single-GPU algorithmic cost

| Method | What is trained | Canonical steps | GPUs | sec/step (window-mean; interval†) | GPU-h pure loop | **GPU-h canonical (val-amortized)** |
|---|---|---:|---:|---|---:|---:|
| SEAGLE-PTQ | rotation R5 only (16.78M) | 3,000 | 1 | 1.756 [1.70,1.80] | 1.46 | **1.56** |
| SEAGLE-QAT (incremental) | draft core QAT (236.0M, R5 frozen) | 3,000 | 1 | 1.709 [1.70,1.75] | 1.42 | **1.52** |
| SEAGLE-RT (RT-A, cached teacher) | full draft from scratch (235.9M) | 41,685 | 1 | 0.224 [0.2247,0.2306]‡ | 2.59 | **2.60** |
| SEAGLE-RT (RT-B, hybrid teacher) | + online W4A4 teacher for 34.7% | 41,685 | 1 | 0.719 [0.6992,0.7381]‡ | 8.32 | 8.34 |

† LK-family intervals are RESOLUTION limits (20-step/integer-second
markers, ±0.05 s/step) — cross-round total-elapsed agreement is
≤0.3% (QAT 616/616/616 s), i.e. reproducibility is better than the
timer; bootstrap CIs over quantized samples degenerate and are NOT
reported as uncertainty. ‡ RT intervals are per-step bootstrap CIs
(true per-step timing). Val-amortization: canonical LK configs run
eval-every 100 — measured +6.6% (controlled valcad round, identical
final VAL state); RT validation is +0.15% (actual-run measured).
p5/p95 columns in benchmark_step_times.csv are order statistics
(≈min/max at n=15) — annotated, not used downstream. The R6
median-of-pooled artifact (31/19 s partial interval) is replaced by
the window-mean here. All three methods fit on one GPU; same physical
GPU for every job.

**The surprise result:** normalized canonical compute of full
native retraining is only **1.7× SEAGLE-PTQ** and **1.7× QAT-
incremental** — and **0.84×** QAT-total (RT-A; teacher-cache
generation reported separately per the benchmark contract) — or
**5.3×/5.5×/2.7×** when the teacher is computed inline (RT-B). The often-quoted ~33× gap of the completed
strict run (267.9 GPU-h) is NOT algorithmic: it is deployment
overhead — world-8 DDP with per-microbatch fp32 all-reduce on
P2P-less RTX-4090s plus grad-checkpoint recompute — quantified below.

## Rotation cost separate (§10/§20)

| Method | Rotation | Canonical steps | sec/step | Wall h | GPU-h |
|---|---|---:|---:|---:|---:|
| SEAGLE-PTQ | R5 (REQUIRED) | 3,000 | 1.750 | 1.46 | 1.46 |
| SEAGLE-PTQ | R6 (OPTIONAL/ablation — excluded from headline) | 3,000 | 1.632 | 1.36 | 1.36 |
| SEAGLE-QAT | R5 reused from PTQ (counted once) | 0 | — | 0 | 0 |
| SEAGLE-RT | none during core scratch training | 0 | — | 0 | 0 |

Target R_T (shared by ALL three): ~10 GPU-h EST — excluded from the
differential comparison, shown as shared target preparation.

| Cost view (A+B, canonical val-amortized) | Core GPU-h | R5 GPU-h | R6 | Total required |
|---|---:|---:|---:|---:|
| SEAGLE-PTQ | 0 (no draft training) | 1.56 | 0 | **1.56** |
| SEAGLE-QAT incremental | 1.52 | reused | 0 | **1.52** |
| SEAGLE-QAT from base ckpt | 1.52 | 1.56 | 0 | **3.08** |
| SEAGLE-RT (RT-A) | 2.60 | 0 | 0 | **2.60** |

Separate categories (§11): D calibration — α/GS grid ≈3.0 GPU-h
(historical, DOCUMENTED; shared by PTQ and QAT; analytic + eval, not
training). E teacher-cache generation (RT only) — 4.64 GPU-h
measured for the 65.3% cache (full-corpus equivalent ≈7.1). F
validation — QAT canonical cadence (eval-every 100) measured at
**+6.6%** step-time amortized (1.42 → 1.51 GPU-h with validation);
RT validation 0.05 h measured in the actual run. Setup/model-load
(excluded from sec/step, measured): LK jobs ~40 s; RT-A ~13 s; RT-B
~170 s (target build).

## Ratio table (§21)

| Ratio (canonical val-amortized) | RT-A based | RT-B based |
|---|---:|---:|
| SEAGLE-RT / SEAGLE-PTQ | **1.7** | 5.3 |
| SEAGLE-RT / SEAGLE-QAT incremental | **1.7** | 5.5 |
| SEAGLE-RT / SEAGLE-QAT total | **0.84** | 2.7 |
(pure-loop variants: 1.8/1.9/0.9 and 5.8/5.9/2.9 —
tables/cost_ratios.csv + cost_ratios_amended.csv)

## VIEW B — practical preparation cost (§23; not a compute-efficiency claim)

| Method | GPUs normally used | Prep wall time | GPU-h |
|---|---:|---:|---:|
| SEAGLE-PTQ (R5 + calib) | 1 | ~4.6 h | 4.56 |
| SEAGLE-QAT incremental | 1 | 1.52 h | 1.52 |
| SEAGLE-RT | 8 | 33.48 h (ACTUAL) | 267.9 (ACTUAL) |

## Historical cross-check (§22)

| Quantity | Bench projection | Historical actual | Δ | Reading |
|---|---:|---:|---:|---|
| QAT sec/step | 1.700 | 2.181 (AAQ logs 19,626 s / 3 / 3,000) | −22.0% | historical runs shared the box with 5-8 concurrent jobs; this bench is contention-free. Uniform shift across methods → direction-preserving |
| RT hybrid w1 sec/step | 0.726 | 0.953 (strict-RT preflight) | −23.8% | same contention effect (preflight ran during 8-way benching) |
| RT practical GPU-h | 8.41 (w1-hybrid normalized) | 267.9 (w8 actual) | ×31.9 | **parallelization overhead factor** of world-8 sync DDP on no-P2P 4090s (940 MB fp32 all-reduce per micro-batch, grad-ckpt recompute), NOT benchmark error and NOT algorithmic necessity |

The two −22/−24% deltas agree within 2 points → the short-benchmark
methodology reproduces historical throughput up to a disclosed,
uniform contention factor. The ×31.9 factor is the honest
reconciliation between "algorithmic" and "as-deployed" RT cost.

## §17 secondary normalization (why the naive expectation inverts)

| Method | examples/step | ≈tokens/step | sec/step | tokens/s | steps/s |
|---|---:|---:|---:|---:|---:|
| PTQ R5 | 32 windows | 1,536 | 1.750 | 878 | 0.57 |
| QAT | 32 windows | 1,536 | 1.700 | 904 | 0.59 |
| RT-A | 4 convs | 5,940 | 0.227 | 26,167 | 4.41 |

The LK-family step is a K=4 multi-depth chain over 32 windows with
STE-quantized weights re-quantized every step; the RT step is a
single teacher-forced forward/backward of a 1-layer draft on cached
features. RT's per-token cost is ~29× LOWER; its 13.9× step-count
disadvantage does not close that gap. **The expensive part of RT is
not the optimizer loop — it is the teacher (cache generation or
online W4A4 forwards) and the multi-GPU deployment inefficiency.**

## §25 answers

1. PTQ under identical 1-GPU conditions: **1.56 GPU-h** canonical
   (1.46 pure loop; R5) + 3.0 calibration (separate category).
2. QAT: **1.52 GPU-h** canonical incremental (1.42 pure loop;
   +6.6% measured validation cadence).
3. Strict RT: **2.60 GPU-h** (RT-A) / 8.34 (RT-B teacher-inclusive).
4. GPU-hours: table above.
5. R5: 1.56 h canonical / 1.46 pure (1 GPU, 3,000 steps, 1.756 s/step).
6. R6: 1.44 h canonical / 1.35 pure (measured for completeness).
7. R6 required by canonical PTQ: **NO** — optional ablation
   (canonical grid excluded it; headline charges R5 only).
8. Incremental QAT after PTQ: 1.52 GPU-h canonical.
9. Total QAT from original checkpoint: 3.08 GPU-h (+3.0 calib
   shared; R5 counted once).
10. RT vs PTQ: **1.7×** (RT-A) / 5.3× (RT-B) normalized; ~59×
    practical-vs-practical (267.9 vs 4.56).
11. RT vs QAT: **1.7×/0.84×** (incr/total, RT-A) — 5.5×/2.7× (RT-B);
    ~176× practical-vs-incremental.
12. Agreement with RT=267.9: the w1 projections do NOT directly
    extrapolate to w8 (nor should they); component-level
    cross-checks agree to −22/−24% (uniform contention), and the
    ×31.9 residual is identified as measured DDP/teacher deployment
    overhead — the methodology validates.

## §26 paper-safe statements

"Under a normalized identical single-GPU environment, full native
drafter retraining requires 2.6 GPU-hours of optimizer-side compute
(8.3 including inline teacher computation), compared with 1.6 for
SEAGLE-PTQ's rotation learning and 1.5 for SEAGLE-QAT's incremental
training (canonical validation cadence amortized) (3,000 steps each; RT 41,685 steps; all measured from ≥300
contention-free optimizer steps × 3 rounds on the same RTX 4090)."

"The gap arises from the required optimization budget and trainable
scope, rather than deliberately asymmetric hardware — and, for RT
specifically, the dominant *practical* costs are teacher-feature
production and multi-GPU synchronization overhead (measured ×31.9 on
this P2P-less host), not the draft optimizer updates themselves."

Fairness gates honored (§16): no RT-specific slowdowns; cached
teacher allowed (bit-exact, Gate P1 of the strict study) exactly as
PTQ/QAT use their precomputed LK teacher corpus; identical GPU, env
flags, and measurement contract for all methods.
