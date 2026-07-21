# Phase A audit — how the previous R_D was actually trained

Study: EAGLE-1 Exact-Path LK-Loss Draft Rotation Re-evaluation
(branch `exp/eagle1-lk-exactpath-draft-rotation`, base commit 5b05438).
Audited code: `src/eagle_spinquant/draft_rotation.py`,
`scripts/train_eagle_draft_rotation.py`,
`scripts/build_rotation_training_cache.py` as of TLDR-KV4 commit 4967d8f.
Numerical evidence: `scripts/audit_previous_rd_training.py` →
`<run>/tables/previous_training_audit.csv`.

## Verdict (one paragraph)

The previous negative conclusion about independent draft rotations was
reached with a trainer whose forward path differed from the runtime in five
material ways: (1) decoder weights were quantized in the ORIGINAL basis while
runtime quantizes the R_D-conjugated weights — an in-code comment
(`draft_rotation.py:155-159`) explicitly acknowledged this proxy gap; the
resulting weight-space discrepancy on the real `o_proj` (NMSE 0.0297) is
**2.1× larger than the entire runtime quantization error it was supposed to
model** (0.0144); (2) the runtime applies draft R2 (V/O) and R4 (down-proj
Hadamard) before quantization — the trainer applied neither; (3) the weight
clip search used a 6-point L2 grid over [0.75, 1.0] instead of the official
SpinQuant MSE search (grid=100, maxshrink=0.8), which alone changes the
quantized weights by NMSE 0.021 and leaves 16% more quantization error;
(4) activation quantization granularity MATCHED (both whole-concat per-token
asymmetric — corrected during this audit; the branchwise quantizer in the
codebase was an H2 ablation only), but the trainer's custom quantizer
(continuous min offset) and the runtime's official ActQuantizer (integer
zero-point) disagree per-value by NMSE 0.075 on representative inputs —
about 2× the activation-quantization error itself (0.036–0.038) — while
producing similar overall error levels; (5) the teacher
was a top-64-renormalized, temperature-2 distribution over RAW dataset text
windows, evaluated by proxy top-1 — never the full-vocabulary deployed
distribution on model-generated states, and never runtime micro-AL.
Training was also short (400–500 steps, batch 8, single seed, Adam 2e-3
with no schedule/warmup/clipping) and rotation-only (no draft-weight
capacity control). **The previous "independent R_D is unnecessary"
conclusion is therefore provisional**, exactly as this study's charter
assumes.

## Q(W) vs Q(R_DᵀWR_D): the non-commutation counterexample

Toy 4×4 (from the audit script): with a Givens rotation R (angle 0.7) and

```
W = [[ 1.00,  0.02, -0.01,  0.005],
     [ 0.03, -1.20,  0.04,  0.02 ],
     [ 0.50,  0.50,  0.50,  0.50 ],
     [-0.01,  0.90, -0.02,  0.01 ]]
```

per-channel symmetric 4-bit RTN gives

```
max | Q(W)·R  −  Q(W·R) |  =  0.0501     (weight scale ≈ 0.16)
```

i.e. roughly a third of a quantization step is lost by quantizing in the
wrong basis, per element, on a benign matrix. Quantization is a per-channel
scaled rounding lattice; an orthogonal change of basis does not map lattice
points to lattice points, so Q(·) and right/left rotation cannot commute
except in measure-zero cases.

Real-model magnitude (draft layer 0 `o_proj`, the actual learned R_T):

| quantity | NMSE vs the runtime-exact reference |
|---|---|
| runtime path `Q(R_TᵀWR_T)` vs unquantized `R_TᵀWR_T` | 0.01437 |
| trainer path `R_Tᵀ·Q(W)·R_T` vs runtime path | **0.02972** |

The trainer's decoder forward therefore disagreed with the runtime decoder
by *more than the quantization noise it was trying to optimize through*.

## Property-by-property comparison

| Property | Previous training | Runtime evaluation | Match? |
|---|---|---|---|
| Trainable parameters | R_D (4096²) + optional log-α (never enabled in saved ckpts) | none | — |
| Frozen parameters | draft fc/decoder/head/embed, R_T, γ | same weights | MATCH |
| Training steps | 400 (wave1/2), 500 (mixed) | — | SHORT (spec ref ≥1000–3000) |
| Batch size | 8 windows, no accumulation | — | SMALL (spec ref eff. ≥32) |
| Learning rate | Adam 2e-3, no warmup/schedule/grad-clip | — | UNVALIDATED |
| Calibration windows | ~1050 per teacher | — | PILOT SCALE |
| Window length | T=48 prefix | — | — |
| Draft depth K | 4 | tree depth ≤ 5 (mc_sim_7b_63) | ~ |
| Dataset mixture | raw text: wiki/c4/sharegpt/gsm8k/code | model-generated states | MISMATCH |
| Teacher-forced vs free-running | teacher-forced only | free-running | MISMATCH |
| Top-k truncation | top-64 renormalized, τ=2, τ²-scaled | full vocab, τ=1 | TRUNCATED |
| Objective coefficients | deployKL 1.0 / targetCE 0.5 / rank 0.3 / feature 0.5 / self 0.2 / fptarget 0.3 | n/a | — |
| Validation selection | proxy top-1 vs teacher top-1 | runtime micro-AL | PROXY ONLY |
| P3 handling | α folded exactly (e·α through W_e/α); α fixed 32.0 in all ckpts | folds ckpt α | PARTIAL (never co-optimized) |
| Quantization location (projections) | Q(folded W) ✓ | Q(folded W) | MATCH |
| Quantization location (decoder) | **Q(W) original basis, then conjugate** | **Q(R_DᵀWR_D)** | **MISMATCH (admitted in code)** |
| R2/R4 in AR path | absent | R2 on V/O, R4 on down_proj, then Q | MISMATCH |
| Weight clipping | 6-point L2 grid [0.75,1.0] | SpinQuant MSE grid=100 shrink=0.8 | MISMATCH |
| Activation quantization | whole-concat per-token asym (custom, continuous offset) | whole-concat per-token asym (official, integer zero-point) | GRANULARITY MATCH / impl values differ (NMSE 0.075) |
| Decoder execution shape | full-prefix re-forward per depth, no KV | incremental 1-token with KVCache | SHAPE MISMATCH |
| RoPE convention | interleaved even/odd pairs (`_rope`) | HF `rotate_half` (first-half/second-half) — different dimension pairing, different attention values | MISMATCH (found during this audit) |
| Draft KV4 (t4kv4 deploy) | never simulated | quantizes appended K/V | ABSENT |
| Train/eval quantization parity test | none | — | ABSENT |

## Which gaps are material (audit ranking)

1. **Decoder quantization basis** — proxy error 2.1× the modeled signal.
2. **R2/R4 absence** — the runtime quantizes different matrices entirely
   for V/O and down_proj.
3. **Top-64 τ=2 teacher** — LK/acceptance is a full-distribution overlap
   property; a renormalized top-64 at the wrong temperature optimizes a
   different functional.
4. **Clip-search mismatch** — 16% extra weight-quantization error and a
   different argmin, so the trainer optimized against the wrong lattice.
5. **Activation-quantizer implementation** — same granularity, but the
   trainer's continuous-offset quantizer and the runtime's integer-zero-point
   quantizer disagree per value by ~2× the quantization noise (NMSE 0.075
   between outputs), so the trainer optimized through a noticeably different
   noise realization than the one deployed.
6. Teacher-forced raw-text windows, proxy validation, short optimization,
   single seed, rotation-only capacity — all limit the strength of any
   negative conclusion.

## Consequence for this study

No new training may reuse `RotatedDraftTrainer._decoder` or its quantizers.
The replacement (`src/eagle_spinquant/exact_qat_rotated_draft.py`) must fold
R_D into every decoder linear (with R2/R4 where the runtime applies them),
quantize the FOLDED weights with the official SpinQuant quantizers, quantize
activations branchwise, execute the same tensor shapes as the runtime
incremental path, and pass the Gate D parity harness before any mandatory
candidate is trained.
