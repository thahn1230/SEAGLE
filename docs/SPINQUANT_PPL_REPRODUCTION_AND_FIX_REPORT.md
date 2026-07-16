# SpinQuant PPL Reproduction & Fix — FINAL REPORT

Status: **COMPLETE** (pending adversarial verification pass) — all phases
executed: contract audit, official reproduction (Gates A–D), 10-seed sweep,
chat rotation training (100 steps, official recipe, 4-GPU), recipe
ablations, corrected EAGLE matrix + fixed-tree grader + verifier
consistency. §11 layerwise localization SKIPPED by its own precondition
(the properly-configured learned W4A4 PPL, 6.9629, is not excessive).

Branch `exp/eagle1-spinquant-ppl-reproduction-fix`. Vendored SpinQuant
8f47aa3 (untouched), EAGLE 4a9cf3a. Official evaluator = torchrun ptq.py
under the pinned venv (transformers 4.44.2); "official contract" = the same
math in-process (`official_ppl.py`, GATE-B-verified).

Terms used exactly: **random-Hadamard control** (unlearned R1/R2 from seeded
Hadamard draws) vs **learned SpinQuant** (Cayley-optimized R1/R2). A
random-Hadamard RTN model is never labeled as the paper pipeline.

## Headline numbers so far (official evaluator, WikiText-2, seq 2048)

| config | PPL | CE | Δ CE vs own FP16 |
|---|---|---|---|
| base FP16 | 5.4697 (2× bit-identical) | 1.6990 | — |
| base rh0-RTN W4A4KV16 | 7.9412 | 2.0721 | +0.3730 |
| base learned-RTN W4A4KV16 (official artifact) | **6.1972** | 1.8241 | **+0.1250** |
| base learned-GPTQ W4A4KV16 | 6.8590 | 1.9256 | +0.2265 |
| base learned-GPTQ +act_order | 6.8337 | — | — |
| base learned-GPTQ no-clip | 7.8433 | — | — |
| chat FP16 | 6.9452 | 1.9380 | — |
| chat rh0-RTN W4A4KV16 (official ptq.py) | 10.6272 | 2.3634 | +0.4254 |
| chat rh0-RTN W4A4KV16 (in-process build) | 10.5254 | 2.3538 | +0.4158 |
| chat learned-RTN W4A4KV16 | **6.9629** | 1.9406 | **+0.0026** |
| chat learned-GPTQ W4A4KV16 | 8.4600 | 2.1353 | +0.1973 |
| chat learned-RTN W8A8KV16 | 6.9491 | 1.9386 | +0.0006 |

Paper anchors (base): FP16 ≈5.5 ✓ reproduced; learned-RTN ≈6.1 ✓ reproduced;
learned-GPTQ ≈5.9 ✗ not reproduced by the released code (see Q4/Q7).

## The 18 questions

### 1. Is the original PPL 10.3263 numerically reproducible?

YES — bit-exact. Rebuilding the grader-path W4A4 target and rerunning the
original `wikitext_ce` reproduces CE=2.33469 / PPL=10.3263 / n=32767
exactly (audit/current_per_chunk_losses__rh0_w4a4.csv), as does the FP16
baseline (1.92539 / 6.8578). The same model measured on the official
contract gives 10.5254.

### 2. Does the custom evaluator match the official SpinQuant evaluator?

YES on the same tokens (loss math exact): with aggregation compared on a
shared token tensor the CE difference is ≤1.0e-4 (chat 9.0e-5, base
1.0e-4). The custom evaluator's headline numbers differed from official
because of CONTRACT, not math: (a) it scored only the first 32,768 tokens
of the test set (chat effect −0.0111 CE, base −0.0097), (b) its tokenizer
prepended BOS (−0.0012). The EAGLE-vendored model implementation is
logit-equivalent to stock HF (CE Δ 5e-5). Corrected evaluator: the
official contract (`official_ppl.py`), verified against torchrun ptq.py
(Δ≈2e-4, dtype-limited). All new target-quality claims use it; old numbers
are relabeled "32k-prefix metric".

### 3. Does the repository reproduce the paper's base-model FP16 PPL?

YES: 5.4697 via official ptq.py, deterministic across reruns (bit-identical
logs), tokenizer LlamaTokenizerFast add_bos=False add_eos=False, 166×2048
tokens (tail truncated). GATE A pass.

### 4. Does it reproduce learned SpinQuant W4A4KV16 RTN/GPTQ within a reasonable range?

RTN: YES — 6.1972 vs paper ≈6.1, using the officially released
`7B_W4A4KV16_lr_1.5_seed_0/R.bin` (paper artifact) + released quantizer.
GPTQ: NO — 6.859 (shipped flags), 6.834 (+act_order), 7.843 (no clip), all
above the paper's ≈5.9 and above RTN. The README notes paper numbers come
from Meta's internal codebase; the released HF-code GPTQ leg does not
reproduce them while everything it shares with the RTN leg (rotation,
eval, act quant) does. GATE C: PASSED via RTN; GPTQ gap documented as a
release-code/version difference. The corrected pipeline therefore uses
**learned SpinQuant RTN** as its primary recipe.

### 5. How much of the discrepancy is caused by base versus chat model?

FP16 baselines: base 5.4697 vs chat 6.9452 (CE +0.2389) — the raw
"10.33 vs 5.9" comparison was invalid on this ground alone. Under
identical rh0-RTN quantization the chat model degrades slightly more:
ΔCE +0.4254 (chat) vs +0.3730 (base), i.e. chat-specific extra fragility
≈ +0.05 nats. Learned-rotation transfer will refine this (PENDING).

### 6. How much is caused by random versus learned rotation?

On base (only model with the official learned artifact so far): learned
−0.2480 nats vs random seed 0 (7.9412 → 6.1972). This is the single
largest identified factor. Chat-specific number PENDING training.

### 7. How much is caused by RTN versus GPTQ?

With the released code, GPTQ is WORSE than RTN (+0.10 nats on base with
shipped flags); the paper's GPTQ advantage (−0.03 vs RTN) is not
reproducible from the released code. RTN-vs-GPTQ is therefore NOT a
contributor to the project's gap (the project already used RTN).

### 8. Is the current in-process EAGLE target path faithful to official ptq_model?

YES — GATE D: all 451 transformed (post-fake-quant) weight matrices have
bit-identical SHA256 between the in-process build and official ptq_model
on the same checkpoint + R.bin; 0 quantizer-config differences; PPL on the
official contract 10.5254 vs 10.6272 (ΔCE 0.0096, <1%). Residual per-layer
activation drift (rel-L2 0.025→0.05 through the stack, max_abs equal to
quantization grid steps) is fp16 kernel noise between the two Llama
implementations amplified at A4 bucket boundaries — not a transform error.

### 9. Are R1, R2 and online Hadamard transformations applied correctly?

YES by construction-equality: bit-identical transformed weights across all
451 modules (embeddings, Q/K/V/O with R1/R2 folding, MLP, head) prove the
rotation application (including gamma fusion and head rotation) matches
official exactly. Online-Hadamard (R4) settings are identical in the
quantizer-config diff (0 differences).

### 10. Is activation clipping/asymmetry faithful to the paper?

The in-process activation quantizers carry identical configs to official
ptq_model (a_asym=True, per-token, a_clip_ratio=1.0, groupsize −1); the
official code applies no activation clip search at eval (ratio 1.0), and
the project matches it. (The paper text's clipping discussion concerns
weights — w_clip MSE search — which both pipelines apply identically.)

### 11. Is seed 0 an unusually poor Hadamard rotation?

NO. 10-seed sweep (official recipe + evaluator): chat 10.43–12.95
(median 11.04), seed 0 = 10.63 — BELOW median (slightly lucky).
Base 7.63–8.49 (median 7.83), seed 0 = 7.94 (typical).
Hypothesis C rejected.

### 12. What is the corrected W4A4KV16 PPL for Llama-2-7b-chat-hf?

**6.9629** (official evaluator; chat-specific learned SpinQuant rotation,
official 800-sample/100-step recipe reproduced on 4 GPUs with global batch
8, RTN + w_clip). GPTQ variant: 8.4600 (released-code GPTQ gap, as on
base). W8A8KV16 control: 6.9491.
CAVEAT (in-distribution): the rotation is Cayley-optimized on wikitext-2
TRAIN with W4A4 in the loop, so wikitext PPL is in-distribution for the
rotation; W4A16 under the same rotation scores BELOW FP16 (6.3034 vs
6.9452), a QAT-overfit signature. Out-of-distribution behavior is measured
by the corrected EAGLE matrix (Q14).

### 13. What is the corrected CE delta relative to chat FP16?

**+0.00255 nats/token** (ln(6.9629/6.9452)), vs +0.4254 for the
random-Hadamard control — the learned rotation removes ~99% of the
wikitext CE delta. Same caveat as Q12.

### 14. Does correcting the target pipeline change the 3×3 AL matrix?

**No — statistically unchanged.** Corrected (learned-rotation targets,
same learned R1 for the draft transforms, 80×128 greedy MT-bench):

| AL | D16 | D8 | D4 |
|---|---|---|---|
| T16 | 3.6427 | 2.1493 | 1.0455 |
| T8 | 3.6040 | 3.4113 | 1.2712 |
| T4 | 3.3394 | 2.9428 | 1.2557 |

Deltas vs the random-Hadamard control are within cross-run noise (stock
anchor moved +0.015 across runs/GPUs) except T4_D16 (+0.060, a small real
improvement at most); max |interaction| 1.3006 vs 1.3028; 18/20 contrasts
significant in both. **A ~3.7-PPL wikitext improvement in the W4A4 target
translated into ≤0.06 AL on MT-bench** — target-side PPL is not predictive
of EAGLE acceptance (fig 06), because the AL loss is trajectory-driven and
MT-bench is out-of-distribution for the wikitext-trained rotation.

### 15. Does it change the "generous versus degraded" conclusion?

Partially — it REFINES it. Fixed-tree grader with the learned W4A4 target
(189 rounds, recheck 0/189 flips again): unchanged 76.7% (vs 74.1%), mean
Δaccept −0.0265 (vs −0.037), and the INCREASE side flips from
degradation-driven to benign: BENIGN_GENEROSITY 0→11,
DEGRADATION_INDUCED_GENEROSITY 11→1 (rank-flips 9→6, flattening 2→2).
Decreases barely move (false rejections 16→15, alignment loss 11→9).
So with a properly learned rotation, residual AL increases are quality-
preserving agreement, not degradation artifacts — while the dominant live
effect remains the trajectory shift. (The learned-W4A4 grader wikitext CE
is 1.9137, below fp16's 1.9254 on the 32k-prefix metric — the QAT-overfit
signature again; recorded, not celebrated.)

### 16. Which old results remain valid?

- The 3×3 AL matrix and all component ablations (the AL side never claimed
  paper-quality SpinQuant; its quantized targets are now precisely labeled
  "random-Hadamard control, seed 0" — GATE D proved that pipeline faithful
  to the official transform for that configuration).
- The evaluator's CE math (GATE B) and every RELATIVE comparison computed
  on the same 32k-prefix metric (both ends biased identically).
- Prior anchors: verifier top-1 path sensitivity, forensics results —
  untouched by this study.

### 17. Which old results must be retracted or relabeled?

- RETRACT nothing; RELABEL: "W4A4 target PPL 10.3263" →
  "Llama-2-7b-chat, random-Hadamard seed 0, fake W4A4KV16, RTN + MSE clip,
  32k-prefix metric (official-contract equivalent 10.53)". It must never
  be compared against the paper's learned-SpinQuant 5.9–6.1 (different
  model, different rotation class, different metric window).
- Any prose implying the project ran "SpinQuant (paper)" quantization →
  "random-Hadamard control".

### 18. What target quantization configuration should future EAGLE experiments use?

**Chat-specific learned SpinQuant R1/R2 (`learned_chat_w4a4kv16`) + official
RTN + w_clip + asymmetric per-token activations, KV16, official-contract
evaluation** — staged in-repo and wired via `--rotation-kind`. Rationale:
- w_clip is non-negotiable (removing it: PPL 80.2).
- asymmetric activations beat symmetric (+0.056 CE if dropped).
- GPTQ NOT recommended: released-code GPTQ underperforms RTN on both
  models (release-code/paper gap, Q4/Q7).
- Retrain the rotation per quantization config (the W4A4-trained rotation
  is co-adapted: W16A4 under it costs +0.19 while full W4A4 costs +0.003).
- For AL work specifically, expectations must be set by the AL matrix, not
  PPL: the corrected target changes AL by ≤0.06 (Q14), so the projection-
  protection and interface-basis policies from the bitwidth-AL study are
  unchanged.

## Gate log

- **GATE A** PASS: base FP16 5.4697 ≈ paper 5.5, deterministic ×2.
- **GATE B** PASS: evaluator loss math exact (≤1e-4 shared-token);
  contract deviations quantified; official contract adopted.
- **GATE C** PASS-via-RTN: learned-RTN reproduces paper; released-code
  GPTQ gap documented (6.83–6.86 vs paper 5.9) — not project-caused.
- **GATE D** PASS: 451/451 weight hashes identical; 0 config diffs;
  PPL Δ<1% same-config; kernel-noise-only activation drift.

## Execution deviations

1. Physical GPU 7 absent (persists). Initially single-GPU (CVD=6).
2. User authorization (2026-07-15, verbatim in environment.txt) expanded
   usage to GPUs 0–5; all verified idle before use. Training on 0–3
   (nproc 4 × grad-accum 2 = official global batch 8), sweeps on 4/5,
   evals on 6.
3. Base rotation retraining was killed after measuring 29 min/step (~48 h
   single-GPU); superseded by the officially released rotation artifact
   (higher fidelity than any retrain).
4. A transient CUDA context was created on a non-permitted GPU during a
   1-second venv smoke test (no computation; recorded in environment.txt).
5. GPTQ paper number not reproducible from released code (Q4/Q7).
