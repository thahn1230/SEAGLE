# Bitwidth → Accepted-Length Component Causality — FINAL REPORT

Status: **IN PROGRESS** — pilot (20×64) + gates complete; final 80×128 matrix
running; component ablations pending. PRELIMINARY numbers are n=20 pilot
values; they will be replaced by the 80-prompt final values.

- Branch: `exp/eagle1-bitwidth-al-component-causality`
- Target: meta-llama/Llama-2-7b-chat-hf · Draft: yuhuili/EAGLE-llama2-chat-7B
- EAGLE v1 @4a9cf3a · SpinQuant @8f47aa3 · greedy MT-bench, mc_sim_7b_63 tree
- tau = accepted draft tokens + 1 bonus token; AL = mean tau per cycle,
  averaged per prompt then across prompts
- Quantization is **fake quant** (SpinQuant RTN + MSE clip weights,
  per-token asymmetric activations); no real-kernel latency claims anywhere.
- KV cache fp16 in every cell.

## Hardware deviation

Physical GPU 7 dropped off the bus mid-study (2026-07-15; `nvidia-smi` lists
indices 0–6 only). Per policy `CUDA_VISIBLE_DEVICES=6,7` was kept and all runs
executed serially on the remaining visible device (cuda:0 = physical GPU 6).
GPUs 0–5 were never touched. Separately, the shared root disk hit 0 B free
mid-pilot; `outputs/draft_ckpts`, `outputs/rotations` and the HF weight cache
were moved to `/data/thahn1230` with byte-verified rsync + symlinks (no
results lost; stock-group shard predates the incident and was preserved).

## Architecture contracts (per spec §2)

| cell | target | draft interface |
|---|---|---|
| T16_D16 | stock fp16 | stock EAGLE (native alias head / copied embed) |
| T8/T4_D16 | fused fake-W8A8/W4A4 (a_t = n·R1) | A-explicit bridge (R1ᵀ→γ) + original fp16 draft |
| T16_D8/D4 | **stock fp16 (untouched)** | concat-selective, first block **[W_e\|W_h] (identity mode, h_t input)**, recurrent [W_e\|W_h·R1], post-R1, head W_lm·R1 |
| T8/T4 × D8/D4 | fused quantized | concat-selective (gamma_R1 first mode, a_t input) |

Draft D8/D4 = fake W8A8/W4A4 on both pre-R projections + 7 AR decoder linears
(R2/R4 applied); draft embedding + scoring head fp16 and **isolated**
(ownership audit: `docs/TARGET_DRAFT_PARAMETER_OWNERSHIP_AUDIT.md`).

## The 25 questions (spec §16)

### 1. What are all nine Target × Draft AL values?

PRELIMINARY (pilot 20×64) — FINAL (80×128) pending:

| AL | D16 | D8 | D4 |
|---|---|---|---|
| **T16** | 3.4158 | 2.0037 | 1.0393 |
| **T8** | 3.3922 | 3.2425 | 1.2529 |
| **T4** | 3.0079 | 2.6915 | 1.2140 |

### 2. Is Target degradation monotonic FP16 → W8A8 → W4A4?

PRELIMINARY: at D16, yes but W8A8 is statistically indistinguishable from
fp16 (Δ = −0.024, 95% CI [−0.096, +0.043], inconclusive/near-equivalent);
W4A4 costs −0.408 [−0.564, −0.253]. At D8/D4 the "target effect" is **not**
monotonic in the raw table (T8_D8 > T16_D8), but those columns confound
target precision with the draft interface basis (see Q5/Q13).

### 3. Is Draft degradation monotonic FP16 → W8A8 → W4A4?

PRELIMINARY: yes in every row. Magnitude depends strongly on the row:
D8 costs −1.412 under T16 (identity/h_t interface) but only −0.150 under T8
(gamma_R1/a_t interface). D4 collapses AL to 1.03–1.25 in every row.

### 4. Which precision axis is more sensitive?

PRELIMINARY: the **draft axis**, by a wide margin at 4 bits (D4 ≈ −2.2 to
−2.4 vs T4 ≈ −0.4 at fixed other-axis-fp16). W8A8 is near-free on both axes
when the interface is rotated (T8_D8 = 3.24 vs 3.42 stock).

### 5. Are Target and Draft effects additive / sub / super-additive?

PRELIMINARY: strongly **non-additive** (max |interaction| = 1.26 at n=20).
Dominant driver: the interface-basis contract — T16_Dq cells feed the
draft's first projection the *unrotated* h_t (identity mode), while Tq_Dq
cells feed rotated a_t, so draft activation quant is far more damaging in
the T16 row. This is an architecture-contract effect, not a pure precision
interaction (spec §2.3 mandates these pairings).

### 6. Which Target components cause AL loss?

PENDING — component ablations (tbody / thead / tembed groups).
Fixed-tree evidence so far: on identical trees the W4A4 target re-grades
only 25.9% of rounds differently with mean Δaccept −0.037; live T4_D16 drop
is −0.408 ⇒ ~90% of the live target effect is **trajectory-driven** (the
degraded target commits different tokens, moving the prefix distribution),
not grading-driven.

### 7. Which Draft components cause AL loss?

PENDING — component ablations (draft group: embed / first / recurrent / AR /
head × {W8A16, W16A8, W8A8, W4A16, W16A4, W4A4}). Prior studies (carried
context): first-vs-recurrent W4A4 asymmetry is activation-driven (W4A16
symmetric); e-block dominates fc per-row absmax 100%.

### 8. Why is the Projection Layer more sensitive than the AR decoder?

PENDING (H1–H8 tests). Note: "Projection is the most sensitive among the
components measured so far" until embed/head ablations land.

### 9. How much does the Draft embedding contribute?  — PENDING (ablation)

### 10. How much does the Draft LM Head contribute?  — PENDING (ablation)

### 11. Does LM-head quantization change the Draft tree without significantly changing hidden features? — PENDING

### 12. Does Projection quantization corrupt the complete recurrent trajectory? — PENDING (depth-resolved accept + hidden-drift capture)

### 13. Is first-Projection extra damage caused by A4?

PRELIMINARY: pilot adds cross-row evidence: identical D8 draft loses 1.41 AL
when its first projection consumes unrotated h_t (T16 row) vs 0.15 when it
consumes rotated a_t (T8 row) — consistent with activation-outlier damage at
the first projection input (H7). Confirmatory W16A4/W4A16 splits PENDING.

### 14. Does branchwise concat quantization recover AL? — PENDING (branch group)

### 15. Is gamma folding responsible for first-weight outliers? — PENDING (distribution capture; prior study: e-block dominates absmax)

### 16. Does recurrent error accumulate with depth? — PENDING (accepted-depth histograms by cell exist; per-depth acceptance decay analysis to come)

### 17. Does W8A8 recover most of the Draft accuracy?

PRELIMINARY: yes **when the interface is rotated**: T8_D8 = 3.2425 (95.6% of
T8_D16). Under the T16 identity interface it recovers far less (2.0037 =
58.7% of T16_D16).

### 18. Which component should remain FP16/8-bit in a mixed-precision policy? — PENDING (needs Q6–Q14)

### 19. When Target AL increases: benign generosity / flattening / degradation / rank flip / path artifact?

ANSWERED (fixed-tree grader, 189 rounds, 12 prompts): for the W4A4 target,
increases are 11.6% of rounds — DEGRADATION_INDUCED_GENEROSITY 11,
ACCIDENTAL_RANK_FLIP 9, FLATTENING_GENEROSITY 2, BENIGN_GENEROSITY 0.
EXECUTION_PATH_ARTIFACT: 0 (fp16 recheck flips 0/189 → grader itself is
path-clean). W8A8: 98.9% unchanged (1 rank-flip increase, 1 stricter loss).

### 20. When Target AL decreases: moved away from Draft proposals or false rejections?

ANSWERED (same run): W4A4 decreases are 14.3% of rounds —
DEGRADED_FALSE_REJECTION 16 (top-1 still in tree; rejected anyway) vs
STRICTER_ALIGNMENT_LOSS 11 (top-1 left the proposed tree / tree mass fell).
So both mechanisms occur, false rejections slightly ahead.

### 21. Why did Stock and transformed FP EAGLE differ in AL (3.4158 vs 3.4116)?

ANSWERED (forensics, deterministic settings, 20 prompts): the transform is
**algebraically exact** — swapping only the two pre-R projections to fp32
reproduces stock AL exactly (3.4158, 0/20 prompts diverging). With fp16
projections the difference (3.4134 vs 3.4158 under deterministic flags) is
fp16 GEMM rounding at near-tie top-k boundaries: top-k sets 98.14% identical;
83.8% of top-k decisions certified stable (margin > 2·δ∞); 1/20 prompts
diverge.

### 22. Was the difference fixed or characterized as finite-precision instability?

ANSWERED: characterized (not "fixed" — no bug existed). No basis/concat/
gamma/dispatch/sharing error found (STOP GATE A pass). We do NOT claim
bitwise equivalence in fp16; certificates quantify the instability.

### 23. Which conclusions use fake quantization?

All AL numbers in this study (every matrix cell with T≠16 or D≠16, the
grader, ablations). Real-kernel speed is out of scope; prior real-INT4
backends (tinygemm W4A16, QuaRot W4A4) exist in the repo but were not used
here.

### 24. Which conclusions are path-confounded?

GATE B labeling: the **W4A4 target's own verification is partially
execution-path-sensitive** — cross-shape top-1 agreement 0.84–1.0 (mean
≈0.92; anchors 0.906–0.911 reproduced) vs 0.97–1.0 (W8A8) and 1.0 (fp16).
Therefore T4-row live AL values carry a path-sensitivity component and are
labeled accordingly. The fixed-tree grader results are NOT path-confounded
(0/189 recheck flips); W8A8-row results are effectively clean.

### 25. Final recommended Target/Draft/component precision policy? — PENDING (synthesis after ablations)

## Gate log

- **STOP GATE A (forensics)**: PASS — algebra exact; residual = fp16 rounding
  at near ties; certificates recorded (`equivalence_forensics/summary.json`).
- **STOP GATE B (path consistency)**: PASS WITH LABELS — T4-row live results
  labeled path-sensitive (see Q24); grader immune.

## Quality anchors

WikiText-2 token-level CE (32,767 tok): fp16 1.9254 (PPL 6.86); W8A8 1.9280
(PPL 6.88); W4A4 2.3347 (PPL 10.33 — RTN+clip fake quant without GPTQ, hence
above the SpinQuant-paper full-pipeline ballpark; used only as a relative
degradation measure).

## Deviations from spec

1. GPU 7 hardware loss → serial single-GPU execution (see above).
2. Root-disk 0 B incident mid-pilot → resume from t8 group; large caches
   relocated to /data (symlinked).
3. Phase 5 (embed/head isolated-config validation) executes immediately
   before Phase 9 ablations (same intent: catch config bugs before the big
   run) because the GPU was continuously occupied by gated phases 6–8.
4. The fixed-tree grader gained an fp16-recheck control + explicit
   EXECUTION_PATH_ARTIFACT category (consumes `path_artifact_top1_tol`)
   after a pre-run adversarial review found the spec-listed category was
   otherwise unreachable; 7 confirmed defects were fixed before first run
   (commit 629fe18).
