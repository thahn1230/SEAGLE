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

FINAL (80 prompts × 128 tokens, greedy MT-bench,
`runs/bwal_final_20260715_0249` → `artifacts/.../final_matrix/`):

| AL | D16 | D8 | D4 |
|---|---|---|---|
| **T16** | 3.6276 | 2.1328 | 1.0500 |
| **T8** | 3.6076 | 3.4156 | 1.2747 |
| **T4** | 3.2791 | 2.9881 | 1.2443 |

Pilot (20×64) reproduced the same ordering at a lower scale (3.4158 stock;
128-token generations have longer easy continuations, raising all cells).

### 2. Is Target degradation monotonic FP16 → W8A8 → W4A4?

FINAL: at D16, yes and ordered — W8A8 is statistically indistinguishable
from fp16 (Δ = −0.020, 95% CI [−0.064, +0.023], inconclusive/near-
equivalent); W4A4 costs −0.349 [−0.439, −0.257]. At D8/D4 the raw "target
effect" is **not** monotonic (T8_D8 = 3.42 > T16_D8 = 2.13: +1.28 [+1.22,
+1.35]), but those columns confound target precision with the draft
interface basis mandated by the spec's architecture contracts (see Q5/Q13).

### 3. Is Draft degradation monotonic FP16 → W8A8 → W4A4?

FINAL: yes in every row, all significant. Magnitude depends strongly on the
row: D8 costs −1.495 [−1.568, −1.426] under T16 (identity/h_t interface)
but only −0.192 [−0.234, −0.150] under T8 and −0.291 [−0.369, −0.215] under
T4 (gamma_R1/a_t interface). D4 collapses AL to 1.05–1.27 in every row.

### 4. Which precision axis is more sensitive?

FINAL: the **draft axis**, by a wide margin at 4 bits (D4: −2.58 under T16 /
−2.33 under T8, vs T4 target: −0.349 at D16). W8A8 is near-free on both axes
when the interface is rotated (T8_D8 = 3.4156 = 94.2% of stock 3.6276).

### 5. Are Target and Draft effects additive / sub / super-additive?

FINAL: strongly **non-additive** (max |interaction| = 1.30 at n=80;
18/20 contrasts significant).
Dominant driver: the interface-basis contract — T16_Dq cells feed the
draft's first projection the *unrotated* h_t (identity mode), while Tq_Dq
cells feed rotated a_t, so draft activation quant is far more damaging in
the T16 row. This is an architecture-contract effect, not a pure precision
interaction (spec §2.3 mandates these pairings).

### 6. Which Target components cause AL loss?

ANSWERED (ablations n=20, stock draft, anchor 3.4158): **entirely the
transformer body.** Target embedding (3.38–3.46 across all six modes) and
target LM head (3.37–3.45) are free even at W4A4. Body: W8A16/W8A8/W16A8
free (3.39–3.43); W4A16 −0.31; W16A4 −0.17; W4A4 −0.41 (sub-additive;
weight-4-bit slightly heavier than act-4-bit). Cross-check: body-W4A4
(3.0079) equals the pilot matrix T4_D16 cell exactly.
Mechanism (fixed-tree grader): on identical trees the W4A4 target re-grades
only 25.9% of rounds differently with mean Δaccept −0.037; the live drop is
~10× larger ⇒ ~90% of the live target effect is **trajectory-driven** (the
degraded target commits different tokens, shifting the prefix distribution
the draft must continue), not grading-driven.

### 7. Which Draft components cause AL loss?

ANSWERED (ablations n=20, identity-mode adapter under stock target;
anchor 3.4158). Single-component AL:

| component | W8A16 | W16A8 | W8A8 | W4A16 | W16A4 | W4A4 |
|---|---|---|---|---|---|---|
| embed | 3.42 | 3.42 | 3.44 | 3.41 | 3.42 | 3.41 |
| head | 3.42 | 3.41 | 3.43 | 3.39 | 3.43 | 3.43 |
| AR decoder | 3.42 | 3.40 | 3.42 | 3.39 | 3.33 | 3.28 |
| recurrent proj | 3.40 | 3.26 | 3.27 | 2.61 | 2.20 | 1.97 |
| **first proj** | 3.40 | 2.17 | 2.10 | **1.28** | 1.72 | **1.04** |

Ordering: **first projection ≫ recurrent projection ≫ AR decoder ≈ head ≈
embed ≈ 0**. `draft_full__w4a4` (1.039) ≈ matrix T16_D4 (1.050) —
the full-draft collapse is fully explained by the two projections.

### 8. Why is the Projection Layer more sensitive than the AR decoder?

ANSWERED — with embed/head ablations now included, "Projection is the most
sensitive component" is licensed. Converging evidence:
(a) **Input dynamic range**: the projection consumes the concat [e|h] whose
h-slice, in identity mode, is the raw h_t with unsuppressed channel
outliers; per-token quantization of that input is what W16A8/W16A4 degrade
(3.40→2.17/1.72 with exact weights). The AR decoder operates after this
bottleneck on already-projected states and has R2/R4 rotations applied.
(b) **Weight dynamic range**: the concat weight's per-channel absmax is
dominated by the e-block (ratio 2.8–19.9×), and the identity-mode h-block
(0.262) is 4.3× larger than the folded gamma_R1 h-block (0.061) — 4-bit
weight grids waste range on e-columns, explaining W4A16 first = 1.28.
(c) **Single point of failure**: every draft state at every tree depth
passes through the projection once per token; the AR decoder's error is
partially absorbed by the head's argmax, but projection error compounds
through the recurrence (first → recurrent → …).

### 9. How much does the Draft embedding contribute?

ANSWERED: nothing measurable — 3.41–3.44 across all six modes (isolated
copy quantized; target storage untouched, verified by data_ptr audit).

### 10. How much does the Draft LM Head contribute?

ANSWERED: nothing measurable — 3.39–3.43 across all six modes, including
full W4A4 on the isolated scoring head (3.43).

### 11. Does LM-head quantization change the Draft tree without significantly changing hidden features?

ANSWERED (behaviorally): head-only quantization leaves hidden features
untouched by construction (it is applied only to the isolated scoring
head), and the resulting AL is statistically at stock for every mode — so
whatever ranking perturbations it introduces do not alter the accepted
tree in practice at 7B scale (top-10 candidate margins dominate 4-bit head
noise).

### 12. Does Projection quantization corrupt the complete recurrent trajectory?

ANSWERED: yes. Quantizing ONLY the first projection (one application per
cycle) already collapses AL to 1.04–2.17 depending on mode; quantizing only
the recurrent projection (applied at depths ≥1) costs less at equal mode
(1.97–3.40). Depth histograms (`analysis/depth_hist.csv`, fig 04) show D4
cells lose almost all depth-≥1 acceptances — once the first projected state
is wrong, deeper draft states are unrecoverable, i.e. the error propagates
through the whole recurrence rather than averaging out.

### 13. Is first-Projection extra damage caused by A4?

ANSWERED — **A4 is a large cause but NOT the sole one; under the identity
interface, W4 weights are even more damaging.** First-projection-only
splits (n=20, identity mode): W16A4 = 1.72 (activation-only damage) but
W4A16 = 1.28 (weight-only damage) — both catastrophic, weights worse. This
refines the prior pure-R1-architecture finding ("asymmetry is
activation-driven; W4A16 symmetric"): the result is **interface-dependent**.
In identity mode the h-block weights are 4.3× larger (0.262 vs 0.061
folded), so a 4-bit per-output-channel grid whose scale is set by the
dominant e-block (absmax 0.744) quantizes the h-block coarsely.
Cross-row activation evidence stands: an identical D8 draft loses 1.495 AL
consuming unrotated h_t vs 0.192 consuming rotated a_t (H7), and
branchwise scales recover A4 to 91% of stock (Q14).

### 14. Does branchwise concat quantization recover AL?

ANSWERED: **yes for activations, decisively.** Per-branch activation scales
[Q_e(e)|Q_h(h)] at A4 on both projections: 1.4589 (full-concat single
scale) → 3.1089 (branchwise), +1.650 [1.449, 1.872] — 91% of stock.
At W4A4 it helps but cannot rescue the 4-bit weights: 1.0415 → 1.2104
(+0.169 [0.135, 0.209]). Branch asymmetry confirmed: quantizing only the
e-branch at A4 is free (3.422), only the h-branch costs (3.145);
Δ(e-vs-h) ≈ +0.28 in both the fp16- and A8-other-branch variants.

### 15. Is gamma folding responsible for first-weight outliers?

ANSWERED (weight-channel capture, `distributions/weight_channel_absmax.npz`):
**No.** Folding D_γ·R1 *shrinks* the h-block per-channel absmax (identity
0.262 → gamma_R1 0.061; recurrent [W_e|W_h·R1] 0.037). The e-block dominates
per-channel absmax in every mode (0.744; e/h absmax ratio 2.8 identity,
12.2 gamma_R1, 19.9 recurrent) — confirming the prior finding that the
embedding slice, not gamma folding, sets the concat projection's weight
dynamic range.

### 16. Does recurrent error accumulate with depth?

ANSWERED: yes — two lines of evidence. (a) Depth histograms: quantized-draft
cells concentrate mass at accepted-depth 0 (D4 rows: >85% of cycles accept
zero draft tokens), while stock spreads to depth 4+. (b) Component split:
recurrent-only quantization (applied only at depths ≥1) still costs up to
−1.45 (W4A4), showing depth-≥1 states are corrupted by repeated projection
passes even when the first projected state is exact.

### 17. Does W8A8 recover most of the Draft accuracy?

FINAL: yes **when the interface is rotated**: T8_D8 = 3.4156 (94.7% of
T8_D16 = 3.6076). Under the T16 identity interface it recovers far less
(2.1328 = 58.8% of T16_D16).

### 18. Which component should remain FP16/8-bit in a mixed-precision policy?

ANSWERED: keep the **draft projections (first + recurrent) at ≥8 bits** —
they are the only components whose quantization below 8 bits collapses AL.
Everything else tolerates 4 bits: draft embed/head/AR decoder, target
embed/head, and (at ~0.4 AL cost) the target body. At 8 bits everything is
safe: the all-8-bit draft costs −0.19 under a rotated interface and the
W8A8 target is free.

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

### 25. Final recommended Target/Draft/component precision policy?

RECOMMENDATION (fake-quant AL evidence only; no latency claims):

1. **Target W8A8 (rotated, fused): adopt freely** — AL −0.020 (CI includes
   0), wikitext CE +0.003, verifier path-clean, fixed-tree grading 98.9%
   unchanged.
2. **Target W4A4: acceptable where memory dominates** — costs −0.35 AL
   (~10% of stock) driven by the body's trajectory shift; embedding/head
   need no protection. Label results path-sensitive (Q24).
3. **Draft: quantize everything EXCEPT the projections to 4 bits for
   free** (embed, scoring head, AR decoder ≤ −0.15). Keep both projections
   at ≥8 bits; W8A8 projections cost only −0.19 AL when the draft
   consumes a rotated interface (a_t basis).
4. **Never feed a quantized draft the unrotated h_t** (identity mode):
   the same W8A8 draft loses 1.50 vs 0.19 — interface basis is the single
   largest lever in the whole study.
5. If A4 activations on the projections are required, use **branchwise
   [e|h] scales** (recovers to 91% of stock at exact weights); 4-bit
   projection *weights* remain the hard floor — no measured mitigation
   rescues W4 projections (best 1.28).

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
