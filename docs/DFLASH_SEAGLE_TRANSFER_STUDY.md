# SEAGLE → DFlash Transfer Study (2026-08-08)

**Question**: do SEAGLE's SpinQuant-aware speculative-decoding techniques
(interface folding, scale migration, draft-aware rotation) generalize from
EAGLE-1 to DFlash — i.e., to *target-hidden-conditioned speculative decoding*
in general?

**Answer (one line)**: the *interface-aware quantization* principle transfers
and is decisive; the specific EAGLE remedies do not transfer 1:1 — DFlash's
bottleneck is outlier-driven (not branch-imbalance-driven), its second
interface (persistent context K/V) dominates after W_c is fixed, and the
deployable remedy is **context rotation reusing the target's own R1**
(no training, +84 MiB weight views), not a draft-residual rotation.

Tags: [확보] = statistically significant + verified; [provisional] =
conditional; [미확인] = not measured.

## 0. Setup

- Target `meta-llama/Llama-3.1-8B-Instruct` (bf16, HF 4.57.3, sdpa);
  draft `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` @ d3af30d
  (5 Qwen3-style layers, target_layer_ids [1,8,15,22,29], **block_size 10**,
  mask 128002, shared target embed/lm_head, tie_word_embeddings=false).
- DFlash @ 94e4abc, branch `exp/dflash-spinquant-seagle-transfer`;
  SEAGLE @ 0f6966a (read-only module port: SpinQuant quantizers/rotation,
  scheduler, RCAL conventions). 8× RTX 4090.
- Rotations: learned R1/R2 via SpinQuant Cayley-SGD (100 steps, 4×4090,
  4h02m, w16a4kv16 config, loss 2.41→2.05) at
  `outputs/rotations/llama31_w16a4kv16/R.bin`; random-Hadamard R.bin for
  diagnostics (`llama31_hadamard`). R4 online Hadamard (14336 = 28×512).
- Quant policy (SEAGLE): weights per-channel sym RTN+MSE-clip, acts
  per-token asym; draft scope = fc + Q/K/V/O + MLP; embed/head/norms/RoPE/
  softmax/KV-cache bf16. All fake-quant.
- Metric: official DFlash acceptance length tau (accepted+1 per cycle),
  cycle-pooled primary; greedy; max_new 1024; datasets mtbench80(2-turn)/
  gsm8k200/humaneval164/sharegpt80 (SEAGLE frozen selection) + math500.
- Stats: same-prompt paired cluster bootstrap 3000 + Holm (122 comparisons,
  75 rejected at 0.05).

## 1. Gates (§32) — all PASS

A baseline reproduction (mtbench 3.862 / gsm8k 4.213 / humaneval 4.892 /
sharegpt 3.839 / math500 4.026; paper SGLang refs 4.32/4.91) ·
B SpinQuant target (PPL fp16 6.667 = rot 6.665; **H_i^rot = H_i·R1 verified
for every residual layer, relerr ≤ 0.023** — single global R1 measured, not
assumed; index-32 exclusion documented = post-final-norm γ_f artifact) ·
C explicit == folded (fp64 relerr < 1e-10; e2e agree; MP3 FP-invariance
< 1e-10 = Gate D) · E/F quantizer audits in every arm · G orthogonality
(learned R_C orth err 3.8e-6) · H hooked generate == upstream bitwise;
RCDraft identity parity 1.000 · I shared embed/head untouched (restore
transforms explicit; checksum via view-free design) · J RCAL replay
self-consistency · K block/layer-ids pinned per comparison.

## 2. RQ1 — target rotation breaks the DFlash interface [확보]

mtbench, draft fp16, learned R:

| interface | rot_fp16 | W8A8 | W4A4 |
|---|---|---|---|
| A1 naive (rotated H → stock W_c) | 1.000 | 1.000 | 1.000 |
| A2 explicit unrotation | 3.809 | 3.802 | 3.734 |
| A3 folded W_c | 3.807 | — | — |

- Naive = **total collapse to tau 1.0** (Δ +2.73…+2.81 vs A2, p=3e-4 all
  targets). Coordinate mismatch is fatal, exactly as in SEAGLE.
- Fold is lossless: A2 vs A3 Δ −0.002 n.s.; rotation itself is free:
  A0 vs A2_TR Δ −0.053 n.s. [확보]

## 3. RQ2 — W_c imbalance: EAGLE-style? NO; outlier-driven [확보]

Measured (24 calib prompts, §6 protocol): cross-source act RMS ratio only
**1.11×** (1.15→1.28 by depth) — the EAGLE e/h asymmetry does **not**
replicate. The dominant obstacle is outliers: original-basis absmax 322 vs
RMS 1.15, kurtosis ~6.8e4, concat-A4 zero-code **94.6%**; naive W4A4 fc
output NMSE 0.054. R1 rotation of the interface fixes exactly this: absmax
17.2, zero-code 57%, NMSE 0.020. Weight-side block imbalance exists (W_i
RMS ratio 2.33×, deeper sources smaller) and is what MP3 targets.

W_c is the dominant single component [확보]: one-component W4A4 (T16,
mtbench-40): **fc 1.94** ≪ v 3.04 < down 3.31 < k 3.32 < q 3.35 < gate 3.43
< up 3.43 < o 3.49 (fp16-draft ≈ 3.9; full draft naive 1.40).

## 4. RQ3 — multi-source P2/MP3 (fc-only ladder, T16, mtbench-40) [확보]

naive 1.941 → +MP3 2.358 → +P2 3.211 → +rot 3.381 → rot+MP3 3.453 →
rot+P2 3.491 → rot+P2+MP3 3.504 (rot+P2 vs +MP3 addition n.s.).

- MP3 (m_i = D^{β_i}, geomean gauge, coord-descent on proxy NMSE; orig m =
  [1.46,1.29,1.12,0.97,0.49]) helps +0.42 SIG but is the weakest fix —
  static per-source scales cannot fix dynamic outliers.
- P2 (per-source dynamic A4 scales) +1.27 SIG — strong and free.
- Interface rotation dominates (+1.44 SIG).
- **Full-draft interplay**: at T16, P2 +1.01; at rotated targets the folded
  interface already delivers the rotated basis and P2/MP3 add ~nothing
  (T4: −0.02/−0.05; n.s. after Holm). EAGLE's P3 story transfers only as
  "fix the interface basis", not as "migrate static scales". [확보]

## 5. RQ7-grid — 3×3 precision grid (4-dataset mean tau) [확보]

| T\D | D16 | D8 | D4 (best interface) |
|---|---|---|---|
| T16 (fp16) | 4.201 | 4.161 | 1.464 naive / 2.577 +P2 |
| T8 (W8A8) | 4.141 | 4.123 | 2.465 folded / **3.514 +P2+R_C(RC1)** |
| T4 (W4A4) | 3.862 | 3.848 | 2.246 folded / **3.167 +P2+R_C(RC1)** |

- **W8A8 draft is near-free at every target** (T16: −0.04) — a sharp
  contrast with EAGLE-1, where D8@T16 collapsed to 2.17. DFlash's
  cross-attention context injection is more robust to mild draft noise.
- W4A4 target costs −0.34 (D16 row); W8A8 target −0.06 [확보].
- Quantized D4 drafts are much stronger under rotated targets (2.25-2.47 vs
  1.46) — the SEAGLE "interface, not precision matching" effect reproduces.

## 6. RQ4/RQ5 — rotations: R_D infeasible-cheap, R_C wins, R_T-reuse suffices

- **RD3 proof** (manifests/ROTATION_BOUNDARY_ANALYSIS.md): with shared
  embed/head, R_D ≠ R_T cannot be zero-overhead — two boundary transforms
  are irreducible (residual identity path carries no draft weight). Costs:
  RD1 2 GEMMs/block; RD2 views 1.96 GiB (tie=false → both matrices). [확보]
- **Structural finding**: ANY draft-side rebasing (even R_D = R_T) needs
  ctx-specific K/V weight views (84 MiB) because shared k_proj/v_proj serve
  two differently-normalized branches (ctx has NO per-layer norm). [확보]
- **Context K/V is the second interface bottleneck** (§15 measured): H_t is
  one outlier-heavy distribution (absmax 25.8/RMS 0.94, A4 zero 83%);
  ctx-branch NMSE up to 10× the draft-branch; optimal scalar migration
  s_ctx ≈ 1.0 → K2 scaling rejected by measurement; outliers again. [확보]
- **R_C (context-only rotation, folded ctx views)** on the all-quantized-
  linears W4A4 draft + P2 at T4 (scope reminder: fc+Q/K/V/O+MLP quantized;
  embed/head/norms/RoPE/softmax/KV bf16): RC0 (identity) 2.094 → **RC2
  (learned, L0 γ=5, 300 steps) 3.072** on mtbench (+0.98 SIG); **4-ds mean
  2.239 → 3.143** (+0.90, every dataset SIG). That closes **55.7 %** of the
  gap to the fp16 draft (4-ds D16 3.862); mtbench-only closure is 55.3 %.
  [확보]
- **RC1 (reuse the target's own R1 as R_C, no training) = 3.167 4-ds**
  (mtbench 3.098 / gsm8k 3.092 / humaneval 3.400 / sharegpt 3.077) —
  **statistically tied with, and numerically above, the learned RC2 3.143**
  (RC1→RC2 Δ −0.018…−0.032, n.s. on every dataset after Holm). RC0→RC1
  +0.75…+1.12 SIG on all four. **Learned rotation adds nothing over reusing
  R_T at the context interface**; the deployable recipe needs no training.
  RC1 closes **57.2 %** of the fp16-draft gap (RC2 55.7 %). [확보]
- **Transfer to a W8A8 target** (matched arms, all folded+P2, 4-ds):
  no-rotation G_T8D4_p2 2.439 → RC1xT8 **3.514** / RC2xT8 3.514
  (+1.01…+1.22 SIG per dataset; RC1 vs RC2 n.s. again). That closes 63 % of
  the gap to the fp16-draft T8 row (4.141) — the recipe transfers across
  target precision and works better there. Note the RQ7 grid cell T8/D4
  (2.465) is a *different* arm (folded WITHOUT P2). [확보]
- RC0 == plain QLinear protocol (n.s.) — protocol parity validated. [확보]

## 7. RQ6 — persistent KV injection & block-position behavior

Survival curves (plots/survival_mtbench.*, tables/survival_mtbench.csv):
quantization suppresses the whole curve and the damage grows with block
position — P(L≥5) is 0.317 for the fp16 draft, 0.006 for the naive W4A4
draft, 0.074 with folded+P2 (RC0). R_C lifts every position and lifts deep
ones most: RC1 P(L≥5) 0.219 (3.0× RC0), P(L≥8) 0.069 (7.3× RC0), against
the T4 fp16-draft ceiling 0.297/0.140. So the correction is not a
first-position-only fix — it restores the tail, which is where DFlash's
parallel block earns its speedup. [확보] Ctx K/V error enters every draft layer through
the persistent cache; ctx-only W4A4 (smoke) alone halves tau (6.4→3.1-ish
sample). Per-position KL/TV logit decomposition: [미확인] (not measured;
survival + agreement only). Block-size ablation: B=8 inference on B=10
checkpoint: absolute tau −0.13 (T16) / −0.04 (RC2) SIG, accepted fraction
tau/B rises (0.386→0.467; 0.307→0.379) — quantization effects stable
across B. [확보]

## 8. RCAL (T4 arms, mtbench, exact linear-block replay) [확보]

| arm | AL_q | AL_0 | RCAL | SAL | LAL | AFS |
|---|---|---|---|---|---|---|
| G_T4D4 folded | 1.129 | 1.093 | 1.033 | 0.096 | 0.060 | 0.930 |
| RC2 | 2.072 | 1.992 | 1.860 | 0.212 | 0.132 | 0.915 |

RCAL ranking = AL ranking; ~85% of the R_C AL gain is reference-consistent
(RCAL +0.83 of +0.94 proposal-only); no deceptive-AL arm. Gate J
self-replay clean. (RCAL beyond T4×mtbench: [미확인].)

## 9. Runtime / folding (fake-quant timings — no real-kernel claims)

- Fake-quant dominates absolute times (T4 verify 104 ms vs fp16 21.7 ms) —
  ms/token here must NOT be read as deployment latency. [확보 for
  breakdown structure only]
- Real interface costs are small: embed restore 0.03-0.16 ms/block, head
  restore ~0.1 ms delta, extract+concat 0.03 ms; interface-only rotation at
  T16 (unrotated target) costs a runtime per-branch GEMM (F2).
- Folding classes (tables/folding_audit.json): R1→W_c fold F0; MP3 weight
  side F0; MP3 activation scale F2 (cannot pre-fold into target weights);
  R_C = F0 with +84 MiB ctx K/V views (or F2 GEMM); shared embed/head
  restores F1 (absorbable into +0.98 GiB draft-only views each); R_D
  boundaries NF (proof); R4 online Hadamard F2; hidden extraction F2
  inherent (present in FP16 DFlash too).

## 10. Final recommendation

**W8A8 target + W8A8 draft** if 8-bit suffices: 4.12-4.14 (4-ds),
essentially free. For a W4A4-linears draft: **folded interface + P2 +
context rotation reusing R_T (RC1)** — no training, +84 MiB ctx K/V views,
F0 deployment. Measured 4-ds tau: **3.167 at a W4A4 target, 3.514 at a
W8A8 target** (vs 2.239 / 2.439 without the context rotation); the learned
R_C is statistically tied everywhere, so training buys nothing.
Static MP3: not recommended
(dominated). Draft-residual R_D: not recommended for DFlash (RD3 cost
asymmetry; R_C achieves the learned-rotation benefit at 1/23 the memory).

## 11. Honesty record / limitations

- 3H-vs-5H feature-count ablation infeasible without the (unreleased)
  training recipe — replaced by branch-deletion proxy (wc_branch_stats:
  deletion NMSE rises with depth). [미확인 as an AL ablation]
- R_D was analyzed (proof + costs), **not trained**; no claim that R_D
  would or would not beat R_C in AL. L1/L2 objectives not run (L0 only);
  RC1≈RC2 makes further objective tuning low-value. K1 arm subsumed by
  per-token quantization granularity (documented).
- Per-position KL/TV, target-quality absolute metrics, real low-bit
  kernels: [미확인]. "Algorithmic portability established; real low-bit
  runtime integration remains future work."
- Incidents: none fatal — 114 scheduler jobs, 0 failures, 0 give-ups;
  one overnight idle gap (queue drained without auto-refill) fixed with a
  drain-watcher; Gate-B index-32 false alarm documented (γ_f fusion).

## 12. RQ7 verdict

SEAGLE's core is **not** EAGLE-specific. What generalizes: (1) target
rotation breaks any target-hidden-conditioned interface and folding repairs
it losslessly; (2) the quantization bottleneck concentrates at the
target-hidden interface projections; (3) interface-basis choice (rotation)
dominates static scale migration. What is DFlash-new: the second interface
(persistent ctx K/V through shared projections), the ctx-view requirement
for any rebasing, and the R_T-reuse context rotation as a training-free
deployable remedy.

Artifacts: runs/dflash_seagle_transfer_20260807_180238 (tables/, plots/,
cycles/, stats/, manifests/), rotations at
/home/thahn1230/dflash_workspace/outputs/rotations/.

## 13. §37 questions — direct answers

**Q1 How do the DFlash and EAGLE target-hidden interfaces differ?** EAGLE
concatenates one token embedding with one target hidden and feeds a single
autoregressive draft stream (`y = e W_e + h W_h`). DFlash concatenates FIVE
target residual hiddens (layers 1/8/15/22/29) through `fc` (4096×20480) +
RMSNorm into one context feature, which is injected as K/V into every draft
layer and *persists* in the draft KV cache; drafting is a parallel masked
block, not a rollout. So DFlash has two interfaces (fusion projection and
context K/V), and its second one has no EAGLE analogue. [확보]

**Q2 Is the SpinQuant coordinate mismatch a real problem here?** Yes,
maximally: tau 1.000 with the naive interface at every target precision;
+2.73…+2.81 recovered by unrotation/folding. [확보]

**Q3 Range/weight imbalance in W_c?** Activation-side: essentially none
(RMS ratio 1.11×). Weight-side: yes (W_i RMS ratio 2.33×, deeper sources
smaller). The damaging property is outliers, not imbalance. [확보]

**Q4 Does P2-like separate activation quantization help?** Yes, strongly
and for free at an unrotated interface (fc-only +1.27; full draft at T16
+1.01). Under a rotated/folded interface it is redundant (n.s.). [확보]

**Q5 Does Multi-P3 help?** Marginally: fc-only +0.42 SIG, weakest of the
three fixes, and it adds nothing on top of rotation+P2 (n.s.). Static
per-source scales cannot address dynamic outliers. [확보]

**Q6 Global vs source-wise scale?** Global scalar is pure gauge under
per-token A4 + per-row W4 (verified: FP-invariance < 1e-10, quantized codes
unchanged). Source-wise is the only meaningful variant, calibrated m =
[1.46,1.29,1.12,0.97,0.49] (original basis) / [1.32,1.03,1.31,0.88,0.64]
(rotated). [확보]

**Q7 Where does the bottleneck move after W_c?** To the persistent context
K/V path: after fixing fc (rot+P2 → 3.49 of ~3.9), the full W4A4 draft still
sits at 2.24 (4-ds) and only the context rotation recovers it to 3.17. Among
per-component sensitivities v_proj is the next worst (3.04). [확보]

**Q8 Does persistent KV injection amplify quantization error?** Yes: one
outlier-heavy H_t is quantized once and consumed by all five layers through
shared k/v_proj, with ctx-branch NMSE up to 10× the draft branch
(k_ctx 0.082-0.116 vs k_noise 0.009-0.109; v_ctx 0.211-0.330). Because the
context is cached, that error persists across drafting iterations. [확보]

**Q9 More target features better under quantization?** [미확인] — the 3-H
checkpoint and the training recipe are unreleased, so an AL-level 3H/5H
ablation is impossible. Proxy: branch-deletion NMSE grows with source depth
(wc_branch_stats.csv), i.e. deep sources carry more unique signal; nothing
in the measurements suggests fewer sources would quantize better.

**Q10 Does R_D ≠ R_T raise AL?** [미확인] — not trained. Rationale for not
spending it: the context rotation already captures the learned-rotation
benefit at 1/23 of the memory, and RC1 shows even learning is unnecessary
*there*. No claim either way about R_D's AL. [확보 for the cost analysis]

**Q11 What does R_D ≠ R_T cost given shared embed/head?** Two irreducible
boundary transforms (RD3 proof): either 2 fp32 4096² GEMMs per block
(RD1) or draft-only rotated embedding + head views at
128256×4096×2B each = **1.96 GiB** (tie_word_embeddings=false). [확보]

**Q12 Is R_C more deployment-friendly?** Yes: +84 MiB of ctx-specific K/V
views (23× cheaper), no boundary transforms, F0, and — with R_T reuse — no
training and no extra checkpoint. It delivers +0.93 (T4) / +1.08 (T8) 4-ds.
[확보]

**Q13 F0/F1/F2/NF classification?** tables/folding_audit.json: F0 = R1→W_c,
MP3 weight side, R_C via ctx views; F2 = MP3 activation scaling, R4 online
Hadamard, target-hidden extraction, interface-only rotation at an unrotated
target; F1 = shared embed/head restores (absorbable into +0.98 GiB views
each); NF = R_D boundary transforms. [확보]

**Q14 Precision-grid results?** §5 table. Draft W8A8 is near-free at all
targets; W4A4 target costs −0.34; naive W4A4 draft collapses; the best
W4A4-draft recipe reaches 3.167 (T4) / 3.514 (T8). [확보]

**Q15 How much of the AL gain is exact RCAL gain?** For R_C at T4/mtbench:
AL_q 2.072, RCAL 1.860 vs RC0-level baseline 1.033 — about 85 % of the
proposal-only gain is reference-consistent; AFS 0.92, no deceptive-AL
arm. [확보]

**Q16 Different runtime trade-offs from parallel drafting?** Yes in kind:
DFlash's draft cost is one parallel block forward (3.55 ms fp16) against a
much larger verify (21.7 ms), so interface transforms (0.03-0.16 ms) are
proportionally negligible — unlike EAGLE, where a per-cycle 4096² rotation
sat inside an autoregressive rollout. Absolute numbers here are fake-quant
and must not be read as deployment latency. [provisional]

**Q17 Is SEAGLE EAGLE-specific?** No. The principle — quantization damage
concentrates at the target-hidden interface and is fixed by getting the
interface *basis* right — transfers intact. The specific EAGLE mechanisms do
not: scale migration is marginal here, and DFlash contributes a new
interface (persistent context K/V) whose remedy (R_C := R_T, ctx views) is
DFlash-specific and training-free. [확보]

## Appendix A — open items closed (2026-08-08 evening,
runs/dflash_openitems_20260808_191739, 22 jobs)

**A.1 L1 objective (previously unrun).** A context rotation trained with
the adaptive CE+TV hybrid (L1, 300 steps, same calib cycles) reaches
mtbench 3.062 — *below* R_T reuse (3.098, Δ −0.036, p=0.002) and below the
L0-trained rotation (3.072). Triple null: neither L0 nor L1 training beats
reusing the target's own R1 at the context interface. The training-free
recommendation stands. [확보]

**A.2 Source-ablation proxy for the 3H/5H question (previously 미확인 at
the AL level).** Zeroing one source branch at eval (mtbench, Δτ vs its own
baseline):

| source (layer) | FP16 T16 (base 3.862) | W4A4+RC1 T4 (base 3.098) |
|---|---:|---:|
| H_1  | −0.049 | −0.144 |
| H_8  | −0.051 | −0.026 |
| H_15 | −0.430 | −0.540 |
| H_22 | −0.270 | −0.525 |
| H_29 | −0.839 | −0.652 |

Deep sources dominate in FP16 (H_29 −0.84); under quantization the
mid-deep sources (H_15/H_22) become relatively *more* critical, and no
source is close to free except H_8. Nothing here suggests fewer sources
would quantize better — consistent with 5H ≥ 3H persisting under W4A4,
though a true retrained-3H comparison remains future work. [확보 as proxy]

**A.3 RCAL for the recommended recipe on all four datasets (previously
T4×mtbench only).** RC1 (R_T reuse), proposal-only:

| ds | AL_q | AL_0 | RCAL | AFS |
|---|---:|---:|---:|---:|
| mtbench | 2.098 | 2.010 | 1.876 | 0.913 |
| gsm8k | 2.092 | 2.001 | 1.867 | 0.912 |
| humaneval | 2.400 | 2.327 | 2.188 | 0.926 |
| sharegpt | 2.077 | 1.991 | 1.853 | 0.911 |

AFS 0.91-0.93 uniformly; SAL ≈ 10.6 % of AL_q on every dataset; no
deceptive-AL behaviour anywhere. [확보]

**A.4 Block-size B=16 inference (B=10-trained checkpoint).** T16 fp16:
3.860 vs 3.862 (flat; accepted fraction drops 0.386→0.241). RC1: 2.933 vs
3.098 (−0.165, p=0.08 n.s.). B=10 remains the operating point; the
training/inference block mismatch caveat applies. [확보]

**A.5 Determinism check.** Re-running RC1 with cycle recording reproduces
all four dataset taus to 4 decimals (3.0983/3.0919/3.4003/3.0770). [확보]
