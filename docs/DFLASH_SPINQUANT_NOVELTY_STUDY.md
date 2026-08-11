# DFlash × SpinQuant Novelty Study (VSQ) + Full-Interface Distribution & Intervention Amendment (FIDI)

Branch `exp/dflash-vanilla-spinquant-novelty-grid`. VSQ run:
`runs/dflash_vanilla_spinquant_novelty_20260810_104957` (RD). FIDI run:
`runs/dflash_full_interface_distribution_intervention_20260811_074001` (FIDI).
Continued across a server migration on 2026-08-11 (handoff @ec1fd18; see
`HANDOFF_FOR_CLAUDE.md`, `RD/manifests/PROGRESS_NOTE.md`,
`RD/manifests/QAT_SELECTION_NOTE.md`). Companion adversarial self-audit:
`docs/DFLASH_SPINQUANT_NOVELTY_ADVERSARIAL_AUDIT.md`. Verifier:
`seagle_port/verify_dflash_spinquant_novelty.py`.

## 0. Question and verdict

**Question.** If one applies textbook ("vanilla") SpinQuant — learned R1/R2
rotations trained with the official pipeline, correctly folded — to both the
Llama-3.1-8B-Instruct target and the DFlash draft, does W4A4 quantization of
the DFlash system just work? Or does the DFlash architecture (a draft
conditioned each cycle on target-derived context `H_t = hidden_norm(W_c
concat(H1,H8,H15,H22,H29))`) create quantization pathology outside vanilla
SpinQuant's model-local closure that requires architecture-specific
correction — and if so, at exactly which interface should rotation or scale
be inserted?

**Verdict (CASE D, refined).** Vanilla SpinQuant does NOT solve the system:
correctly-folded vanilla SpinQuant reaches 4-ds mean AL **1.836** vs FP16
**4.202**. Three different corrections at three different interfaces are each
independently, significantly required:

| interface | correction | kind | 4-ds effect (paired bootstrap, all sig after Holm) |
|---|---|---|---|
| `H_t` → ctx K/V input | R_C = R1_T (context rotation, folded into ctx K/V views) | rotation | +1.31…+1.88 per ds (P4a) — dominant lever |
| `W_c` input (20480 concat) | P2 per-branch activation scales | scale granularity | +0.18…+0.52 on top of R_C (P3); **hurts alone** (−0.11 val) |
| draft weights | QAT (SpecForge-semantics, 400 steps) | weight adaptation | +0.10…+0.24 on top of P2+R_C (P5) |

Final system (M6 = vanilla SQ + R_C + P2 + QAT): 4-ds mean **3.823** = 84.0%
of the fp16→vanilla gap recovered ((3.823−1.836)/(4.202−1.836)); on HumanEval
the residual gap to FP16 is not statistically significant (M6 Δ−0.242,
p=0.068 raw; both the M5 and M6 humaneval FP16-gap cells fail Holm across
the 53-comparison family). The dominant single lever is the H_t rotation —
a genuinely DFlash-specific, cross-model-interface correction: FIDI shows the
W_c fold algebraically cancels R1_T so the fused-context loop sits outside
the model-local SpinQuant closure (deployed-forward H_t kurtosis 180–210 and
A4 NMSE 0.45 AFTER complete vanilla SpinQuant; 2.9 / 0.018 after R_C).
CASE E ("draft rotations can be replaced by target's") is rejected on direct
measurement (G1: −0.28 validation AL, significant). CASE A likewise (M3 ≪ M0).

## 1. Systems and protocol

- Target `meta-llama/Llama-3.1-8B-Instruct` (bf16 reference); draft
  `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` @d3af30d (5 layers, GQA 8KV,
  fc `W_c`∈R^[4096×20480], taps [1,8,15,22,29], block 10, bidirectional).
- W4A4 policy: weights per-row sym RTN (draft) / per-channel sym + MSE clip
  (target), activations per-token asym, KV cache bf16, embed/head/norms bf16.
  Target rotations: official SpinQuant `optimize_rotation` (w4a4kv16), 3
  seeds; chain uses s1 (sha16 f433a88b9cd40a4f; s2 nominal wikitext best by
  Δ0.007 = noise; **s1↔s2 deployed equivalence measured**: S2CHK Δ+0.065,
  p=0.244, n.s.).
- Eval: greedy, max_new 1024, B=10; datasets mtbench80(2-turn)/gsm8k200/
  humaneval164/sharegpt80; τ = official DFlash micro-tau (cycle-pooled);
  4-ds mean = arithmetic mean of dataset-level τ. Method/checkpoint selection
  ONLY on gsm8kvalid (n60, T512) or trainer-side held-out CE.
- Statistics: paired cluster bootstrap on prompts, n=3000, seed 0, two-sided
  p floored at 1/n (never 0), Holm step-down across all 53 comparisons
  (name-deduped; 43 rejected — the 10 non-rejections are the two designed
  equivalence checks, the two humaneval FP16-gap cells, three null
  component/I7 cells, and the three supplementary equivalence checks of §8).
  Tables `RD/stats/bootstrap_*.json`, `holm_adjusted.json`.

### Gates (all PASS)
- Gate C (target rotation quality, ×3 seeds): wikitext-16×2048 T-PPL fp16
  7.437 / RTN 189.9 / random-Hadamard 10.89 / R1-only 10.17–10.68 / R1+R2
  **8.79–8.85** (`RD/tables/target_ppl.csv`).
- Gate D (RotQuantDraft basis): R=I bits16 == stock (rel 5.8e-7); random
  rotation invariance rel 1.5e-6 (pre-pause, logged in PROGRESS_NOTE).
- Gate E (draft rotation quality, block CE): fp16 2.41 / RTN 7.23 / random
  4.83 / R1D 4.96 / R1D+R2D **4.47** (honest: R1_D alone < random)
  (`RD/tables/draft_quality.csv`).
- SpecForge parity: ported trainer forward bitwise-matches OnlineDFlashModel
  (`RD/tables/specforge_parity.json`).
- FIDI regression gate (this server, after instrumentation edits):
  old-vs-new RotQuantDraft forward **bitwise identical** (bits 4 and 16);
  fp_components=ALL == bits16 bitwise; capture taps bitwise-neutral
  (`FIDI/tables/fidi_gate_d.json`).
- FIDI R1f FP-parity (§3-B): rot_fp16 target + folded fc reproduces the fp16
  H_t (kurt 80.40 vs 80.37, rms/absmax equal) — folds are FP-preserving.
- Cross-server replication: M3cyc (fresh mtbench replay of the M3 config)
  reproduced the canonical M3 shard **exactly** (Δ=0.0000, identical
  per-prompt tau vectors); reconstructed `cache/sharegpt.jsonl` reproduced
  the canonical M0 sharegpt shard **bit-identically** (80/80 prompts,
  r=1.0000) — see QAT_SELECTION_NOTE for the reconstruction protocol.

## 2. Headline AL ladder (FIDI/tables/final_4dataset_al.csv)

| Method | MT | GSM8K | HumanEval | ShareGPT | 4-ds mean |
|---|---:|---:|---:|---:|---:|
| M0 FP16 | 3.8619 | 4.2129 | 4.8920 | 3.8391 | **4.2015** |
| M1 naive W4A4 RTN | 1.8740 | 1.6681 | 1.6435 | 1.6105 | 1.6990 |
| M2 vanilla SQ RAW (interface-broken) | 1.0000 | – | – | – | (diagnostic) |
| M3 vanilla SpinQuant (correct folds) | 1.6749 | 2.0050 | 2.0990 | 1.5643 | **1.8358** |
| M5a + R_C | 3.2030 | 3.3123 | 3.9836 | 3.0313 | 3.3825 |
| M5 + P2 + R_C | 3.4395 | 3.5510 | 4.5042 | 3.2089 | 3.6759 |
| **M6 + QAT (final)** | **3.5420** | **3.7877** | **4.6500** | **3.3121** | **3.8230** |

Notes. M3 ≈ M1 on mtbench/sharegpt — correctly-applied vanilla SpinQuant is
barely better than naive RTN end-to-end, because the binding constraint (H_t)
is untouched by it. M2 (RAW = model-local rotations without the mandatory
interface folds) collapses to τ=1.000: interface handling is a correctness
requirement, not an optimization (P6: M2→M3 +0.599, p=3.3e-4).

Preregistered comparisons (all per-dataset, paired; `RD/stats/`):
P1 M0→M3 −2.19…−2.79 (all sig) · P4a M3→M5a +1.31…+1.88 (all sig) ·
P4b M3→M5 +1.55…+2.41 (all sig) · P3 M5a→M5 +0.18…+0.52 (all sig) ·
P5 M5→M6 +0.102/+0.237/+0.146/+0.103 (all sig) · GAP M0 vs M6
−0.320/−0.425/−0.242(p=0.068, n.s.)/−0.527 · P6 M2→M3 +0.599 (sig) ·
S2 seed check +0.065 (n.s., desired) · replication Δ=0.0000.

RCAL (mtbench replay-based conservative AL; AL_q convention = τ−1): M5 AL_q
2.440 / RCAL 2.191 / AFS 0.915 (`RD/tables/rcal__M5_p2rc__mtbench.json`);
M3-config, regenerated cycles (M3cyc): AL_q 0.675 / RCAL 0.624 / AFS 0.934
(`RD/tables/rcal__M3mtbench.json`).

## 3. Validation ladder (gsm8kvalid; selection pool)

M-family: V_M3 2.0166 / +P2 1.9023 (**worse**, −0.114 sig) / +MP3 1.9225
(worse) / +R_C 3.2132 / +P2+R_C 3.4070 / G1 (R1_D:=R1_T) 1.7337 (−0.283 sig)
/ G1+RC 3.276 / R_D=I 1.134.
Q-family (QAT; RD/manifests/QAT_SELECTION_NOTE.md): pilots (Q2, 100 steps)
3e-3/1e-2 no-move, 3e-2 6.361, 1e-1 5.445 (prev server) ≈ 5.400 (this
server, regenerated hcache), **3e-1 5.218 best**, 1e0 diverged; mains
(400 steps, lr 3e-1) Q2 3.602 / Q3 3.479 / Q5 no-move from init 2.644;
per-arm amendment Q5@3e-2 → **2.389**. Deployed validation: V_Q2 2.4936 /
V_Q3 2.7424 / V_Q5 3.1479 / V_Q5b 3.5009 / **V_Q5bp2 3.6690 ← selected M6**
(caveat: P2 applied at eval only; disclosed).
Component arms (§16; deployed validation AL, vs V_M3 2.0166): restore-V
2.5637 (+0.547) > restore-K 2.2612 (+0.245) > restore-Q 2.0397 (+0.023,
p=0.036, Holm-n.s.) > restore-O 2.0298 (paired Δ+0.013, p=0.315, n.s.);
only-X arms (everything FP
except X): only-V 2.3141 (worst) < only-K 2.9212 < only-Q 3.7611 ≈ only-O
3.7784. Reading: V is the dominant fragile projection class, K second; Q/O
minor — fully concordant with the CE-proxy grid.
I7 SmoothQuant-style per-channel scale: proxy NMSE worse at every α in both
bases (`FIDI/tables/i7_smooth_proxy.csv`); deployed validation is decisive:
V_I7m3 pooled 1.1242 — WITHOUT R_C the scale is actively destructive (paired
Δ−0.892 vs V_M3, p=3.3e-4) — and V_I7m5 3.1865 on top of R_C is a null
(Δ−0.027, p=0.287). Scaling neither replaces nor augments rotation at H_t.

## 4. FIDI mechanistic findings (four stages, paired replay of one frozen fp16 trajectory)

Instrumentation: capture taps in the deployed RotQuantDraft forward
(regression-gated bitwise-neutral), 7 configs × {mtbench,gsm8k}:
R0 fp16 / R0q naive-W4A4 / R1f rot_fp16-parity / R1 vanilla-SQ (R_D=I) /
R2 +R1_D/R2_D (=M3) / R3 +P2+R_C (=M5) / R4 learned-R_C diagnostic.
`FIDI/tables/capture__*.json`, aggregated in `four_stage_activation_stats.csv`
etc. All replay the SAME fp16 trajectory `cyc__M0cyc__{ds}` (paired).

**S1 (H_i sources).** Model-local statistics healthy after R1_T; cross-source
imbalance mild (`cross_source_imbalance.csv`): rms max/min ≈ 1.6–1.9,
weight-side source-block imbalance 2.33× (fc_src1 rms 0.142 → fc_src29
0.061). Pathology is NOT source-scale mismatch alone → pure branch scalars
cannot fix it (and measurably don't: MP3/P2-alone hurt).

**S2 (W_c).** Weight-benign: kurt 3–5 (stock and folded), worst W4 NMSE 0.102
(stock RTN) / 0.047 (folded); SVD unremarkable (`draft_weight_stats.csv`,
`wc_weight_stats.csv`). §7 alignment: per-channel activation magnitude vs
W_c column norm anti-correlates (full-concat Spearman **−0.835**; per-branch
Pearson −0.14…−0.27 stock basis) — the trained W_c *exhibits compensatory
alignment* with target outlier channels (Tier-1 wording per §27; no training
trajectory available for causal claims). Error decomposition
(`wc_aw_decomposition.csv`): A 0.0174 / W 0.0350 / interaction 0.0008 —
small, additive; P2 trims the A term (0.0147).

**S3 (Z_t, H_t) — the core.** Deployed-forward measurements
(`ht_stats.csv`, `ht_qparam_stats.csv`, gsm8k):

| config | H_t kurtosis | A4 NMSE | code entropy (of 4 bits) |
|---|---:|---:|---:|
| R1 vanilla SQ complete | 180.5 | 0.4496 | 1.90 |
| R3 pre-R_C | 195.0 | 0.4463 | 1.87 |
| R3 post-R_C | **2.9** | **0.0182** | **3.21** |

(hcache counterfactual, `RD/tables/mechanism_stats.csv`: naive 69.5 =
vanilla-SQ-complete 69.4 vs R_C 3.0; NMSE 0.417/0.417/0.019 — same verdict
on independent data.) Z_t itself: kurt 635, absmax 1444 — but it is never
quantized (internal). Vanilla SpinQuant leaves H_t untouched because
`fold_wc` must cancel R1_T for FP correctness — the fused-context loop is
**outside the model-local rotation closure**. This is the architecture-specific
novelty locus.

**S4 (K/V cache).** Write-point audit (`FIDI/manifests/TENSOR_MAP.md`):
single cache write per layer (model.py:238), K stored post-k_norm+RoPE,
V stored raw; ONLY ctx entries persist (crop(start) discards the B
draft-token entries every cycle — "draft KV cache" is transient by
construction). Distributions: pre-norm ctx-K projection is wild (kurt 24,
absmax/rms ≈ 23) but the STORED cache is tame (K kurt 4–11, V kurt 3.6–5)
and low-bit-friendly: hypothetical KV4 NMSE 0.010–0.020, KV8 ≈ 5e-5
(`hypothetical_kv_quant.csv`). R_C changes stored-cache distributions only
marginally (R1 vs R3: K kurt 11.1→9.2, KV4 0.0201→0.0189) — **R_C fixes
projection-input quantization accuracy, not the cache distribution, which
was never the problem** (§13 A/B/C disentangled). KV-cache quantization is a
plausible future add-on (diagnostic only here, per §28).

**Draft weights (§14–§17).** All 92 audited weight views quant-benign
(`draft_weight_quant_stats.csv`); quadrant decomposition
(`qkvo_aw_decomposition.csv`): ctx K/V **activation-driven** in every layer
(A-only/W-only = 34×/27× for K/V), all other sites "mixed" with small
magnitudes, no interaction-driven site. Tier-1 compensatory alignment also
present at K/V ctx views (V Pearson ≈ −0.25 all layers, K ≈ −0.13); Tier-2
channel-ablation sensitivity tracks channel RMS (r 0.90–0.97) — outlier
channels dominate sensitivity, so FP-benign compensation fails under a
shared per-token A4 scale. Component sensitivity (CE proxy,
`qkvo_component_sensitivity.csv`): only-V +1.77 CE ≫ only-K +0.80 > only-fc
+0.40 > only-Q +0.30 ≈ only-O +0.28 > only-MLP +0.18; restoring fc alone
*hurts* (+0.37) — fc quantization acts as an implicit outlier smoother, an
interaction worth noting. Per-layer grid: V@0–2 dominant, no single
pathological layer. Conclusion: no additional draft-side transform is
warranted (A5 as a *new transform* is correctly absent); the validated
draft-side lever is QAT weight adaptation (M6), consistent with the
compensatory-alignment picture.

**Interventions (§10, §20).** I5 global scalar: proven a bit-exact
quantization gauge under per-token A4 + per-row W4 (50/50 cells code-equal,
output dev 0.0; `scale_invariance_test.csv`) — search arm closed by proof.
I6 branch scalar: at W_c input it is the P2/MP3 family (and under P2
granularity a per-branch scalar is again a gauge — algebraically explaining
why MP3 does not stack on P2); at H_t no branch structure exists. I7
per-channel scale: worse at every α (proxy) and confirmed no-gain in
deployed validation (above). Rotation is the unique effective correction at
H_t. Full placement map: `FIDI/tables/intervention_map.csv`.

## 5. Answers to the 23 required questions (§30)

1. **Are H_i low-bit-unfriendly?** Mildly; model-local R1_T/R2_T handles them
   (T-PPL 8.79 vs RTN 189.9). Not the bottleneck.
2. **How much does learned R1_T improve them?** RTN 189.9 → R1-only ~10.4 →
   +R2 8.79–8.85 T-PPL; source-level A4 diagnostics improve accordingly.
3. **Is W_c itself W4-unfriendly?** No — kurt 3–5, W4 NMSE ≤0.10 (RTN) and
   lower with MSE clip; W-only output NMSE 0.035.
4. **Did W_c learn compensatory alignment?** Tier-1 yes (Spearman −0.835
   full-concat; consistent per-branch negatives). Causal language withheld
   (no init/trajectory checkpoints — Tier-3/4 unavailable).
5. **W_c failure activation- or weight-driven?** Neither dominates and both
   are small at the fc product level (A 0.017 / W 0.035 / interaction
   0.0008); W_c's real harm is *downstream*, via the H_t distribution.
6. **Does vanilla SpinQuant already fix W_c?** The mandatory fold is a
   correctness requirement (M2 RAW τ=1.0); beyond that vanilla SQ neither
   helps nor hurts W_c specifically.
7. **Does W_c create a problematic Z_t/H_t?** Yes — Z_t kurt 635, H_t kurt
   180–210 with 26–33 absmax on unit-RMS scale.
8. **Is H_t still problematic after correct vanilla SQ?** Yes — kurtosis and
   A4 NMSE numerically unchanged (0.45); the fc fold cancels R1_T.
9. **Does H_t require an extra context rotation?** Yes — R_C delivers +1.31
   to +1.88 AL per dataset; ctx-A8 (HP2 3.415 vs M5a-mtbench 3.203) shows higher
   ctx precision alone also recovers it, confirming the locus.
10. **Could scalar/channel scaling replace R_C?** No — global scalar is a
    gauge (proof); per-channel scaling worsens proxy NMSE at every α and
    yields no deployed gain (V_I7m3/V_I7m5).
11. **Are draft Q/K/V/O weights quantization-unfriendly?** No — all views
    benign (max W4 NMSE 0.035 outside fc).
12. **Is V inherently more sensitive than K, and why?** Yes (only-V CE +1.77
    vs +0.80; restore-V −1.27; AL arms concur). V lacks any normalization
    between projection and storage/use (K passes k_norm), and ctx-V error is
    the largest activation-driven term (A-only 0.53).
13. **Do K/V weights show adaptation to target-context outliers?** Tier-1
    yes (V ≈ −0.25, K ≈ −0.13 column-norm anti-correlation), Tier-2 ablation
    consistent; phrased as "exhibit compensatory alignment" per §27.
14. **Which draft layers are most quantization-sensitive?** V@0–2 (CE proxy
    +0.40…+0.49); no single-layer pathology; depth trend mild.
15. **Are stored K/V cache tensors low-bit-friendly?** Yes — KV8 ~5e-5,
    KV4 0.010–0.020 NMSE.
16. **Does rotation improve the cache distribution or only projection
    accuracy?** Only projection accuracy (R1 vs R3 stored stats ~equal).
17. **Where should rotation be inserted?** Exactly one new place: H_t → ctx
    K/V input (R_C folded into ctx views), on top of standard R1_T/R2_T
    (target) and R1_D/R2_D (draft residual stream).
18. **Where is scaling useful?** Only P2 (per-branch A4 granularity at the
    W_c input), and only jointly with R_C.
19. **Which transforms are mathematically redundant?** Global scalar at H_t
    (gauge); per-branch scalars under P2 granularity (gauge → MP3+P2
    redundant); per-row weight rescales under per-row sym W4.
20. **How much AL does each location recover?** H_t/R_C +1.31…+1.88 (P4a);
    W_c/P2 +0.18…+0.52 (P3); draft-QAT +0.10…+0.24 (P5); total M3→M6
    +1.99 4-ds-mean points (84% of the M0 gap).
21. **After vanilla SpinQuant, what residual structural problem remains?**
    The fused-context interface: H_t enters ctx K/V with per-token A4 in a
    basis no model-local rotation reaches, with V the most exposed consumer.
22. **Any additional draft-side correction needed beyond H_t R_C?** No new
    transform; QAT weight adaptation is validated and complementary.
23. **True architecture-specific novelty, or does vanilla SpinQuant solve
    it?** Vanilla SpinQuant does not solve it (M3 1.836 ≈ M1 1.699); the
    required corrections are DFlash-interface-specific. CASE D.

## 6. Honest caveats

1. **M6's P2 is eval-composed**: Q5 was trained without fc_p2; composing P2
   at eval wins validation (+0.17 over V_Q5b) despite the train/eval
   granularity mismatch. RESOLVED post-hoc (§8): Q6 trained WITH P2 ties
   V_Q5bp2 on validation (Δ−0.041, n.s.) — the mismatch costs nothing
   measurable.
2. **Q5 LR is a per-arm amendment** (pilot LR was Q2-init-scaled; 3e-1/1e-1
   no-move from Q5's better init; 3e-2 selected on trainer-val CE only).
3. **sharegpt.jsonl reconstruction**: original byte checksum not reproduced
   (33 recipe variants tried); functional identity proven instead
   (bit-identical per-prompt fp16 eval, 80/80). Frozen-contract status:
   functionally intact, byte-level provenance replaced (new sha in
   QAT_SELECTION_NOTE).
4. **R0q captures** (naive-quant trajectory) show *lower* H_t kurtosis (24)
   than FP — quantization noise blurs outliers while destroying accuracy;
   distribution-shape comparisons across broken configs are interpretive,
   not evidential.
5. **Learned R_C (R4/RC_L0)** is carried as a diagnostic only (DFST-era
   checkpoint; not re-selected under the current W4A4 rotation chain).
6. **RCAL for M3** comes from regenerated cycles (M3cyc — which reproduced
   the canonical shard exactly), not the original pruned `cycles/` files.
7. **A5 disclaimer**: no new draft-side transform was added — §22's guard
   ("do not invent another method if Q/K/V/O weights are benign") is
   respected; M6's QAT is weight adaptation within the existing
   parameterization.
8. Remaining GPU-noise caveat: greedy evals proved bit-reproducible across
   servers here (M3cyc, M0shk), so cross-server drift is not a live concern
   for these tables.

## 7. Complexity/runtime accounting (CASE D honesty requirement)

R_C = R1_T reuse: zero new learned parameters. CORRECTED BY THE BASIS AUDIT
(`FIDI/tables/basis_audit.json`, verdict D): the deployed implementation is
class A — ctx K/V weight views are folded @R_C offline, but the activation
side is an EXPLICIT RUNTIME rotation `Ht = Ht @ R_C` ([S,4096]×[4096,4096]
dense, once per draft forward, shared by all 5 layers; measured 335 µs at
S=512 fp32 on a 4090; 16.8 MMAC/token ≈ 2× the ctx K+V projection FLOPs).
Any earlier "zero runtime cost" wording is retracted. A fully-folded form B
(W_c_B = R_C^T @ W_c_fold, exploiting n(zR)=n(z)R for bare RMS) is
FP-equivalent (verified to 3e-15) and would remove the runtime op, but it
moves the rotation across the W_c W4 quantization boundary (77% of integer
codes change) — adopting B would require re-validating the W_c quant arm;
all reported AL numbers used A consistently (proxy and eval share the code
path), so comparisons are apples-to-apples. P2: five per-branch scales per
token at the fc input (negligible). QAT: training-time cost only (~8 min ×
1 GPU at 400 steps); deployment weights unchanged in format. Memory: ctx
K/V views double k/v weight storage for the draft (2×1024×4096×2 tensors
per layer, bf16 ≈ 84 MB total) — already paid in M3; R_C adds +64 MiB for
the fp32 rotation matrix at runtime (or 32 MiB bf16).

## 8. Post-hoc supplementary arms (queued after selection freeze; labels explicit)

Three GPU-idle-fill experiments close the two disclosed weaknesses; none
alters the frozen M6 selection (all are equivalence tests that came out
null, as hoped):

- **Rotation-choice controls (audit Q3).** Replacing R_C = R1_T with a
  fresh random orthogonal matrix (QR, seed 1234) or a random Hadamard
  (minted by `seagle_port/fidi_mint_rc_controls.py`) on the M5a config:
  V_RCrand 3.1781 (Δ−0.035 vs R1_T, p=0.131) and V_RChad 3.1950 (Δ−0.018,
  p=0.543) — statistically indistinguishable. **The novelty is the
  placement of a rotation at the H_t→ctx-K/V interface, not the specific
  matrix**; R1_T reuse remains the natural training-free choice, and this
  strengthens the deployment story (any orthogonal works, zero training).
  It equally sharpens the claim we do NOT make: there is no evidence of
  target-basis-specific structure in R_C's benefit.
- **Q6 = Q5-recipe QAT trained WITH fc_p2** (lr 3e-2; closes caveat #1):
  train CE 2.561→2.3145 (better than Q5@3e-2's 2.389), deployed validation
  V_Q6 3.6275 vs V_Q5bp2 3.6690 (Δ−0.041, p=0.192, n.s.) — the eval-composed
  P2 costs nothing measurable; the frozen M6 stands.
- **RCAL for the final system** (M6 mtbench cycles): AL_q 2.542 / RCAL
  2.2655 / AFS 0.9097 (`RD/tables/rcal__M6_q5bp2__mtbench.json`) — M6's
  RCAL exceeds M5's 2.191 at comparable acceptance fidelity (AFS 0.910 vs
  0.915): the QAT gain is reference-consistent, not deceptive acceptance.

## 9. Reproduction

Environment: `python -m venv --system-site-packages venv` + transformers
4.57.3 + fast_hadamard_transform (built from source, CUDA 12.6 nvcc via
conda); HF_HOME with Llama-3.1-8B-Instruct (+`chat_template.jinja` and
`.no_exist` markers if the snapshot predates the split-file format — see
QAT_SELECTION_NOTE). Assets: `assets/rotations` → `../outputs/rotations`,
`assets/draft_rotations` → `RD/rotations/draft`. Rebuild `RD/hcache_w4a4_s1`
via `seagle_port.vsq_cache_hidden` (corpus committed at
`RD/tables/train_corpus.jsonl`; regenerated cache reproduced pilot CE to
0.045). Scheduler: `launch_vsq_sched.sh` (single instance);
queue/events under `RD/scheduler/`. FIDI: capture `seagle_port/fidi_capture.py`
(configs above), analyses `fidi_weights/fidi_awdecomp/fidi_alignment/
fidi_qkvo_ce/fidi_smoothcal/fidi_scale_invariance/fidi_tables`, gates
`fidi_gate_d.py`. Statistics `seagle_port/vsq_fidi_stats_battery.py`.
Verification `seagle_port/verify_dflash_spinquant_novelty.py`
(missing=0 / mismatch=0 required).
