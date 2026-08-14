# Acceptance-Aware QAT for W4A4 EAGLE-1 — Final Report

Run: `runs/eagle1_aaq_canonical_20260813_082500` (canonical track; host
gpusystem, torch 2.6.0+cu124, CUDA 12.4, driver 550.163.01, 8x RTX 4090,
env `seagle`). Pilot track (re-optimized R1_T, superseded):
`runs/eagle1_acceptance_aware_qat_20260812_120000`.
Preregistration: pilot run `configs/preregistration.md` (+ amendments
§6/§6b, canonical restoration addendum in `audit/cross_server_compliance.md`).

**Cross-server contract**: all numbers below are measured on this host
against the ORIGINAL-server canonical function (R.bin sha 4b7e91d2,
RD_HYB_s2 sha 20c03f00, anchor sha a7c6ccc8). Baseline gate: all five
PTQ baselines reproduce the original-server per-dataset micro-taus
EXACTLY (17/17 comparisons, deltas ±0.0000 — bit-level cross-server
reproduction). Every paired statistic uses same-server shards only.
All quantization is fake-quant; no real-INT4 speed claims.

## Executive answers (the 12 pre-registered questions)

**1. Does acceptance-aware QAT beat conventional EAGLE-style QAT?**
Yes at the recipe level, decisively: tuned (LR 1e-6) LK-hybrid chain-QAT
beats the published-contract conventional QAT (anchor-init, ShareGPT,
LR 1e-5) on every family — F2 (GS+R5) 4/4 SIG (+0.143..+0.255 per
dataset), F4 (GS) 4/4 SIG, F5 (LS+R5) 4/4 SIG. **At the objective level
the honest answer is narrower**: under the SAME tuned chain contract,
conv-vs-hybrid differences are small and mixed (GS: n.s. on 3/4,
humaneval +0.087 favoring conv; GS+R5: hybrid ahead by ~+0.01-0.03 and
by +0.033 at calib). The dominant factors are optimization strength
(LR) and the training pipeline, not the loss identity.

**2. Does it beat the corresponding PTQ baseline?** Yes, everywhere:
Naive +0.36, GS +0.27, LS +0.28 mean4 (all 4/4 SIG); GS+R5 +0.132
(F1: 4/4 SIG, the study's primary question); LS+R5 +0.076 (F3: 3/4
SIG; humaneval n.s. −0.019).

**3. Does GS+R5 finally benefit from QAT when it optimizes verifier
acceptance?** **Yes — CASE A.** GS+R5: AA-tuned 3.6113 > PTQ 3.4788 >
conv-QAT 3.4270 (mean4). F1 significant on all four datasets after
Holm. Critically, this required the LR pilot's 1e-6: at the canonical
matched LR (1e-5) every objective REPRODUCES the historical
non-additivity (matched-hybrid vs PTQ at GS+R5: n.s./negative,
gsm8k −0.089 SIG, humaneval −0.216 SIG). The historical "QAT doesn't
help R5" result was an optimization-strength artifact, not an
objective-mismatch artifact.

**4. Does LS+R5 benefit?** Yes: AA-tuned 3.6025 vs PTQ 3.5267 (+0.076,
3/4 SIG) and vs conv-QAT 3.4343 (+0.168, 4/4 SIG).

**5. Which objective best predicts actual greedy micro-tau?** At
matched budget the 600-step pilot ranking (hybrid > exptau > survival >
conv > greedy > accsurv at GS) did NOT persist at full budget; at the
tuned LR, conv-in-chain ≈ hybrid at GS and hybrid slightly leads at the
R5 anchors. Survival/greedy variants were dominated at every scale
tested. Verdict: LK-hybrid is the best single choice (never worse,
sometimes better), but no objective is a strong predictor by itself —
the surrogate-tau link is mediated by the training regime (see Q6/CASE
D evidence).

**6. Does hidden reconstruction get worse while acceptance improves?**
Yes — the pre-registered dissociation was observed directly: in matched
pilots, hybrid/exptau/survival raised val expected-tau (+0.20..+0.36)
while hidden SmoothL1 WORSENED (e.g. 0.662→0.694); conv improved both
but gained less tau. And the converse (CASE-D signature) also occurred:
at LR 1e-5 x 3000 steps training alphas reached 0.85-0.95 while
deployed tau FELL below PTQ — teacher-forced surrogate improvement
does not guarantee deployed greedy tau.

**7. Does standalone draft quality matter for final SD correctness?**
No. Final outputs are governed by the verifier; the AA draft (whose
hidden-reconstruction metrics are worse) leaves output parity unchanged
(Q8).

**8. Are speculative outputs exactly identical to W4A4 target-only
greedy outputs?** Not for ANY draft, including PTQ baselines — a known,
draft-independent property of the W4A4 tree verifier (A4 bin-flips
between tree-shaped and AR execution; FP16 config = exact 1.0
prefix-match 10/10). Pre-registered criterion was therefore parity
INVARIANCE: baseline 0.1879 vs AA draft 0.1829 mean prefix-match
(10 prompts x 128 tokens) — no draft-dependent degradation. The
verifier-shape audit (median TV(p_tree,p_seq)=0.0173, ~20x below the
optimized acceptance gap; argmax agreement 0.9255) certified the
sequential corpus teacher as deployment-matched to first order.

**9. Early or late speculative depths?** Training-time gains
concentrate at later depths (pilot alpha deltas grow with k), and the
free-running diagnostic shows the remaining headroom is also at depth
≥2: teacher-forced alpha [0.70,0.75,0.57,0.58] vs free-running
[0.70,0.59,0.40,0.34] (E[tau] 2.85 vs 2.67) for the selected GS+R5 AA
draft — the teacher-forced trajectory overstates deep-depth acceptance.

**10. Does AA-QAT avoid the GSM8K/HumanEval degradation of conventional
QAT?** Yes. Historical conv-QAT+R5 degraded GSM8K/HumanEval vs R5-PTQ;
canonical CQH reproduces that pattern (gsm8k −0.104 SIG, humaneval
−0.127 SIG vs B9). AA-tuned instead GAINS on gsm8k (+0.151 SIG) and
humaneval (+0.059 SIG) at GS+R5.

**11. Does AA-QAT alter down_proj as aggressively as conventional
QAT?** No — the §25 prediction held exactly: conventional QAT flips
42.6-43.3% of down_proj W4 codes (and ~7-8% of attention sites);
tuned AA-hybrid flips 0.8-1.2% (down 0.008, o 0.004); the tuned conv
chain sits between (down 0.025-0.028). Same deployed tau family,
~40x less weight-code movement.

**12. Final recommended deployment recipe?**
`GS (or LS) -> R5 (RD_HYB_s2) -> tuned chain QAT (LK-hybrid, K=4
teacher-forced, core-lr 1e-6, 3000 steps, c4-calib checkpoint
selection)` — mean4 3.61 vs 3.48 for the previous PTQ-only recipe.
Per §32 the PTQ-only recipe should be reconsidered: the improvement is
significant on 4/4 datasets at the primary anchor. If retraining is
undesirable, PTQ remains a strong floor; conventional QAT should NOT
be used on R5-corrected deployments.

## Final table (original scale, micro-tau; median seed of 3, all seeds in tables/)

| Method | PTQ | conv-QAT (published contract) | AA-QAT (tuned LK-hybrid) | ΔAA−PTQ | ΔAA−conv |
|---|---:|---:|---:|---:|---:|
| Naive | 1.2942 | 1.8042 | 1.6242 | +0.3300 | −0.1800 |
| GS | 3.3257 | 3.4473 | 3.5972 | +0.2715 | +0.1499 |
| LS | 3.3429 | 3.4215 | 3.6035 | +0.2606 | +0.1820 |
| GS+R5 | 3.4788 | 3.4270 | **3.6113** | **+0.1325** | **+0.1843** |
| LS+R5 | 3.5267 | 3.4343 | 3.6025 | +0.0758 | +0.1682 |

(Naive PTQ mean4 from canonical B1 4-dataset panel: 1.2521/1.2746/
1.2409/1.2991. Naive is the one arm where big-data conv-QAT still wins
— the naive interface is so damaged that data volume dominates.)

Per-dataset (median seeds; PTQ rows = exact original-server values):

| Row | MT | GSM8K | ShareGPT | HumanEval | mean4 |
|---|---:|---:|---:|---:|---:|
| GS PTQ (B3) | 2.9955 | 3.4490 | 3.1092 | 3.7490 | 3.3257 |
| GS+R5 PTQ (B9) | 3.1645 | 3.6875 | 3.2351 | 3.8282 | 3.4788 |
| LS PTQ (B5) | 3.0728 | 3.4759 | 3.0942 | 3.7286 | 3.3429 |
| LS+R5 PTQ (B7) | 3.2004 | 3.7390 | 3.3115 | 3.8557 | 3.5267 |
| GS conv-QAT | 3.1876 | 3.6193 | 3.2876 | 3.6946 | 3.4473 |
| GS+R5 conv-QAT | 3.1726 | 3.5832 | 3.2510 | 3.7013 | 3.4270 |
| LS conv-QAT | 3.1695 | 3.6085 | 3.2214 | 3.6866 | 3.4215 |
| LS+R5 conv-QAT | 3.1925 | 3.5998 | 3.2532 | 3.6917 | 3.4343 |
| GS AA-tuned | 3.3310 | 3.8353 | 3.3900 | 3.8327 | 3.5972 |
| GS+R5 AA-tuned | 3.3158 | 3.8381 | 3.4041 | 3.8872 | 3.6113 |
| LS AA-tuned | 3.3125 | 3.8683 | 3.3874 | 3.8457 | 3.6035 |
| LS+R5 AA-tuned | 3.3312 | 3.8264 | 3.4156 | 3.8369 | 3.6025 |
| GS conv-tuned (control) | 3.3202 | 3.8386 | 3.4218 | 3.9193 | 3.6250 |
| GS AA-matched (1e-5) | 3.1400 | 3.6467 | 3.2575 | 3.7368 | 3.4453 |
| GS+R5 AA-matched (1e-5) | 3.1563 | 3.5987 | 3.2334 | 3.6118 | 3.4001 |

## Statistics (paired same-prompt cluster bootstrap, 3000 reps, Holm within family; p floored at 1/3000)

Primary families (AA-tuned vs baseline; + = AA advantage):

| Family | MT | GSM8K | ShareGPT | HumanEval |
|---|---|---|---|---|
| F1 GS+R5 AA vs PTQ | +0.151 SIG | +0.151 SIG | +0.169 SIG | +0.059 SIG |
| F2 GS+R5 AA vs conv | +0.143 SIG | +0.255 SIG | +0.153 SIG | +0.186 SIG |
| F3 LS+R5 AA vs PTQ | +0.131 SIG | +0.087 SIG | +0.104 SIG | −0.019 n.s. |
| F4 GS AA vs conv | +0.143 SIG | +0.216 SIG | +0.103 SIG | +0.138 SIG |
| F5 LS+R5 AA vs conv | +0.139 SIG | +0.227 SIG | +0.162 SIG | +0.145 SIG |

19/20 SIG after Holm. Exploratory: AA vs PTQ 4/4 SIG at GS/LS/Naive;
conv-tuned vs AA-tuned at GS n.s. 3/4 (humaneval +0.087 conv, SIG);
matched-AA vs PTQ at GS+R5 negative (gsm8k −0.089 SIG, humaneval
−0.216 SIG) — the historical non-additivity, reproduced.

## Case verdict (§28/§32)

**CASE A at the recipe level** (GS+R5: AA-QAT > PTQ > conv-QAT, F1 4/4
SIG) — with two pre-registered qualifications that are themselves
findings: (a) **CASE D occurred at matched LR** (LK surrogate improved
while deployed greedy tau fell below PTQ — the surrogate-deployment
link breaks under over-strong teacher-forced optimization; corpus
covers ~19 epochs at 3000x32 windows); (b) at the tuned LR the
objective choice itself is nearly free (CASE-C-flavored at GS), so the
honest claim is **"acceptance-aware, deployment-verifier-taught,
gently-optimized chain QAT"** rather than "the LK loss is the magic
ingredient" (novelty rule §29 respected: the LK loss is LK-Losses
applied to W4A4 draft weights; what is new is the demonstrated
misalignment mechanism and its fix).

## Diagnostics

- **Verifier shape (§4)**: sequential teacher certified; median
  TV(p_tree,p_seq)=0.0173, p90 0.222, argmax agree 0.9255; |Δα| ≤ TV
  bound; tree execution slightly favors its own proposals (R_q 1.895 vs
  R_seq 1.742/cycle).
- **SD parity (§23)**: T16 exact 1.0; W4A4 tree-vs-AR mismatch is
  pre-existing and draft-independent; AA drafts parity-invariant
  (0.183 vs 0.188 baseline).
- **RCAL (§27)**: GS+R5 AA vs base — AL_q 2.316 vs 2.164, RCAL 1.859
  vs 1.767 (~61% of the deployed gain is FP16-reference-consistent),
  AFS 0.829 vs 0.840, no deceptive-AL flag.
- **Code flips (§25/§26)**: conv-QAT down_proj 42.6-43.3% flips vs
  AA-tuned 0.8-1.2%; conv-tuned chain intermediate (2.5-2.8%).
- **Free-running (§17)**: depth≥2 rollout gap (E[tau] 2.85 tf vs 2.67
  free) — on-policy training is the clearest future direction.
- **B5 evaluator bug**: `--draft-cfg d4p3` silently drops
  `--alpha-rec`; LS-PTQ evals must use `d4p3_deploy`. Fixed; pilot
  track's PTQ_ls shards invalidated (documented in audit/).

## Provenance & scope

Canonical restoration: original R.bin/RD_HYB_s2 from git
`exp/eagle1-r6-draft-aware-r2 @8f0dab4`, anchor from HF
`kkkevinnn/seagle-canonical-anchors`; shas verified (4b7e91d2 /
20c03f00 / a7c6ccc8). AA/conv-tuned/matched chain arms init from the
public draft; CQH conv-QAT init from anchor.pt (published contract).
R5 = RD_HYB_s2 frozen everywhere; R6 absent; scales frozen; target
frozen. Selection: c4-calib argmax over cadence ≥1000 (identical rule
for every arm); median-mtbench-seed reporting; no test-set selection.
Pilot-track results (re-optimized R1_T) are archived as a
methodological pilot; every number in this report is canonical-track.
GS conv-QAT canonical CQH reproduces grid B4 within 0.001 mean4
(3.4473 vs 3.448) — the conv-QAT pipeline itself also crosses servers
faithfully. LS+R5 conv-QAT lands 3.4343 vs grid B8 3.472 (−0.038;
seed/anchor-lineage variance, within the grid's documented ≤0.06
spread).
