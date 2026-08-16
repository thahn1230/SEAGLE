# QAT Quantized-Anchor — Frontier Closure Audit (2026-08-16)

> **Closure verdict: B + D** — *Efficiency dominance only* (Outcome B),
> with the mechanism reframed to *W_first basis coupling* (Outcome D).
> The original "frontier performance dominance" claim is **withdrawn**;
> the original "1-2% sparse selective flips" mechanism story is
> **falsified and replaced**.

Five-bullet summary (section 41):

1. **Does anchor beat the full low-LR frontier?** NO on acceptance
   length. Against the strongest tuned plain baselines (conv 1e-6
   mean4 3.6148/3.6198, LK-hybrid 1e-6 3.6113/3.6198), the best HB
   model is statistically indistinguishable on every dataset
   (Families A/B: 0 Holm-significant wins, 0 losses, 4 n.s. each,
   10k paired bootstrap). The HB conv b2% finalist *loses* gsm8k
   (Families C/D: 0W/1L). Tuned plain QAT itself beats PTQ 4W/0L.
2. **Is the result seed robust?** Partially (Seed-B). HB hybrid
   3-seed mean4 3.6129 ± 0.0145 vs tuned plain conv 3.6108 ± 0.0116 —
   nominal +0.002, not significant; 2/3 HB seeds sit above the plain
   seed mean, s1 (3.5986) does not. The *drift* profile is fully seed
   robust (all seeds: H_Q_controlled ≤ 4e-4, D_Q ≈ 0.00066-0.0008).
3. **What fraction of drift is actually hard-budget controlled?**
   0.21% of the best HB model's changed codes (7,276 of 3,506,993).
   H_Q_controlled = 3.1e-5. The reported "H_Q 1.30%" is 97.24%
   W_first h-half (flip rate 20.3% inside that half), 2.6% W_first
   e-half. Calling it a "0.5% whole-model budget" was an accounting
   error: the budget binds a basis in which almost nothing needs to
   move, while the first-fold basis moves freely.
4. **What role does W_first h-half play?** It is the dominant
   adaptation channel (deployment-code counterfactuals, exact frozen
   quantizer): reverting only W_first-h keeps just 26% of the HB gain
   (3.5168 of +0.1488); W_first-h alone reproduces 75% (3.5904); all
   W_first alone 94% (3.6184); the 7,276 controlled flips alone 28%
   (3.5207). Verdict **W-B**: the constrained recurrent-basis update
   induces useful adaptation expressed in the coupled first-fold
   basis. In-cell (|u-c0| <= 0.49) master movement is code-invisible
   in the controlled basis but code-visible through the W_first fold.
5. **Strongest defensible paper claim:**
   *Quantized-anchor hard-budget QAT reaches the tuned-plain-QAT
   acceptance frontier while moving 3.5-4x less deployment-visible
   quantized mass (D_Q 0.00066 vs 0.0023) and leaving the budgeted
   code sites essentially at the PTQ anchor (H_Q_controlled 3e-5 vs
   2.2e-2), because constraining one deployed quantization basis
   channels adaptation into a coupled second basis.* No claim of
   acceptance-length superiority over LR tuning is supported.

---

## 0. Provenance

- Base branch `exp/eagle1-qat-quantized-anchor-causal`, HEAD
  7d9db96 (64453c2 confirmed ancestor). Closure branch
  `exp/eagle1-qanchor-frontier-closure-audit`.
- Environment/artifact record:
  `runs/eagle1_qanchor_closure_20260816/tables/env_record.txt`
  (R.bin 4b7e91d2, RD_HYB_s2 20c03f00, anchor d290220c, fp16-anchor
  a7c6ccc8, GS alpha 32.89964245299412, W4 sym MSE-clip maxq=7 frozen
  scales for anchored deploys, A4 per-token dynamic, torch 2.6.0
  cu124, 8x RTX 4090).
- Evaluation contract identical to the causal study (official
  cycle-pooled micro-tau, mc_sim_7b_63, greedy, max-new 128, pools
  mtbench:80/gsm8k:200/sharegpt:80/humaneval:164, mean4 = arithmetic
  mean of dataset micro-taus, paired same-prompt cluster bootstrap
  10k reps, Holm across the 4 datasets per family).

## 1. Headline reproduction (section 3): PASS

All six headline numbers recomputed from RAW acceptance lists match
the reported values to <= 3e-5 (roundoff):
PTQ 3.4788, plain 1e-5 3.4443, HB conv b2% 3.5877, HB hybrid b0.5%
3.6276, LoRA 3.3914, best soft-cell 3.5634.
CSV: `results/qanchor_closure/reproduction/current_headline_reproduction.csv`.

## 2. Full plain-QAT LR frontier (Phase F1)

All LR points share the canonical corpus/chain/anchors; drift
recomputed for EVERY checkpoint under the exact causal-study anchor
and frozen quantizer (`tables/closure_drift.json`;
`results/qanchor_closure/frontier/plain_lr_frontier.csv`).
Selected-cadence finals (fresh-scale deploy, the baselines' own
contract; reused shards marked in the frontier CSV):

| objective | LR | seeds (mean4) | best | H_Q_all | D_Q |
|---|---|---|---|---|---|
| conv | 3e-7 | 3.5484 (s0, median) | 3.5484 | 0.66% | 0.00068 |
| conv | 1e-6 | 3.5977/3.6148/3.6198 | **3.6198** | 2.2-2.5% | 0.0023-6 |
| conv | 3e-6 | 3.5866 (s1, median) | 3.5866 | 4.1% | 0.0042 |
| conv | 1e-5 | 3.4443/3.4510/3.4237 | 3.4510 | 9.6-13.9% | 0.0099-0.0144 |
| hybrid | 3e-7 | 3.5763 (s2, median) | 3.5763 | 0.54% | 0.00058 |
| hybrid | 1e-6 | 3.6198/3.6113/3.5914 | **3.6198** | 1.0-1.4% | 0.0011-0.0015 |
| hybrid | 3e-6 | 3.5972 (s0, median) | 3.5972 | 2.4% | 0.0026 |
| hybrid | 1e-5 | 3.4001 (s0) | 3.4001 | 5.3% | 0.0057 |

The original causal-study Pareto omitted every point except 1e-5.
The 1e-6 points sit 0.14-0.17 above the 1e-5 baseline used there.

## 3. Corrected H_Q accounting (section 6)

Best HB hybrid b0.5% s0 (`results/qanchor_closure/wfirst/flip_partition.json`):

| partition | flips | rate in partition | fraction of all flips |
|---|---|---|---|
| S_controlled (8 budgeted folds) | 7,276 | 3.1e-5 | 0.21% |
| S_Wfirst_e | 89,535 | 0.53% | 2.55% |
| S_Wfirst_h | 3,410,182 | 20.33% | **97.24%** |
| S_down (within controlled) | 793 | 1.8e-5 | 0.02% |
| total (H_Q_all) | 3,506,993 | 1.30% | 100% |

All figures/tables now report H_Q_all, H_Q_controlled and the W_first
split separately. "hard-budget 0.5%" refers to the CONTROLLED sites
only.

## 4. W_first causal decomposition (Phase F4)

Deployment-code counterfactuals (NOT trainable checkpoints; frozen
anchor scales; constructed codes verified bit-exact at deploy via
--verify-codes; SHAs in
`results/qanchor_closure/wfirst/counterfactual_shas.json`;
unit gates in `configs/build_wfirst_counterfactuals.py` all PASS,
including HB_controlled_only == HB_no_Wfirst, since S_other is empty):

| counterfactual | mean4 | dvs PTQ | fraction of HB gain |
|---|---|---|---|
| PTQ (C0) | 3.4788 | 0 | 0 |
| HB_controlled_only (=no_Wfirst) | 3.5207 | +0.042 | 28% |
| HB_no_WfirstH | 3.5168 | +0.038 | 26% |
| HB_WfirstH_only | 3.5904 | +0.112 | 75% |
| HB_Wfirst_only | 3.6184 | +0.140 | 94% |
| full HB (CHB) | 3.6276 | +0.149 | 100% |

(Gain fractions are descriptive; effects are not additive.)
**Verdict W-B.** The 7,276 utility-selected controlled flips are real
but minor (+0.042); the induced W_first drift carries the gain.

## 5. Joint-basis constraint pilot (section 24)

One variant: HB hybrid b0.5% + differentiable W_first cell loss
(lambda_first = 1, rho 0.45, frozen W_first anchor scales/codes,
computed after the exact W_first transform; --anchor-sites W_first;
step-0 grad-ratio check did not trigger the >10x normalization).
Result: drift profile essentially unchanged (Wf_h 20.2% vs 20.3%,
D_Q 0.000656), calib improved (3.3641 vs 3.2462) but the MT-Bench
pilot came out 3.3027 vs the HB comparator's 3.3797 — clearly
inferior, NOT promoted to 4-dataset evaluation (preregistered pilot
rule). lambda_first = 1 does not close the loophole; W-C
(harmful leakage) is not supported by this pilot.

## 6. Training-seed closure (Phase F3, sections 10-13)

Identical config, seed-only change; identical calib selection rule
(all seeds selected st3000). Table B
(`results/qanchor_closure/tableB_seed_finalists.csv`):

| method | s0 | s1 | s2 | mean | SD |
|---|---|---|---|---|---|
| HB hybrid b0.5% | 3.6276 | 3.5986 | 3.6126 | 3.6129 | 0.0145 |
| HB conv b2% | 3.5877 | 3.5705 | 3.5655 | 3.5746 | 0.0116 |
| plain conv 1e-6 | 3.5977 | 3.6148 | 3.6198 | 3.6108 | 0.0116 |
| plain hybrid 1e-6 | 3.6198 | 3.6113 | 3.5914 | 3.6075 | 0.0146 |
| plain conv 1e-5 | 3.4443 | 3.4510 | 3.4237 | 3.4397 | 0.0142 |

**Seed verdict: Seed-B** for HB hybrid (2/3 seeds above the plain
seed mean; seed mean nominally higher by +0.002; the s0 headline
3.6276 is the best single seed but not reproduced by s1). HB conv is
NOT competitive on tau at seed level (3.5746 < 3.6108). Prompt
bootstrap and seed variability are reported separately; n_seed = 3 —
no hierarchical claim.

## 7. Statistical closure (section 8; Table E)

10k paired cluster bootstrap, Holm across 4 datasets per family
(`stats/final_stats_holm.json`,
`results/qanchor_closure/tableE_statistics.csv`):

| family | result |
|---|---|
| A: HB hyb med vs plain conv 1e-6 med (3.6148) | 0W/0L (4 n.s.) |
| A2: vs strongest plain conv seed (3.6198) | 0W/0L |
| B: vs plain LK 1e-6 med (3.6113) | 0W/0L |
| B2: vs strongest plain LK seed (3.6198) | 0W/0L |
| C: HB conv med vs plain conv med | 0W/**1L** (gsm8k) |
| D: HB conv med vs plain LK med | 0W/**1L** (gsm8k) |
| HB hyb s0 vs PTQ | 3W/0L |
| plain conv 1e-6 med vs PTQ | **4W/0L** |

## 8. Corrected Pareto frontier (sections 27-28, computed)

`results/qanchor_closure/frontier/pareto_frontier_{dq,hq,hqc}.csv`
(19 points; JB pilot excluded per the not-promoted rule; reused vs
new marked). Non-dominated sets (maximize tau / minimize x):

- **tau vs D_Q**: PTQ, plain hybrid 3e-7 s2, **HB hybrid b0.5% s0**.
- **tau vs H_Q_all**: PTQ, plain hybrid 3e-7 s2, plain hybrid 1e-6
  s1, **HB hybrid b0.5% s0**.
- **tau vs H_Q_controlled** (diagnostic only): PTQ, **HB hybrid s0**.

The HB hybrid point is on every frontier, but tuned plain points sit
within noise of it on tau — dominance is in the drift dimension, not
acceptance length. Under seed means the picture is the same: HB
3.6129 @ D_Q 0.00066 vs plain conv 3.6108 @ D_Q 0.0023.

## 9. Selectivity audit (sections 25-26; Table D)

`results/qanchor_closure/selectivity/`:

- Denominator-explicit down audit: HB's down flips are 0.02% (hybrid)
  / 0.6% (conv) OF ITS OWN flips (rate within down 1.8e-5 / 5.6e-4);
  plain tuned models: 20-29% of flips, 1.2-3.9% rate; aggressive:
  37% / 31%. The "avoids down_proj" observation survives, but it is
  a byproduct of the budget: HB's flips are ~95% W_first-h, a tensor
  whose drift the budget does not control.
- HB vs low-LR flipped-index overlap: Jaccard 0.053-0.063 (HB-vs-HB
  across objectives: 0.40). HB does NOT select "the same codes as
  low LR, just fewer" — its changed-code set is nearly disjoint,
  consistent with the basis-coupling mechanism.

## 10. Parity anomaly (Phase F6): reporting artifact

The comparator hashes fp16 BYTES. Forensic rerun
(`runs/eagle1_qanchor_closure_20260816/tables/gateD_qat_parity.json`):
identity-mode W_first: torch_equal=True, allclose(rtol=0,atol=0)=True,
0 NaN/Inf, same dtype/shape/contiguity, byte-mismatch at exactly
**37 elements, all fp16 signed-zero pairs (-0.0 vs +0.0)**; chain
logit parity max|dlogit| = 0.0 at all depths. The deployed
(gamma_R1) mode is byte-exact. Correct wording: "value-exact in both
modes; byte-exact except 37 signed-zero encodings in the identity
mode". No effect on logits; the original PASS verdict stands with
corrected wording.

## 11. Code-frozen LoRA wording (section 30)

Corrected claim: *the tested code-frozen LoRA baseline does not
explain or dominate the anchored-QAT gain under our configuration.*
Tested exactly: ranks {4, 8, 16} at LR 1e-4 and rank 8 at 1e-3; side
branches on all 9 fold GEMM sites (~1.64M params at r16); 3000 steps;
chain objective; calib-selected r16@1e-4 st1500; best final 3.3914
(0W/2L vs PTQ). No generalization beyond these adapter
configurations is claimed.

## 12. Final claim (section 44 falsification test)

The candidate sentence — "reproducible Pareto advantage over the best
tuned plain QAT because it selectively controls deployment-visible
code changes" — is **rejected as stated**: (i) no acceptance-length
advantage over tuned plain QAT survives (0W/0L / 0W/1L); (ii) the
mechanism is not selective control of deployment-visible changes —
97% of visible changes are in an uncontrolled basis. The revised,
supported statement:

> Hard-budget quantized-anchor QAT reaches the same acceptance
> length as the best LR-tuned plain QAT while leaving the budgeted
> deployment code sites at the PTQ anchor (H_Q_controlled ~3e-5,
> ~700x lower than tuned plain) and moving 3.5x less dequantized
> mass overall, because constraining the recurrent-basis codes
> channels adaptation into the coupled first-fold basis
> (W-B mechanism, established by deployment-code counterfactuals).

Whether near-zero controlled-code movement has deployment value
(e.g., partial weight reuse, cache/delta-encoding of INT4 tables) is
future work — the fake-quant pipeline does not demonstrate it.
