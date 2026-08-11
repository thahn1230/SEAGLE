# Standalone draft-aware R2 (R6) under GS W4A4 PTQ — final report

Run `runs/eagle1_gs_standalone_r6_ptq_20260811_203417` · W4A4 target/draft ·
GS scaling (m = 4096^0.42 = 32.8996) · QAT OFF · LS OFF · greedy, max_new
128, mc_sim_7b_63, official cycle-pooled micro-tau, canonical prompt pools.

**Question**: does draft-aware R2 adaptation (R6) provide standalone
benefit under GS W4A4 PTQ when the residual-basis rotation R5 is absent?

**Answer: no statistically detectable benefit (CASE A).** All four
per-dataset deltas are positive but individually n.s.; and neither
sequential nor JOINT co-training with R5 beats R5 alone.

## Setup correctness (§2/§8 of the contract)

- Existing R6 checkpoints (R6_PTQ_s{0,1,2}) were all trained with
  `--rot-fixed-ckpt RD_HYB_s2.pt` — **CATEGORY B (R5-conditioned)** —
  and were therefore NOT reused as the primary arm (kept only as the
  labelled transplant control). Standalone R6 was retrained with the
  exact canonical recipe (lr 3e-4, 3000 steps, hybrid η=3.0 γ_depth=0.8,
  K=4, batch 32, same corpus, TRUE-best checkpoint), the **only** change
  being `--rot shared` (SharedRotation(R_T), trainable=False) instead of
  frozen R5.
- R5-off assertion: every trained checkpoint satisfies `R_D ≡ R_T`
  (atol 1e-6) — the residual basis is bit-for-bit the GS baseline basis.
- **GATE_BASIS PASS**: a synthetic checkpoint (R_D:=R_T, no R6) deployed
  through the same `--draft-cfg rot` path reproduces the canonical GS PTQ
  baseline **exactly** (mtbench tau 2.9955, 3346 cycles — identical to
  canonical B3_T4 to 4 decimals and cycle count). Deployment path,
  quantizers, prompts, and basis are therefore identical across arms.
- Seeds 0/1/2; pre-registered mtbench-median seed reporting (never best):
  standalone median = s2 (3.0654/3.0361/**3.0392**), JOINT median = s1
  (3.1744/**3.1814**/3.1841).

## Main table (official micro-tau)

| Method | MT-Bench | GSM8K | ShareGPT | HumanEval | mean4 |
|---|---:|---:|---:|---:|---:|
| GS PTQ (canonical B3_T4) | 2.9955 | 3.4490 | 3.1092 | 3.7490 | 3.3257 |
| **GS + R6 (standalone, s2)** | 3.0392 | 3.4819 | 3.1204 | 3.8029 | **3.3611** |
| Δ R6 | +0.0437 | +0.0329 | +0.0113 | +0.0539 | +0.0354 |

Paired prompt-cluster bootstrap (3000), GS+R6 vs GS:

| dataset | Δτ [95% CI] | raw p | verdict |
|---|---|---:|---|
| mtbench | +0.0437 [−0.0101, +0.0950] | 0.117 | n.s. |
| gsm8k | +0.0329 [−0.0052, +0.0697] | 0.093 | n.s. |
| sharegpt | +0.0113 [−0.0484, +0.0734] | 0.730 | n.s. |
| humaneval | +0.0539 [−0.0025, +0.1073] | 0.061 | n.s. |

Holm across the battery (8 comparisons incl. controls): **0 rejected**.
Honesty note: the sign is positive on 4/4 datasets (a sign-test-level
curiosity, ~p 0.0625); no individual or corrected test reaches 0.05, and
the effect (+0.035 mean4) is ~4.4× smaller than R5's (+0.153).

## Extended R5/R6 factorial (context)

| R5 | R6 | Method | MT | GSM | Share | HE | mean4 |
|---|---|---|---:|---:|---:|---:|---:|
| OFF | OFF | GS | 2.9955 | 3.4490 | 3.1092 | 3.7490 | 3.3257 |
| OFF | ON | GS+R6 standalone | 3.0392 | 3.4819 | 3.1204 | 3.8029 | 3.3611 |
| ON | OFF | GS+R5 | 3.1645 | 3.6875 | 3.2351 | 3.8282 | 3.4788 |
| ON | ON (seq) | GS+R5→R6 | 3.1698 | 3.6379 | 3.2853 | 3.8244 | 3.4794 |
| ON | ON (**joint**) | GS+R5R6 co-trained | 3.1814 | 3.6641 | 3.2268 | 3.7977 | 3.4675 |

- **JOINT co-training** (user-requested arm: R5 and R6 optimized together
  in one AdamW run, canonical recipe, `--rot residual --train-r2`):
  median-seed mean4 3.4675 (seed range 3.4675–3.5021). vs R5-only, all
  four datasets n.s. (mtbench +0.017 p=.56; gsm8k −0.023 p=.26; sharegpt
  −0.008 p=.81; humaneval −0.030 p=.26) — **co-training also does not
  beat R5 alone**.
- **Transplant control** (R5-conditioned R6 deployed without R5, labelled,
  not primary): mean4 3.3376, all datasets n.s. vs GS — the old R6 carries
  no transferable standalone value either.

## Interpretation (pre-registered CASE A)

Draft-aware rotation gains are **specific to the residual-stream basis
(R5)**. The attention-local V/O rotation site (R6) yields no significant
benefit standalone (+0.035 n.s.), no incremental benefit after R5
(+0.0006, prior study), and no benefit when co-trained jointly with R5
(3.4675 ≈ 3.4788). The final method story stands: **GS + R5 is
sufficient; R6 is unnecessary alone, sequentially, and jointly.**
The 4/4-positive standalone sign pattern is noted for completeness but is
not statistically supported; per contract, no random-R6 control grid was
launched (learned standalone R6 ≈ zero).

## Diagnostics (secondary)

- Standalone R6 training val E[tau]: best 1.90–1.92 across seeds (vs
  R5-only 2.48 on the same protocol) — the surrogate already indicates the
  attention-local site cannot recover the residual-basis gap.
- Standalone-R6 checkpoint shas: 593ac5ca / 407a09b3 / 2921198c
  (R6_STANDALONE_s{0,1,2}, TRUE-best `.best.pt` evaluated).
- GPU policy: GPU 7 excluded per user directive (SCHED_GPUS=0-6).

## Artifacts

`runs/eagle1_gs_standalone_r6_ptq_20260811_203417/`: tables/summary.csv,
stats/bootstrap_pairs_*.json + holm_adjusted.json, shards/ (all arms +
canonical B3_T4/B9_T4 copies), rotations/ (3 standalone ckpts + parity +
transplant), logs/, finalize_stats.py, scheduler queue/events.
