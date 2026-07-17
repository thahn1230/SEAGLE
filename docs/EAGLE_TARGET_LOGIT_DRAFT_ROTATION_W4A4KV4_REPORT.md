# EAGLE-1 Target-Logit-Aligned Draft Rotation with W4A4KV4 and Cross-Dataset Causal Validation (TLDR-KV4)

Run: runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003  
Branch: exp/eagle1-target-logit-draft-rotation-kv4  
Date: 2026-07-18. Fake-quant only; micro-AL only; all KV4 results are "KV4 without R3".

## 8. Phase G: draft-rotation training setup

**Question.** The Phase C/D causal panel established that the entire T8_D8-vs-T16_D8 gap is an exposed-rotated-interface effect (Δ_interface,FP16 = +1.16 to +1.42 micro-AL across the 5 datasets, mean ≈ +1.31; Δ_W8 ≈ 0). Phase G asks whether a *learned* draft rotation R_D — decoupled from the target rotation R_T — can recover acceptance under the deployed W4A4 draft (D4P3), instead of defaulting to R_D = R_T.

**Parameterization (fixed across all checkpoints).** The draft is re-based as W_rec = [W_e | W_h·R_D], with R_D applied post-projection, the AR block conjugated by R_D, and the LM head as W_lm·R_D. The first path keeps the R_T bridge via `first_fold_R`, folded into weights — never a runtime GEMM. `shared` denotes R_D = R_T (the validated default); `drot_none` denotes R_D = I. All quantization is fake-quant; no latency claims anywhere in this study.

**Teacher caches.** Six caches of deployed-target top-64 logits (T_WIN=48 windows, K_DEPTH=4), archived in `rotations/`:

| cache | teacher | corpus |
|---|---|---|
| traincache__t4__wiki-c4-sharegpt-gsm8k-code | t4 (W4A4) | 5-domain mix |
| traincache__t8__wiki-c4-sharegpt-gsm8k-code | t8 (W8A8) | 5-domain mix |
| traincache__t4kv4__wiki-c4-sharegpt-gsm8k-code | t4kv4 (W4A4 + KV4 without R3) | 5-domain mix |
| traincache__fp16__wiki-c4-sharegpt-gsm8k-code | fp16 | 5-domain mix (for `fptarget` term) |
| traincache__mixed | mixed targets | 5-domain mix |
| traincache__t4__wiki | t4 | wikitext only (data ablation) |

**Objectives.** `deployKL` = KL to the deployed-target top-64 logits; `targetCE` = CE against teacher argmax; `rank` = pairwise ranking of teacher top tokens; `feature` = hidden-feature matching; `self` = draft self-consistency; `fptarget` = additional KL to the fp16 target (only in `full_dual`); `selfrecon` = self-reconstruction only. The "full EAGLE recipe" is deployKL+rank+feature+self.

**Trainer.** STE fake quant in the loop (W4 with MSE clip search, a4 per-token), Adam with QR retraction to the orthogonal manifold, 400 steps standard (500 for mixed, 300 for the early DROT_T4_TARGET_KL). Every checkpoint saved alpha = 32.0 — the P3 embedding scale was **not** co-optimized with R_D (documented limitation; see §9 alpha sweep).

**Waves.**
- *Wave 1* — 13 objective/hyperparameter ablations on the t4 teacher cache, 400 steps each: deployKL, deployKL+feature, deployKL+rank, deployKL+rank+feature, deployKL+rank+feature+self, deployKL+targetCE, full_dual, full_gamma{0.5, 0.85, 1.0}, full_tau{1.0, 4.0}, selfrecon.
- *Wave 2* — target-specific full-recipe checkpoints: spec_T8 (DROT_EAGLE_T8), spec_T4 (DROT_EAGLE_T4), spec_T4KV4 (DROT_EAGLE_T4KV4), mixed (DROT_EAGLE_MIXED, 500 steps), plus randominit (trained from a random orthogonal init) and wikionly (t4 teacher, wikitext-only data).

**Proxy validation grid** (`rotation_checkpoints.csv`, val_top1_d1 = depth-1 top-1 agreement with the teacher):

| ckpt | steps | objective | val_top1_d1 |
|---|---|---|---|
| DROT_EAGLE_T8 | 400 | full recipe | 0.3585 |
| DROT_EAGLE_T4KV4 | 400 | full recipe | 0.3208 |
| abl__t4__deployKL | 400 | deployKL | 0.3019 |
| abl__t4__selfrecon | 400 | selfrecon | 0.3019 |
| DROT_EAGLE_T4 / abl full recipe / feature / targetCE / full_dual / full_tau4.0 | 400 | (various) | 0.2830 |
| DROT_T4_randominit / full_gamma0.85 / full_gamma1.0 / full_tau1.0 | 400 | full recipe | 0.2642 |
| abl__t4__deployKL_rank / full_gamma0.5 | 400 | (various) | 0.2453 |
| DROT_EAGLE_MIXED | 500 | full recipe | 0.2000 |
| DROT_T4_wikionly | 400 | full recipe | 0.1698 |
| DROT_T4_TARGET_KL | 300 | deployKL | 0.0566 |

**Gate G (orthogonality).** The `orth_err` column in `rotation_checkpoints.csv` was recorded as NaN; I recomputed it directly from all 20 saved checkpoints (`rotations/*.pt`, fp32): max |RᵀR − I| = 9.54e-07 across all checkpoints (worst: abl__t4__full_gamma0.5). All pass the <1e-3 gate with ~3 orders of margin.

## 9. Phase G: results

All cells in this section are **screening** measurements (20 pinned eval prompts/dataset, 128 new tokens, greedy, tree mc_sim_7b_63, micro-AL = Σtau/n_cycles pooled over 675–1350 cycles per cell). They are not confirmatory numbers.

### 9.1 t4 target (W4A4), full panel — screening micro-AL

| rotation | c4 | gsm8k | humaneval | mtbench | sharegpt | mean |
|---|---|---|---|---|---|---|
| **shared_RT** | **2.9386** | **3.5568** | **3.7420** | **2.7492** | **3.0469** | **3.2067** |
| selfrecon | 2.4675 | 2.8087 | 3.1118 | 2.2945 | 2.5732 | 2.6511 |
| wikionly | 2.3882 | 2.7413 | 3.0058 | 2.2420 | 2.4374 | 2.5629 |
| deployKL | 2.3799 | 2.6195 | 2.8597 | 2.1791 | 2.3975 | 2.4871 |
| full_dual | 2.3885 | 2.5776 | 2.8744 | 2.1803 | 2.3564 | 2.4754 |
| randominit | 2.3806 | 2.5794 | 2.7262 | 2.2101 | 2.4429 | 2.4678 |
| spec_T4KV4 | 2.3166 | 2.5395 | 2.7974 | 2.1717 | 2.3540 | 2.4358 |
| spec_T4 (= full_eagle)* | 2.2875 | 2.4846 | 2.7447 | 2.1201 | 2.2989 | 2.3872 |
| drot_none (R_D = I) | 2.3418 | 2.5434 | 2.5870 | 2.1488 | 2.3086 | 2.3859 |
| spec_T8 | 2.2759 | 2.4609 | 2.6931 | 2.1272 | 2.2777 | 2.3670 |
| mixed | 2.2808 | 2.4449 | 2.6663 | 2.1712 | 2.2708 | 2.3668 |

\* `full_eagle` and `spec_T4` are deterministic duplicate legs of the same t4-teacher full-recipe checkpoint; their rows are numerically identical (shard-skip dedupe).

shared_RT wins every dataset by a wide margin. The rotation trained *for* this target (spec_T4, mean 2.3872) is statistically indistinguishable from the untrained identity (drot_none, 2.3859) and from a rotation trained from random init (2.4678). Training objectives moved the proxy (val_top1_d1 0.057–0.358) but bought essentially nothing at runtime.

### 9.2 t8 and t4kv4 panels — screening micro-AL

**t8 (W8A8):**

| rotation | c4 | gsm8k | humaneval | mtbench | sharegpt | mean |
|---|---|---|---|---|---|---|
| **shared_RT** | **3.0268** | **3.7037** | **3.8444** | **3.0097** | **3.4207** | **3.4011** |
| spec_T4 | 2.8139 | 3.4791 | 3.5361 | 2.7984 | 3.1028 | 3.1461 |
| spec_T4KV4 | 2.7846 | 3.4377 | 3.4352 | 2.7737 | 3.0856 | 3.1034 |
| mixed | 2.6906 | 3.3636 | 3.3592 | 2.7777 | 3.0137 | 3.0410 |
| spec_T8 | 2.5724 | 3.0934 | 3.0722 | 2.5952 | 2.7577 | 2.8182 |

**t4kv4 (W4A4 + target KV4, "KV4 without R3"):**

| rotation | c4 | gsm8k | humaneval | mtbench | sharegpt | mean |
|---|---|---|---|---|---|---|
| **shared_RT** | **3.0574** | **3.5203** | **3.8000** | **2.6922** | **3.1083** | **3.2356** |
| spec_T4KV4 | 2.4297 | 2.5131 | 2.7096 | 2.1601 | 2.3217 | 2.4268 |
| spec_T8 | 2.3718 | 2.4707 | 2.6851 | 2.1344 | 2.3610 | 2.4046 |
| spec_T4 | 2.3552 | 2.5126 | 2.7066 | 2.1406 | 2.3120 | 2.4054 |
| mixed | 2.3578 | 2.4779 | 2.6146 | 2.1355 | 2.3142 | 2.3800 |

### 9.3 Transfer matrix (mean micro-AL over 5 datasets, screening)

| rotation \ target | t8 | t4 | t4kv4 |
|---|---|---|---|
| shared_RT | **3.4011** | **3.2067** | **3.2356** |
| spec_T8 | 2.8182 | 2.3670 | 2.4046 |
| spec_T4 | 3.1461 | 2.3872 | 2.4054 |
| spec_T4KV4 | 3.1034 | 2.4358 | 2.4268 |
| mixed | 3.0410 | 2.3668 | 2.3800 |

Within the alternatives, matched-teacher training does NOT produce the expected ordering: on t8 the matched spec_T8 is the *worst* alternative (2.8182), and the best alternative there is spec_T4 (3.1461). The only target where the matched rotation is nominally top of the alternatives is t4kv4 (spec_T4KV4 2.4268 vs spec_T4 2.4054, spec_T8 2.4046, mixed 2.3800) — differences within screening noise. No alternative comes within 0.25 of shared_RT on any target; deficits by 5-dataset mean are 0.25–0.58 (t8), 0.56–0.84 (t4), 0.81–0.86 (t4kv4).

### 9.4 Paired deltas vs shared_RT, Holm-corrected (`paired_deltas_holm.csv`)

Every rotation-vs-shared comparison is a paired per-prompt bootstrap (n=20 common prompts), Holm-corrected within target family:

| family | comparisons | per-dataset delta range (alt − shared) | p_holm | significant at 0.05 |
|---|---|---|---|---|
| t4 | 55 (11 alternative legs × 5 datasets; 10 distinct rotations, one duplicate leg) | −0.4547 (selfrecon, mtbench) to −1.1550 (drot_none, humaneval) | 0.0275 | 55/55 |
| t8 | 20 (4 × 5) | −0.2112 (spec_T4, mtbench) to −0.7723 (spec_T8, humaneval) | 0.01 | 20/20 |
| t4kv4 | 20 (4 × 5) | −0.5321 (spec_T4KV4, mtbench) to −1.1854 (mixed, humaneval) | 0.01 | 20/20 |

All 95 rotation comparisons favor shared_RT and all survive Holm (file-wide: 100/114 significant; the 14 non-significant rows are the Phase E KV4 deltas and Phase C/D restored-control rows, as expected). **shared_RT beats every alternative rotation on every target and every dataset.**

### 9.5 Alpha sweep (spec_T4, t4 target, mtbench-20 screening)

| alpha | 16 | 22.6 | 32 (saved)* | 45.25 | 64 |
|---|---|---|---|---|---|
| micro-AL | 1.9067 | 2.0275 | 2.1201 | **2.2314** | 2.1950 |

\* The alpha=32 point is the deduped `drot__spec_T4__t4__mtbench` cell (all checkpoints saved alpha=32.0).

Post-hoc alpha recalibration gives +0.11 (2.1201 → 2.2314 at alpha=45.25) — real but ≈5.7× smaller than the −0.63 mtbench deficit vs shared_RT (2.7492). Since alpha was never co-optimized with R_D during training (all ckpts saved alpha=32.0), the per-rotation optimum was left on the table; the verdict is unchanged either way.

### 9.6 Gate B: R_D is a gauge in fp16

With the fp16 draft (d16, noquant), shared vs spec_T4 rotation on t4/mtbench: micro-AL **2.9570 vs 2.9570**, identical **861** cycles, identical tau-bucket profiles (3.17 / 3.165 / 2.8449). Per-prompt tau sequences are bit-identical on 19/20 prompts; the strict bit-identity gate in `gateB_rd_gauge_equivalence.json` therefore records verdict FAIL on the single remaining prompt — one tau flip attributable to fp16 round-off, with zero effect on any pooled statistic. Interpretation: **in fp16 numerics R_D is a pure gauge; the basis only matters once the quantizer acts in it. All acceptance loss of alternative rotations under W4A4 is quantization–basis interaction.**

### 9.7 Central mechanistic conclusion

Three facts triangulate the mechanism:

1. **fp16 gauge freedom (Gate B):** changing R_D with an fp16 draft changes nothing — there is no "representation quality" effect to train for.
2. **Uniform W4A4 loss for every non-R_T basis:** under the deployed W4A4 draft, the identity (drot_none, t4 mean 2.3859), a random-init-trained rotation (2.4678), and every objective-trained rotation (2.3668–2.6511) all land in the same band, ~0.56–0.84 below shared_RT (3.2067) — regardless of objective, data mix, or teacher.
3. **Proxy–runtime disconnect:** val_top1_d1 spans 0.057–0.358 across checkpoints, yet runtime micro-AL is flat across them.

Conclusion: the R_T basis is *quantization-friendly* — SpinQuant's learned target rotation bakes in Hadamard-like activation smoothing, so W4 MSE clip search and per-token a4 ranges are calibrated to well-conditioned distributions. In any other basis (identity, random, or trained-by-KL), those same quantizers degrade roughly uniformly. The training objectives optimize teacher agreement, which the fp16 gauge freedom makes vacuous; they never see the quantity that actually differs between bases — acceptance under the deployed quantizer.

## 10. Decision framework applied

**Criterion 1 (pre-registered): a trained R_D must beat shared R_T on its deployed target in screening before advancing to the confirmatory matrix.** Result: FAILS everywhere — 95/95 paired comparisons favor shared_RT, all Holm-significant (§9.4), on all three targets and all five datasets. The reduced evaluation matrix (no Phase-E-scale confirmatory runs for any alternative rotation) is therefore justified by the framework, not by convenience.

**Recommendation.** Deploy **shared R_T** (R_D = R_T) for the W4A4 draft. Do **not** train draft rotations at this budget. The deployed full-quantization operating point with shared R_T is T4_KV4/D4P3_KV4 micro-AL = 3.0080 vs T4_KV16/D4P3_KV16 = 2.9973 (Phase E, 80 mtbench prompts; KV4 without R3) — draft+target KV4 is acceptance-neutral, so the shared-R_T recommendation extends unchanged to the KV4 deployment.

**What would change the verdict.** Any of the following, none tested here:

1. **Longer training.** 400–500 steps, single seed; the uniform-band result could in principle be a budget artifact, though the identity-vs-trained tie argues against easy gains.
2. **A quantization-aware acceptance objective.** The mechanistic finding (§9.7) says the failure is objective mismatch, not optimization failure: a differentiable surrogate of tree acceptance *under the deployed fake-quantizers* (clip search and a4 ranges inside the loss) is the direct fix; KL/top-1 proxies are provably blind to the basis in fp16.
3. **Joint alpha–R_D optimization.** Alpha was frozen at 32.0 in training; post-hoc recal alone gave +0.11 (§9.5). Co-optimization could compound with (2).
4. **Draft-weight finetuning jointly with R_D.** Rotation-only search leaves the weights calibrated to R_T's statistics; letting weights adapt to the new basis removes the constraint that currently makes R_T the unique quantization-friendly point.

## 11. Limitations and threats to validity

1. **Screening sample size.** All Phase G rotation cells are 20 pinned prompts/dataset (675–1350 pooled cycles). Deltas of 0.21–1.19 with p_holm ≤ 0.0275 are far from the noise floor, but absolute micro-AL values from these cells should not be quoted as confirmatory; only the Phase E 4×4 matrix (80 mtbench prompts) is.
2. **Single seed, short training.** One training seed per checkpoint, 400–500 steps (300 for the early TARGET_KL ckpt). We cannot exclude that a different seed or 10× budget escapes the uniform band, only that nothing in the 20-checkpoint grid did.
3. **Fake quant only.** Every number in this study is fake-quantized; no real INT4/INT8 kernels, no latency or memory-bandwidth claims of any kind.
4. **Alpha sweep scope.** The sweep (§9.5) is mtbench-only, t4-only, spec_T4-only, and post-hoc; no checkpoint was trained with alpha co-optimization (all saved alpha=32.0). The +0.11 recal gain may not transfer across datasets or rotations.
5. **KV4 without R3.** All KV4 results use per-head/per-token asymmetric quantization at append time (group = head_dim = 128) with the R3 online rotation disabled. Claims like "KV4 is acceptance-neutral" are specific to this "KV4 without R3" configuration.
6. **Verifier-correctness caveat (S26).** micro-AL under quantized targets measures acceptance against the *quantized* verifier's greedy continuation, which diverges from target-only AR decoding early: mean greedy prefix-match fraction 1.0000 (T16_D16, exact) vs 0.1992 (T4KV16_D4P3) and 0.1742 (T4KV4_D4P3KV4), deterministic across reruns. EAGLE's lossless-verification guarantee holds only in fp16 numerics; the divergence mechanism is batch-shape sub-ulp numerics amplified by a4 activation-quant bin flips. Acceptance comparisons between rotations remain internally valid (same verifier per cell), but micro-AL under W4A4 is not acceptance against the fp16 model's text.
7. **Gate B strict verdict.** The bit-identity gate JSON records FAIL (19/20 identical); the gauge-equivalence conclusion rests on identical pooled micro-AL (2.9570), identical cycle counts (861), and the single mismatch being one fp16-roundoff tau flip — an interpretation, clearly labeled as such.
8. **macro-AL excluded by design.** All acceptance numbers are micro-AL (Σtau/n_cycles pooled); per-prompt macro averages were never computed and no claim should be read as macro-AL.
9. **Environment deviations.** Root disk reached 0B mid-study (external users); outputs/runs/rotations were relocated to /data/thahn1230 with symlinks and TMPDIR moved; several OOM/relaunch cycles; a shell-precedence bug in one screen launcher created a dead-job window (relaunched); duplicate identity/alpha legs are deterministic and were deduplicated by shard-skip (verified identical, e.g. full_eagle ≡ spec_T4). None of these affect measured values, but they are part of the provenance record. Long-context (Phase F) cells at L ≥ 2048 additionally required the `long_ea_generate.py` replica fixing the stock `ea_model.py:356` hard-coded 1960-token break.

## 6. Phase E — Target x Draft KV-bit matrix (4x4, 80 MT-bench prompts)

All cells: fake-quant, greedy, tree `mc_sim_7b_63`, 128 new tokens, 80 mtbench prompts (the confirmatory-scale Phase E set, not the 20-prompt screening pool). Draft is the shared-R_T concat-selective adapter throughout (D16 = fp16/noquant; D8 = fake W8A8; D4P3 = fake W4A4 + exact P3 embedding scaling, folded first path). Every KV4 leg is **KV4 without R3**: per-head/per-token asymmetric, group = head_dim = 128, quantized at cache-append time. Values are micro-AL [95% CI] (n_cycles), from `micro_al_summary.csv` / `kv4mat__*` shards.

| Target \ Draft | D16 | D8 | D4P3 (KV16) | D4P3 (KV4) |
|---|---|---|---|---|
| **T16 (fp16), KV16** | 3.5762 [3.4681, 3.6803] (2789) | 2.1099 [2.0557, 2.1640] (4670) | 2.9065 [2.8122, 3.0024] (3424) | 2.8839 [2.7933, 2.9777] (3428) |
| **T8 (W8A8), KV16** | 3.5396 [3.4276, 3.6488] (2828) | 3.3539 [3.2558, 3.4546] (3001) | 3.3588 [3.2532, 3.4716] (2974) | 3.3314 [3.2226, 3.4417] (3008) |
| **T4 (W4A4), KV16** | 3.2749 [3.1669, 3.3785] (3067) | 2.8953 [2.8095, 2.9761] (3456) | 2.9973 [2.9093, 3.0893] (3329) | 3.0087 [2.9185, 3.1104] (3333) |
| **T4 (W4A4), KV4** | 3.3116 [3.1869, 3.4429] (3030) | 2.9454 [2.8445, 3.0537] (3405) | 3.0555 [2.9529, 3.1681] (3242) | 3.0080 [2.9004, 3.1221] (3356) |

(The low T16/D8 cell, 2.1099, is the exposed-rotated-interface effect established in the Phase C/D causal panel — a basis mismatch, not W8 damage — and is orthogonal to the KV question.)

### The three KV4 claims (paired per-prompt deltas, 80 common prompts, bootstrap p, Holm within family; `paired_deltas_holm.csv`, delta = second minus first)

**1. Draft KV4 is ~free.** Quantizing the draft's own KV cache to 4 bits moves micro-AL within noise on both the fp16 and the W4A4 target:

| Comparison | delta | 95% CI | p_boot | p_holm | sig. |
|---|---|---|---|---|---|
| T16: D4P3_KV16 → D4P3_KV4 | −0.0226 | [−0.0568, +0.0082] | 0.177 | 1.0 | no |
| T4_KV16: D4P3_KV16 → D4P3_KV4 | +0.0114 | [−0.0425, +0.0641] | 0.695 | 1.0 | no |

**2. Target KV4 (without R3) is slightly *positive* for micro-AL — but not significant.** With the D4P3_KV16 draft, T4_KV16 → T4_KV4 gives +0.0582 [−0.0206, +0.1370], p_boot 0.151, p_holm 1.0. The direction is consistent across draft columns (D16: 3.2749 → 3.3116, +0.0367; D8: 2.8953 → 2.9454, +0.0501; unpaired cell differences, no Holm rows for these), so we state it as a small, directionally consistent, non-significant increase — not a claimed gain.

**3. Full W4A4KV4 deployment ≈ KV16.** The all-in cell T4_KV4/D4P3_KV4 = 3.0080 vs T4_KV16/D4P3_KV16 = 2.9973: paired delta +0.0107 [−0.0659, +0.0882], p_boot 0.75, p_holm 1.0. Adding 4-bit KV on both sides of the already-W4A4 stack costs nothing measurable in acceptance length.

**Audit line.** Per-cell counters confirm the fake KV4 path actually ran (no silent fp16 fallback): draft-KV4 cells quantized 51,203–55,268 draft KV tokens with k/v NMSE ≈ 0.0139–0.0142 / 0.0099; target-KV4 cells quantized 2.85M–12.16M target KV tokens with k/v NMSE ≈ 0.0317 / 0.0099 (identical to 3 decimals across cells, as expected for a deterministic target prefix distribution). Invocation audit: `kv4_invocation_audit.csv` 64/64 PASS.

## 7. Phase F — Context-length sensitivity (wikitext prefixes, L = 256–4096)

Setup: 12 wikitext-2-test prefixes per length bucket (held out from calibration), 64 new tokens each, greedy, tree `mc_sim_7b_63`; all quantized legs fake-quant; KV4 = KV4 without R3. Micro-AL per cell (n_cycles 372–541), from `ctx__*.csv` shards. Absolute levels are lower than Phase E because wikitext continuation is a different domain from the chat-style eval prompts; read trends, not levels.

| Config | L=256 | L=512 | L=1024 | L=2048 | L=4096 |
|---|---|---|---|---|---|
| T4_KV16 / D16 | 2.0210 | 1.9406 | 2.0287 | 1.9453 | 1.5553 |
| T4_KV4 / D16 | 1.9696 | 2.0941 | 2.0556 | 2.0885 | 1.6300 |
| T4_KV16 / D4P3_KV16 | 1.8886 | 1.8908 | 1.7844 | 1.8675 | 1.4251 |
| T4_KV16 / D4P3_KV4 | 1.9975 | 1.8657 | 1.8640 | 1.8534 | 1.4501 |
| T4_KV4 / D4P3_KV4 | 1.9091 | 1.9140 | 1.8365 | 1.9521 | 1.4724 |

**Decay is baseline-driven, not a KV4 effect.** Every configuration — including the fp16-draft, KV16 controls — is roughly flat from 256 to 2048 and then drops sharply at 4096 (1.43–1.63 across the board, from ~1.9–2.1 at short context). The full-KV4 deployment cell T4_KV4/D4P3_KV4 sits at or **above** its KV16 control T4_KV16/D4P3_KV16 at every length (deltas +0.021, +0.023, +0.052, +0.085, +0.047), i.e., the KV4 increment is ≈0 at all context lengths tested; the long-context acceptance decay is a property of the EAGLE draft itself. Quantizer health is stable with length: target-KV4 k-NMSE drifts 0.0318 → 0.0305 from 256 to 4096 (v ≈ 0.0099 flat); draft-KV4 k/v NMSE ≈ 0.0124 / 0.0099 at all lengths.

**The 1960-token stock-EAGLE cap.** Stock `EaModel.ea_generate` hard-codes `if input_ids.shape[1] > 1960: break` (`third_party/EAGLE/eagle/model/ea_model.py:355-356`). For any prompt already longer than 1960 tokens this silently terminates after the first verification cycle, so naive L=2048/4096 buckets would be invalid (≈1 cycle per prompt) while *appearing* to run. Since `third_party` is read-only in this repo, the fix is a replica generator, `src/eagle_spinquant/long_ea_generate.py`, identical in behavior (same `eagle.model.utils` calls; same cycles as stock for sub-1960 prompts) but with the cap replaced by the true KV-buffer capacity (`max_position_embeddings` minus tree transient minus 8). All L ≥ 2048 numbers above were measured only with this fix.

**L=2048/4096 caveats.** (i) 12 prompts per cell and no CIs — screening-scale evidence; the flat-then-drop shape and the KV4≈0 increment are the supported conclusions, not precise levels. (ii) Domain is wikitext continuation, not chat; levels do not transfer to the mtbench cells of Phase E. (iii) The Llama-2 KV buffer is 4096 positions, so the L=4096 bucket uses prompts truncated to L−136 = 3960 tokens (margin for 64 new tokens + tree transient; margin is 8 at shorter L) — it is a ~3960-token-prompt bucket in practice. Peak memory at L=4096 was 20.83–22.03 GB, close to the 24 GB card limit, leaving little headroom for larger trees or longer generations at this length. Fake-quant only; no latency claims anywhere in this study.

## 1. Executive summary

**Study question.** Under a fake-quantized W4A4 (and W4A4+KV4) deployment of Llama-2-7b-chat with a learned SpinQuant-style rotation, can a *draft-side* rotation R_D — trained against the deployed quantized target's top-64 logits — beat the default of sharing the target rotation R_T in the EAGLE-1 draft? All results are fake-quant acceptance-length measurements (micro-AL); no latency or throughput claims are made.

**Headline answer: NO.** The shared target rotation R_T dominates every trained alternative on every deployed target (t8, t4, t4kv4) and every dataset. In fp16 the draft rotation is a pure gauge (Gate B), so the entire acceptance gap between rotations is a quantization-basis interaction — and the training objectives explored here do not find a better basis than R_T. Separately: KV4 (without R3) is nearly free for acceptance, and the causal panel shows the dominant lever on micro-AL is whether the rotated interface is *exposed* to the stock draft, not weight quantization damage.

Key results (micro-AL = sum(tau)/n_cycles pooled over cycles; 20-prompt cells are screening, not confirmatory):

- **shared_RT wins everywhere (screening, 20 prompts/dataset).** On t4: mtbench 2.7492, sharegpt 3.0469, c4 2.9386, gsm8k 3.5568, humaneval 3.7420. The best alternative on t4/mtbench is selfrecon at 2.2945; all 11 alternative legs (10 distinct rotations — full_eagle and spec_T4 are deterministic duplicate legs of the same t4-teacher checkpoint — plus drot_none = identity) sit 0.45–1.16 micro-AL below shared_RT depending on cell. Same ordering on t8 and t4kv4 (e.g. t4kv4/humaneval: shared_RT 3.8000 vs best alternative 2.7096).
- **Statistics.** All 95 shared_RT-vs-alternative paired deltas are negative and Holm-significant at 0.05 (paired_deltas_holm.csv: 100/114 rows significant overall; the 14 non-significant rows are the KV4 deltas — consistent with KV4 being free — and panel controls).
- **Gate B (gauge test).** With a d16 (noquant) draft, shared vs spec_T4 rotations give identical micro-AL 2.9570, identical 861 cycles, and bit-identical tau sequences on 19/20 prompts (1 fp16-roundoff flip; the strict 20/20 bit-identity gate therefore records FAIL, but functional equivalence holds). R_D is a gauge in fp16; all acceptance loss under W4A4 is quantization-basis interaction.
- **Alpha recalibration helps but does not change the verdict.** spec_T4 on t4/mtbench: 1.9067 (a=16), 2.0275 (a=22.6), 2.1201 (a=32, as saved), 2.2314 (a=45.25, best), 2.1950 (a=64) — a +0.11 gain, still 0.52 below shared_RT (2.7492). All trained checkpoints saved alpha=32.0; alpha was never co-optimized with R_D (documented limitation).
- **KV4 without R3 is nearly free (Phase E, 80 mtbench prompts).** Draft KV4: T16 target 2.9065 (KV16) vs 2.8839 (KV4); T4 target 2.9973 vs 3.0087 — both within noise (Holm p=1.0). Target KV4 slightly *raises* micro-AL (T4_KV16/D16 3.2749 → T4_KV4/D16 3.3116). Full 4-bit deployment T4_KV4/D4P3_KV4 = 3.0080 vs T4_KV16/D4P3_KV16 = 2.9973. Counters confirm real invocation: draft k/v NMSE ~0.014/0.010, target k NMSE ~0.032, all tokens counted, no fp16 fallback (audit 64/64 PASS).
- **Interface exposure, not W8 damage, explains the T8_D8 vs T16_D8 gap (Phases C/D, 80 prompts, 5 datasets).** Delta_interface,FP16 (TR16_ROTATED_D8 minus T16_IDENTITY_D8) = +1.16 to +1.42 on all 5 datasets (all Holm-significant); Delta_W8 (T8_ROTATED_D8 minus TR16_ROTATED_D8) ≈ 0 (|delta| ≤ 0.043, none significant). Restoring the interface returns micro-AL to identity level (e.g. mtbench 2.1184 vs 2.1099).
- **Quality context (Phase B).** Official wikitext PPL: fp16 6.9452, W8A8 6.9491, W4A4 6.9629, untrained random-Hadamard 10.6272. Incremental KV4 CE cost: +0.013 (fp16), +0.008 (w8a8), +0.043 (w4a4), +0.075 (rh0) — KV4 without R3 is cheap on a well-conditioned rotation, expensive on a bad one.
- **Context length (Phase F, wikitext prefixes, 12 prompts x 64 tokens).** micro-AL decays with prefix length for all configs (~1.89–2.02 at L=256 → 1.43–1.63 at L=4096); the KV4 increment is ≈0 at every L (T4KV4_D4P3KV4 ≥ its KV16 control at every L, e.g. 1.4724 vs 1.4251 at 4096). L≥2048 required the long_ea_generate fix (Section 3).
- **Verifier-correctness caveat (S26).** Greedy prefix-match fraction vs target-only AR decode: T16_D16 = 1.0000 (exact, 10/10 prompts); T4KV16_D4P3 = 0.1992; T4KV4_D4P3KV4 = 0.1742. EAGLE's lossless-verification guarantee holds only in fp16 numerics; under fake quant, batch-shape sub-ulp numerics are amplified by a4 activation-quant bin flips into different (deterministic, reproducible) greedy continuations. Quantized micro-AL numbers measure acceptance dynamics, not exact AR-equivalence.

## 2. Setup & contracts

**Models.** Target: Llama-2-7b-chat. Draft: yuhuili/EAGLE-llama2-chat-7B (EAGLE-1). Fake quantization only throughout — quantize/dequantize in fp16 compute; no real INT kernels, no latency claims.

**Target quantization contract.** Learned chat rotation `learned_chat_w4a4kv16` (R1 + 32 per-layer R2), R4 online Hadamard active, R3 unused. Deployed targets: **t8** = W8A8, **t4** = W4A4, **t4kv4** = W4A4 + target-side KV4. KV4 contract: per-head, per-token asymmetric quantization, group = head_dim = 128, applied at cache-append time; every KV4 claim in this study is "KV4 without R3".

**Draft configs.** **D16** = fp16 draft through the concat-selective adapter (noquant); **D8** = fake W8A8; **D4P3** = fake W4A4 + P3 exact embedding scaling (alpha, folded into weights). D4P3 uses gamma_R1 on the first path and ar_r2r4. Motivation from the Phase A grid: naive full-draft W4A4 collapses micro-AL to 1.2521, while the embedding-scaled (P3) policy recovers 2.9973 (w4a4 target, mtbench).

**Draft rotation R_D and interface algebra.** Reconstructed concat projection W_rec = [W_e | W_h·R_D]; the post-projection hidden basis is R_D; the AR block is conjugated by R_D; the draft LM head becomes W_lm·R_D. The first path keeps the target-basis bridge via `first_fold_R` (R_T-to-R_D change of basis folded into weights — never a runtime GEMM). `shared` means R_D == R_T (the validated stock path); `drot_none` means R_D == I.

**Trained rotations.** Wave 1: 13 objective-ablation checkpoints (t4 teacher cache, 400 steps; objectives combining deployKL / rank / feature / self / targetCE / dual, plus gamma and tau variants). Wave 2: spec_T8, spec_T4, spec_T4KV4, mixed, randominit, wikionly. Trainer: STE fake quant (W4 MSE clip search, per-token a4), Adam with QR retraction on the orthogonal manifold, teacher = deployed-target top-64 logits, T_WIN=48, K_DEPTH=4. All checkpoints saved alpha=32.0 — alpha was not co-optimized with R_D (limitation; the a45.25 sweep point shows +0.11 was left on the table without changing the verdict).

**Metric.** micro-AL only (never macro): sum(tau)/n_cycles pooled over all verification cycles, tau = accepted+1 per cycle. Tree `mc_sim_7b_63`, greedy decoding (temperature 0); Phases C–E and G use prompts truncated to 1024 tokens with 128 new tokens (Phase F uses wikitext prefixes up to L=4096 with 64 new tokens).

**Datasets / manifests.** Screening: 20 pinned eval prompts per dataset on mtbench, sharegpt, gsm8k, humaneval, c4 (shards `drot__*`). Phase E confirmatory-scale: 80 mtbench prompts (`kv4mat__*`, `panel__*`). Phase F: wikitext prefixes L in {256, 512, 1024, 2048, 4096}, 12 prompts x 64 new tokens (`ctx__*`). Phase B PPL: wikitext.

**Gates.**

| Gate | Check | Status |
|---|---|---|
| Gate D | Phase A baseline reproduction: 9/9 micro-AL cells vs raw records | PASS (max abs diff 0.0004) |
| Gate G | Rotation orthogonality error < 1e-3 for all trained R_D | PASS — asserted at train time (values not persisted; NaN in rotation_checkpoints.csv) and re-established post hoc by recomputation from all 20 saved checkpoints, max err 9.54e-07 (Section 8) |
| Gate B | fp16 gauge equivalence, shared vs spec_T4, d16 draft | Strict bit-identity verdict **FAIL** (19/20 prompts identical, 1 fp16-roundoff tau flip); functional equivalence holds: identical micro-AL 2.9570, identical 861 cycles |
| Gate B3 | KV4 invocation audit (kv4 invoked / kv16 clean / no fp16 fallback / fake-quant mode, 16 cells x 4 checks) | PASS 64/64 |
| S26 | Verifier correctness, T16_D16 greedy prefix match vs AR | PASS (1.0000 exact); quantized cells 0.1992 / 0.1742 — guarantee is fp16-only (documented caveat) |

## 3. Environment & deviations

- **Root-disk crisis.** The root disk (~100% full from external users) hit 0 bytes free mid-study. `outputs/`, `runs/`, and rotation checkpoints were relocated to `/data/thahn1230` with symlinks left in place; `TMPDIR=/data/thahn1230/tmp`. No results were lost; affected jobs were rerun after relocation.
- **GPU policy.** This study's spec mandated GPUs 0–5 only (never 6/7), overriding the repo's earlier 4–7-only policy from previous studies. Enforced per job with a single-device `CUDA_VISIBLE_DEVICES` and an assertion in the eval scripts; recorded launches used devices in {0..5} only.
- **Relaunch incidents.** Several OOM/relaunch cycles occurred. One notable scheduler bug: a screen-5 launch of the form `A && B & C` backgrounded the R_D assignment due to shell operator precedence, silently creating a dead-job window; the legs were relaunched as tm_a/tm_b/tm_c. Benign duplicate identity/alpha legs were executed at various points; all were deterministic with identical results and were deduplicated by shard-skip logic (this is also why the a=32 alpha-sweep point is served by the base spec_T4 shard).
- **long_ea_generate fix (required for Phase F L≥2048).** Stock EAGLE `ea_generate` hard-codes a loop break when `input_ids` exceeds 1960 tokens (ea_model.py:356), which silently truncates generation for long prefixes. A replica generator, `src/eagle_spinquant/long_ea_generate.py`, removes the cap; all Phase F cells at L=2048 and L=4096 were measured only with that fix. Cells at L≤1024 are unaffected.
- **Determinism note.** The S26 divergences and all rerun checks were bit-deterministic across reruns on this stack; discrepancies reported above are systematic (quant-basis numerics), not run-to-run noise.

## 4. Phases A/B — baseline reproduction, target quality, KV4 cost, and verifier correctness

All measurements are fake-quant only (no latency claims). Micro-AL = sum(tau)/n_cycles pooled over verification cycles (tau = accepted+1), tree `mc_sim_7b_63`, greedy, 128 new tokens, prompts truncated to 1024 tokens. Source tables: `tables/RESULTS_DIGEST.md` in the run dir; official-PPL rows from `artifacts/spinquant_ppl_reproduction_fix/summary/`.

### 4.1 Phase A — baseline micro-AL reproduction (Gate D)

The 3×3 target×draft grid (mtbench, 80 prompts) was recomputed from raw per-cycle acceptance records and checked against the previously recorded values (`phaseA_baseline_micro_al.csv`):

| cell | micro-AL (recomputed) | expected | diff | n_cycles |
|---|---|---|---|---|
| T16_D16 | 3.5762 | 3.576 | +0.0002 | 2789 |
| T16_D8 | 2.1099 | 2.110 | −0.0001 | 4670 |
| T16_D4 | 1.0444 | 1.044 | +0.0004 | 9378 |
| T8_D16 | 3.5396 | 3.540 | −0.0004 | 2828 |
| T8_D8 | 3.3539 | 3.354 | −0.0001 | 3001 |
| T8_D4 | 1.2690 | 1.269 | 0.0000 | 7817 |
| T4_D16 | 3.2749 | 3.275 | −0.0001 | 3067 |
| T4_D8 | 2.8953 | 2.895 | +0.0003 | 3456 |
| T4_D4 | 1.2521 | 1.252 | +0.0001 | 8006 |

**Gate D: PASS.** 9/9 cells within the ±0.002 tolerance; max |diff| = 0.0004. All downstream phases build on this grid.

### 4.2 Phase B — official target-quality PPL (wikitext-2, official SpinQuant evaluator, Llama-2-7b-chat-hf)

From `corrected_target_quality.csv`:

| target | PPL | CE | ΔCE vs fp16 |
|---|---|---|---|
| fp16 | 6.9452 | 1.93805 | — |
| learned rotation, RTN W8A8, KV16 | 6.9491 | 1.93861 | +0.00056 |
| learned rotation, RTN W4A4, KV16 | 6.9629 | 1.94060 | +0.00255 |
| learned rotation, GPTQ W4A4, KV16 | 8.4600 | 2.13535 | +0.19730 |
| random-Hadamard RTN W4A4, KV16 (control) | 10.6272 | 2.36342 | +0.42537 |

The learned chat rotation (`learned_chat_w4a4kv16`, R1+32×R2, R4 online, R3 unused) brings W4A4 to within ΔCE +0.0026 of fp16; the random-Hadamard control sits at +0.4254, i.e. the learned rotation removes ~99% of the naive-rotation CE gap at W4A4.

### 4.3 Phase B — incremental KV-sensitive PPL/CE and the cost of KV4 (KV4 without R3)

Separate protocol (`incremental_kv_ppl.csv`): CE/PPL scored only on 3,840 KV-sensitive token positions in long contexts, with 131,072 K and 131,072 V cache tokens quantized per config. These PPLs are not comparable to the official table above (different token set). All KV4 here and everywhere in this report is per-head/per-token asymmetric, group = head_dim = 128, applied at cache-append time, **without R3**.

| backbone | rotation | CE KV16 | CE KV4 | ΔCE (KV4 cost) | PPL KV16 → KV4 | K NMSE | V NMSE |
|---|---|---|---|---|---|---|---|
| fp16 | none | 1.78927 | 1.80190 | +0.0126 | 5.9851 → 6.0611 | 0.0304 | 0.0108 |
| W8A8 | learned | 1.78930 | 1.79719 | +0.0079 | 5.9853 → 6.0327 | 0.0304 | 0.0099 |
| W4A4 | learned | 1.79718 | 1.84010 | +0.0429 | 6.0326 → 6.2972 | 0.0301 | 0.0099 |
| W4A4 | random-Hadamard | 2.20991 | 2.28485 | +0.0749 | 9.1149 → 9.8242 | 0.0302 | 0.0099 |

KV quantization error is essentially constant across backbones (K NMSE ≈ 0.030, V NMSE ≈ 0.010), yet the CE cost of KV4 grows as backbone precision drops: +0.013 (fp16) / +0.008 (W8A8) / +0.043 (W4A4 learned) / +0.075 (W4A4 random-Hadamard). The same KV error is amplified by an already-degraded backbone. Positional buckets show the W4A4 cost is spread across ranges, including long range (ppl_2048_4095: 7.4444 → 7.7911).

### 4.4 KV4 implementation and B3 invocation audit (64/64)

KV4 is implemented as fake quantization of K and V at append time on whichever caches (draft and/or target) the cell configures; no R3 rotation is applied anywhere in the KV path. The B3 audit (`kv4_invocation_audit.csv`) instruments every Phase-E cell — 16 (target, draft, KV) configurations × 4 checks: (i) KV4 actually invoked where configured / KV16 paths clean of any KV quantization, (ii) the other cache untouched, (iii) no fp16 fallback taken, (iv) fake-quant mode active. **Result: 64/64 PASS.** Every KV4 number in this report is backed by a positive invocation check, not just a config flag.

### 4.5 S26 — verifier correctness under quantization

Study: 10 prompts (ids 81–90), 128 greedy tokens each (one prompt terminated at 87), comparing the speculative (tree-mode) output token-by-token against target-only autoregressive decoding of the same target; metric = fraction of the generation that matches as a prefix (`verifier_correctness.csv`, `vc__*.csv`):

| config | mean prefix-match fraction | per-prompt range |
|---|---|---|
| T16_D16 | 1.0000 | 1.0000 – 1.0000 (10/10 exact) |
| T4KV16_D4P3 | 0.1992 | 0.0312 – 0.4688 |
| T4KV4_D4P3KV4 | 0.1742 | 0.0078 – 0.4375 |

The fp16 cell is bit-exact on all 10 prompts, validating the harness and confirming EAGLE's lossless-verification guarantee in fp16. Under W4A4 the guarantee breaks. Mechanism: tree-mode verification evaluates the target on a different batch shape than AR decoding, producing sub-ulp numerical differences; under a4 per-token activation quantization these sub-ulp differences flip quantization bins, which perturbs logits enough to flip the greedy argmax at some step, after which the two decodes diverge permanently (prefix-match measures the divergence point). The behavior is deterministic across reruns — this is not nondeterminism but two deterministic decodes of two numerically distinct functions.

Implication: every micro-AL number in this study remains internally valid — acceptance is defined against the deployed tree-mode target, and the verifier *is* that target — but EAGLE's exact-output-equivalence guarantee is a property of fp16 numerics only. A deployed W4A4 speculative system produces a deterministic output that is not token-identical to W4A4 target-only decoding; any end-task quality claim for W4A4 speculative decoding must therefore be evaluated on the speculative output itself, not inferred from the target.

## 5. Phases C/D — restored-vs-exposed causal panel: the T8_D8 gap is an interface effect

### 5.1 Question and design

Phase A shows T8_D8 = 3.3539 vs T16_D8 = 2.1099 (mtbench): the same fake-W8A8 draft achieves +1.24 micro-AL under the deployed W8A8 target versus the fp16 target. Two candidate causes: (a) W8A8 damage to the target itself, or (b) the interface — under the deployed rotated target, the hidden state handed to the draft is in the rotated basis (exposed) rather than the original basis. The panel separates these with five arms, run on 5 datasets (c4, gsm8k, humaneval, mtbench, sharegpt) × 80 prompts, with both D8 (treatment) and D16 (control) drafts:

- **C0** `T16_IDENTITY_D8` — fp16 target, original-basis interface (baseline)
- **C1** `TR16_ROTATED_D8` — rotated fp16 target (rotation applied, zero quantization), draft exposed to the rotated interface
- **C2** `T8_ROTATED_D8` — W8A8 rotated target, exposed interface (= the deployed T8_D8 configuration)
- **C3** `TR16_RESTORED_D8` — rotated fp16 target, interface un-rotated back to the original basis before the draft
- **C4** `T8_RESTORED_D8` — W8A8 target, restored interface

### 5.2 Panel micro-AL (D8 draft, 80 prompts per dataset)

| dataset | C0 identity | C1 rot-fp16 exposed | C2 W8 exposed | C3 rot-fp16 restored | C4 W8 restored |
|---|---|---|---|---|---|
| c4 | 1.9601 | 3.1225 | 3.1173 | 1.9691 | 1.9915 |
| gsm8k | 2.3009 | 3.6997 | 3.7314 | 2.2921 | 2.2937 |
| humaneval | 2.2863 | 3.7106 | 3.7113 | 2.2851 | 2.2661 |
| mtbench | 2.1099 | 3.3643 | 3.3539 | 2.1184 | 2.1096 |
| sharegpt | 2.0340 | 3.3494 | 3.3065 | 2.0493 | 2.0501 |

The pattern is identical on all 5 datasets: both EXPOSED arms are high, all three original-basis arms (identity and both RESTORED) are low — and C1, which contains no quantization at all, already reproduces the deployed C2 value.

### 5.3 Delta decomposition (paired over 80 common prompts; bootstrap p; Holm-corrected)

Decomposition of the total gap: (C2 − C0) = Δ_interface,FP16 (C1 − C0) + Δ_W8 (C2 − C1). From `paired_deltas.csv` / `paired_deltas_holm.csv`:

| dataset | total gap C2−C0 | Δ_interface,FP16 (C1−C0) [95% CI] | Δ_W8 (C2−C1) | restore residual C3−C0 |
|---|---|---|---|---|
| c4 | +1.1572 | **+1.1624** [1.1105, 1.2161], p_holm 0.0095 | −0.0051 (p 0.827, ns) | +0.0091 (p 0.255, ns) |
| gsm8k | +1.4305 | **+1.3988** [1.3449, 1.4551], p_holm 0.0095 | +0.0317 (p 0.089, ns) | −0.0088 (p 0.210, ns) |
| humaneval | +1.4250 | **+1.4243** [1.3696, 1.4777], p_holm 0.0095 | +0.0007 (p 0.990, ns) | −0.0013 (p 0.850, ns) |
| mtbench | +1.2440 | **+1.2545** [1.1888, 1.3220], p_holm 0.0095 | −0.0104 (p 0.651, ns) | +0.0085 (p 0.135, ns) |
| sharegpt | +1.2725 | **+1.3154** [1.2487, 1.3876], p_holm 0.0095 | −0.0429 (p 0.143, ns) | +0.0152 (raw p 0.006; p_holm 0.084, ns) |

Δ_interface + Δ_W8 reproduces the total gap to the 4th decimal on every dataset. Δ_interface,FP16 is +1.16 to +1.42 and Holm-significant everywhere; Δ_W8 spans −0.043 to +0.032 and is never significant (p_holm = 1.0 in all cases); the restore residual (C3 − C0) is within ±0.016 of zero and never Holm-significant. C4 confirms the W8 side independently: the full W8A8 target with a restored interface lands on the identity baseline on every dataset (e.g. mtbench 2.1096 vs 2.1099).

**D16 control.** The D16 rows are nearly constant across all five arms of every dataset (per-dataset spread ≤ 0.047, e.g. mtbench 3.5393–3.5762, humaneval 4.0039–4.0073): the target-side manipulations barely change the target itself, so the entire D8 effect is attributable to what basis the draft's input interface is in.

### 5.4 Conclusion

The T8_D8 vs T16_D8 micro-AL gap is, on all 5 datasets, entirely an exposed-rotated-interface effect and reproducible in pure fp16 with no quantization anywhere (Δ_interface,FP16 ≈ the full gap, Holm-significant on 5/5 datasets); W8A8 weight/activation damage contributes ≈ 0 (Δ_W8 never significant). Restoring the interface basis returns acceptance to the identity baseline within noise. Note that micro-AL measures draft–target agreement under a given verifier, so this panel establishes attribution of the gap — which side of the system moved — not an end-task quality ranking of the arms.
