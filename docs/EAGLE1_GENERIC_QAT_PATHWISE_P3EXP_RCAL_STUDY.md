# Generic-QAT vs Training-Free Structural P3, Pathwise Exponent Migration, and Reference-Consistent Acceptance Length (RCAL)

**Run**: `runs/eagle1_generic_qat_pathwise_p3exp_rcal_20260731_154652`
**Branch**: `exp/eagle1-generic-qat-pathwise-p3exp-rcal` (from `3f77b6c`)
**Anchor**: `checkpoints/eagle1_fresh_fp16_anchor/anchor.pt`
(SHA256 `a7c6ccc85ac1e1db7aa8d03e8f06294ed9f1b0fa…`, verified)
**Status**: RESULTS PENDING — numbers marked `TBD(phase)` fill in as the
phase-2/3/4 chains complete.

> **RCAL measures fidelity to an FP16 reference target. It is NOT a
> measure of absolute semantic correctness.** A method can be perfectly
> reference-consistent and still produce wrong answers wherever the FP16
> system would; conversely a quantized verifier that "fixes" a reference
> mistake is scored as drift. This caveat applies to every RCAL/SAL/LAL/
> AFS number in this document.

## 1. Questions

1. **GOAL A** — Can a *generic* W4A4 draft QAT (no P3 migration, no
   structural prior, quantizers only) recover the acceptance-length
   collapse that training-free structural D4P3 fixes? Previous QAT arms
   were NOT generic: they trained on top of the P3-migrated fold, so
   they could not answer this.
2. **GOAL B** — Does a normalized exponent parameterization
   `m(D, beta) = D**beta` (D = 4096) with *independent* first/recurrent
   migration strengths beat the single global legacy alpha?
3. **GOAL C** — Are AL gains *reference-consistent*? RCAL replays every
   deployed proposal tree under the FP16 reference target in lockstep
   and decomposes deployed acceptance into reference-consistent and
   drift components.

## 2. Method definitions (final names)

| Name | Definition |
|---|---|
| F16 | fp16 target + fp16 stock draft (upper anchor) |
| NPTQ | naive W4A4 draft PTQ, no migration (alpha = 1) |
| GQAT | **generic** W4A4 draft QAT: `migration_mode="none"`, factor 1.0, no embedding multiply, no W_e division, no P2, no rotation training; quantizer contract/anchor/teacher/budget identical to prior QAT arms |
| LP3 | training-free structural D4P3 with legacy calibrated alpha (45.254834 fp16-target / 32.0 int4-target) |
| EP3-G | exponent P3, single global beta for both paths |
| EP3-P | exponent P3, independent (beta_first, beta_recurrent) |
| LP3-RD | LP3 + learned independent draft rotation R_D (`LK2_AUXG_s0`) |
| LP3-QAT | prior (non-generic) QAT on the P3 fold, re-evaluated here |

Targets: T0 = fp16 target; T1 = SpinQuant W4A4 target (KV16,
`learned_chat_w4a4kv16`). GQAT is realized as the validated QAT trainer
with alpha pinned to 1.0 — `E*1.0` and `W/1.0` are bitwise identities,
so step 0 ≡ NPTQ (audited: `manifests/gqat_audit_*.json`, all
`migration_mode=none`).

## 3. Pre-registered decisions (recorded before results)

- AFS thresholds {0.90, 0.95, 0.98}.
- EP3-P primary selection **Rule 3**: among the top-3 MT-Bench pairs,
  max AL_q subject to AFS >= 0.95 (fallback max AFS).
- Non-inferiority margin 0.05 in **both** RCAL and AL.
- Beta search is never performed on MT-Bench: coarse/fine NMSE grids ->
  5x5 pairs by projection NMSE -> top-9 on 20-prompt held-out calib AL
  -> top-3 on MT-Bench -> Rule 3.
- LR pilot: {3e-7, 1e-6, 3e-6, 1e-5}, select by 20-prompt calib AL.

## 4. RCAL protocol (§14–§15)

Per deployed speculative cycle (Tq target + method draft, greedy): record
prefix length, the full proposal tree (flattened `tree_tokens`), the
Tq-accepted `(best_q, R_q)`; verify the *same* tree from the *same*
prefix with the FP16 reference T0 held in lockstep on a second GPU, its
KV always compressed along the **deployed** trajectory. `R_RC` = longest
common prefix of the accepted proposal sequences `S_q`, `S_0`.

- Aggregates (ratio-of-aggregates over cycles, FP64 fsum):
  `RCAL = Σ R_RC / N`, `AL_q = Σ R_q / N (+1 root = official tau)`,
  `AL_0 = Σ R_0 / N`, `SAL = AL_q − RCAL` (spurious), `LAL = AL_0 −
  RCAL` (lost), `P_A = ΣR_RC/ΣR_q`, `R_A = ΣR_RC/ΣR_0`,
  `AFS = 2 P_A R_A / (P_A + R_A)`.
- Branch divergence can give `R_RC < min(R_q, R_0)` (different accepted
  branches sharing a shorter prefix) — measured, not assumed away.
- **Acceptance-length contract** (audited in
  `docs/EAGLE_ACCEPTANCE_LENGTH_CONTRACT.md`): official EAGLE tau = 1 +
  accepted-draft-length; RCAL metrics live on proposal tokens only.
- **Identity control (passed)**: T0 as both deployed and reference gives
  RCAL = AL_q = AL_0 = 2.5608, SAL = LAL = 0, AFS = 1 on 148 cycles;
  decomposition identities hold to 1e-12 (FP64) in tests.
- **Offline replay equivalence**: `replay_eagle_proposals_reference_
  target.py` reconstructs the reference trajectory from the records
  alone and must reproduce the inline lockstep `S_0` exactly:
  TBD(phase3).

## 5. First vs recurrent projection-input distributions (§7)

From raw unscaled captures (`p3exp_capture_*.pt`, spy on the deployed
split forward):

- fp16 target: e/h RMS ratio — first path **0.0070**, recurrent path
  **0.0091** (~30% difference; both ≈ 2 orders of magnitude imbalance,
  the structural D4 failure).
- int4 target: TBD(phase3 pathdist).
- Best exponent pairs differ by path and *reverse direction across
  targets*: fp16 (beta_first 0.46, beta_rec 0.44 — m 45.9 vs 38.9;
  legacy 45.25 ≈ the recovered beta 0.46 mapping, an independent
  validation of the legacy calibration); int4 (beta_first 0.40, beta_rec
  0.44 — m 27.9 vs 38.9). The int4 first path passes through the
  gamma_R1 interface basis, which compresses the e-slice differently.

## 6. GQAT training (§5–§6)

- LR pilot (600-step, 20-prompt calib AL): monotone in LR for both
  variants; T0 {3e-7: 1.1948, 1e-6: 1.2450, 3e-6: 1.3408, 1e-5:
  **1.4795**}, T1 {3e-7: 1.4604, 1e-6: 1.5533, 3e-6: 1.6732, 1e-5:
  **1.7131}**. Selected 1e-5 for both. **CAVEAT: grid-top boundary** —
  the optimum may lie above 1e-5; a wider pilot was out of budget. NPTQ
  references: 1.111 (T0) / 1.3395 (T1) — recovery without structure is
  slow, consistent with GOAL A's hypothesis.
- 3 seeds x {T0, T1}, 3000 steps, ckpt every 500, best-val (held-out
  calib AL) primary + final reported: TBD(phase3 cksel).

## 7. Results (MT-Bench, official tau) — TBD(phase3/4)

Method matrix, RCAL decomposition table, bootstrap CIs
(prompt-cluster, 3000 reps, paired dAL_q AND dRCAL), cross-dataset
finalists (single-seed, labeled), overhead: filled by
`FINAL_OUTPUT_DRAFT.txt` after phase 4.

Reference numbers from completed sibling studies (same anchor/protocol):
F16 mtbench tau 3.5341; NPTQ 1.111/1.3395; LP3 2.9955/2.9666; LP3-RD
3.0629 (prior run; re-evaluated here); LP3-QAT finals 3.19–3.33.

## 8. Does generic QAT learn P3-like rebalancing? (§8)

W_e/W_h RMS ratio per checkpoint (`gqat_rebalancing_*.json`):
anchor ratio vs LP3-effective ratio (anchor with W_e/alpha) vs GQAT
trajectory: TBD(phase3). If GQAT's ratio migrates toward the
LP3-effective value, QAT is *learning* the structural rebalancing that
P3 encodes in closed form.

## 9. Tree-vs-sequential verifier fidelity (§19)

Controls B (T0-seq vs Tq-seq token agreement along the deployed
trajectory — pure quantization drift, no tree shape) and C (Tq
tree-accepted vs Tq sequential argmax — execution-shape drift of
dynamic A4 under tree masks): TBD(phase3). A shape-dependent A4 warning
fires if tree != seq agreement < 0.99 under T1.

## 10. Decision rules A–H (§29) — verdicts TBD(phase4)

Recorded here verbatim before results; the report will mark each rule
used/unused with CI support. Deceptive-AL-gain flag = paired CI
`dAL_q > 0` AND `dRCAL <= 0` (bootstrap, 3000 reps).

## 11. Related work and novelty (§28)

See `docs/RCAL_RELATED_WORK_AND_NOVELTY_AUDIT.md`. Closest: QSpec
(arXiv 2410.11305), QuantSpec (2502.10424), ML-SpecQD (2503.13565), HSD
(2601.05724). None perform paired same-proposal replay under a reference
verifier with LCP decomposition; the name RCAL was not found. Claims are
scoped accordingly (no claim that verifier-drift analysis per se is
novel; the *metric construction* and its identity-controlled
decomposition are the contribution).

## 12. Limitations

- RCAL is reference-fidelity, not semantic correctness (see banner).
- GQAT LR selected at the pilot grid top (boundary) — GQAT numbers are
  a lower bound on generic-QAT recovery in LR.
- Cross-dataset finalists are single-seed (labeled).
- Task quality is proxied by target wikitext-2 PPL (5.9851 fp16 /
  6.0326 int4, identical target builds, prior-run provenance) plus
  generated-token agreement with T0; no end-task exact-match harness in
  this run.
- Beta-pair AL/RCAL surfaces are evaluated only on the pre-registered
  top-9/top-3 subsets (masked cells in figures); the NMSE surface is the
  only fully-populated 5x5.
- fake-quant (no real INT4 kernels); speed numbers are acceptance-based.
