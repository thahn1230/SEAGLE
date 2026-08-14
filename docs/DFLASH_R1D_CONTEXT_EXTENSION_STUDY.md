# DFlash R1_D Context Extension Study (R1DCE)

**Question:** before keeping a separate context rotation R_C, is it enough to
extend the draft's existing global rotation R1_D to the target-derived
context hidden: `H_t -> H_t @ R1_D` before the ctx A4 quantizer, with
mathematically paired ctx K/V weight views?

**Answer: NO — but it comes surprisingly close, and the reason it falls
short settles the mechanism question.**  Primary outcome (§28, central
question): **CASE B** — R1_D alone recovers most of the context-rotation
effect (+1.14 of the +1.20 validation gap) but the current R_C remains
Holm-significantly better, because R1_D is a *worse mixer* on the H_t
outlier directions.  Among training-free rotations a **CASE E overlay**
holds (R1_T/Hadamard/random all tie).  The preregistered optional second
stage then delivered **CASE D on top**: one jointly-trained shared
rotation R1_DC (used for BOTH the draft residual and H_t) significantly
**beats** the separate R1_D + R_C deployment on validation AND on all 4
test datasets (4-ds mean 3.5207 vs 3.3825, p≤3.3e-4 per dataset) —
separate matrices can be eliminated by joint context-aware optimization,
with an AL gain, not just parity.

Run dir: `runs/dflash_r1d_context_extension_20260814_123938`
(pointer `runs/R1DCE_RUN_DIR`).  Preregistration:
`configs/preregistration.md` (frozen before any new run).
Source of truth: basis-contract audit verdict D (@70cc502) — all
identities re-verified here (GATES 1–9, 9/9 PASS, `tables/gates.json`).

## 1. Phase-0 — what the matrices actually are (§4/§33)

| fact | value |
|---|---|
| R1_T | `outputs/rotations/llama31_w4a4kv16_s1/R.bin` key R1, file sha f433a88b, tensor sha 90b1571a, orth 2.6e-5, det +1 |
| R1_D | `VSQ_RD/rotations/draft/R1D_s1r1.pt.best` key R1_D, file sha 11445947, tensor sha 1b78dd75, orth 7.6e-5, det +1 |
| current R_C | **`--vsq-rc rt` ⇒ R_C == R1_T exactly** (same tensor sha; every M5/M6 job) |
| current R_C == R1_D? | **No.** ‖R1_T−R1_D‖_F/√d = **1.4143 ≈ √2**, tr(R1_Tᵀ R1_D)/d = **−0.0001** |

The two learned rotations are as different as two independent random
rotations — "R1 exists on both sides" does NOT mean they are similar.
All candidate matrices, hashes, and pairwise similarities:
`tables/rotation_matrix_similarity.csv`, `tables/phase0_matrix_audit.json`;
frozen arm ckpts + manifest in `rotations/`.

## 2. Correctness gates (§31) — 9/9 PASS (`tables/gates.json`)

- GATE-2: `(H_t R1_D)((W γ_hid [R2]) R1_D)ᵀ` reproduces stock FP ctx K/V to
  ≤1.4e-6 (fp32-stored views; audit contract <1e-5) on real H_t.
- GATE-5: deployed order proven at source + tensor level — rotation
  `Ht = Ht @ rc_matrix_buf` (vsq_draft_rot.py:252) precedes the ctx A4
  sites `act_fake_ste(Ht, cab)` (:277-278); deployed xc integer codes ==
  recomputed A4(H_t@R1_D) codes (match 1.000000).
- GATE-6 (§12): sequential `(Ht@R1_D)@R1_T` == combined `Ht@R_combo`
  (FP max_rel 6.8e-7; A4 code match 0.999998).  **C7 "R1_D then R_C" is
  ONE combined orthogonal rotation, not a two-stage method**
  (`tables/sequential_vs_combined_parity.csv`).

## 3. Mechanism — does R1_D flatten H_t? (§6/§7/§8)

Same 8000 reservoir-paired H_t tokens for every candidate
(`captures/ht_sample.npz`; FP-path H_t, FIDI convention).
`tables/ht_candidate_stats.csv`, `ht_a4_candidate_stats.csv`,
`ctx_kv_aw_decomposition.csv`, `ctx_kv_layerwise_nmse.csv`,
`ctx_cache_error.csv`.

| candidate | kurtosis | absmax | top-0.1% energy | A4 NMSE (mean) | SQNR | ctx-K AW NMSE | ctx-V AW NMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 240.1 | 39.6 | 37.8% | 0.4824 | 3.2 dB | 0.2736 | 0.5969 |
| **R1_D** | **4.26** | **12.0** | **2.6%** | **0.0689** | **11.6 dB** | **0.0239** | **0.0688** |
| R1_T (= current R_C) | 2.89 | 5.1 | 0.31% | 0.0179 | 17.5 dB | 0.0108 | 0.0309 |
| Hadamard | 2.87 | 5.4 | 0.32% | 0.0177 | 17.5 dB | 0.0107 | 0.0309 |
| random (4 seeds) | ~3.00 | ~5.5 | ~0.5% | 0.0193–0.0199 | ~17.1 dB | ~0.0112 | ~0.0323 |
| learned RC_L0 (DFST) | 4.10 | 15.2 | 2.2% | 0.0541 | 12.7 dB | 0.0201 | 0.0578 |
| R1_D@R1_T (C7b) | 3.01 | 5.6 | 0.5% | 0.0197 | 17.1 dB | 0.0112 | 0.0325 |

So: R1_D removes the catastrophic pathology (kurt 240→4.3) but **leaves
two saturated channel ridges** (see `figs/FIG_Ht_noRot_vs_R1D_sameZ.*`,
`FIG_Ht_R1D_vs_currentRC_sameZ.*`) and lands **3.9× worse A4 NMSE** than
any generic well-mixing rotation.  A trained draft-residual rotation is a
*worse* H_t mixer than a random one — it has structure aligned with the
H_t outlier directions that partial-mixes them.  K/V error is
activation-dominated in every arm (A-only ≈ AW), so the H_t A4 site is
the binding constraint, layer-uniform, K and V alike (no CASE-E-of-§11
V-specific failure).

## 4. Validation AL (§9) — gsm8kvalid n60, T512, cycle-pooled tau

`tables/validation_context_rotation_al.csv`, `tables/bootstrap.csv`,
`tables/holm.csv`.  Reproduction anchors are **bit-exact** (C0 == prior
V_M3 == 2.0166; C2 == prior V_M5_rconly == 3.2132; Δ = 0.0000).

| Context policy | tau | Δ vs C0 | H_t A4 NMSE | K NMSE | V NMSE |
|---|---:|---:|---:|---:|---:|
| none (C0) | 2.0166 | — | 0.482 | 0.274 | 0.597 |
| **R1_D (C1, primary)** | **3.1582** | **+1.142** | 0.069 | 0.024 | 0.069 |
| R1_T == current R_C (C2/C3) | 3.2132 | +1.197 | 0.018 | 0.011 | 0.031 |
| Hadamard (C4) | 3.1950 | +1.178 | 0.018 | 0.011 | 0.031 |
| random ×4 (C5) | 3.178–3.203 | +1.16–1.19 | 0.019–0.020 | 0.011 | 0.032 |
| learned DFST RC_L0 (C6) | 3.2251 | +1.209 | 0.054 | 0.020 | 0.058 |
| R1_D@R1_T combo (C7b) | 3.2014 | +1.185 | 0.020 | 0.011 | 0.032 |
| R1_D@ΔR residual (C8) | 3.2025 | +1.186 | (§5) | | |
| shared R1_DC (C9) | (see §6) | | | | |

Preregistered primaries (paired bootstrap 3000, Holm):

| pair | Δtau | 95% CI | p | Holm |
|---|---:|---|---:|---|
| P1 C0 vs C1 | +1.1416 | [+1.062, +1.227] | 3.3e-4 | **SIG** |
| P2 C1 vs C2 | +0.0550 | [+0.017, +0.094] | 0.0047 | **SIG** |
| P3 C1 vs Hadamard | +0.0368 | [−0.019, +0.096] | 0.195 | n.s. |
| P4 C1 vs random s101 | +0.0347 | [−0.012, +0.082] | 0.146 | n.s. |
| P5 C1 vs C7b combo | +0.0432 | [−0.001, +0.091] | 0.062 | n.s. |

**Decision Gate A: FAIL** on both preregistered criteria — (a) C2's
superiority over C1 is Holm-significant; (b) H_t A4 NMSE(C1) = 3.9×
NMSE(C2) ≫ 1.10×.  A separate, well-mixing context rotation REMAINS
necessary; extending R1_D alone is not sufficient.

## 5. C8 — residual ΔR (§13–15): small correction is NOT enough

`R_context = R1_D @ ΔR`, ΔR Cayley-orthogonal identity-init, Adam 2e-3
(repo rotation-training infra — deliberately NOT official SpinQuant
SGDG/Stiefel; documented in the trainer), objective = deployed W4A4 ctx
K/V reconstruction NMSE on the calibration rows, best-val checkpoint
(`tables/residual_rotation_stats.csv`).

- ctx-KV val NMSE: 0.0461 (pure R1_D) → **0.0196** — reaches the generic
  level within 25 steps.
- **‖ΔR−I‖_F/√d = 0.82**, mean rotation angle 0.71 rad (p99 2.86 rad):
  the required correction is a LARGE rotation, not a small residual.
- Validation AL: **3.2025** — statistically ties current R_C.

Conclusion of §15: R1_D is **not** "already near the right context
basis" — the context geometry is structurally different from the draft
residual geometry.  Any large well-mixing completion works, which is the
CASE-E signature again.

## 6. C9 — one shared R1_DC (§16–17)

Shared matrix construction `R1_DC = R1_D_frozen @ D` (identity-init
Cayley; exact warm start), used simultaneously for the draft residual
(`rq.R1`) and the context path (`rc_matrix_buf`, live graph tensor);
dual objective L = λ_draft·L_DFlash-native + λ_ctx·L_ctxKV-W4A4 on the
same frozen corpus; λ_ctx ∈ {0.3, 1.0} (validation-only sweep).
`tables/shared_r1dc_quality.csv`.

λ=1.0: draft val CE 2.800 → **2.554 (improves)**, ctx NMSE 0.0468 →
0.0239, ‖R1_DC − R1_D‖_F/√d = 1.41 — the shared matrix rotates
essentially all the way to a new orthogonal basis while the draft loss is
indifferent (λ=0.3: CE 2.571, ctx 0.0250, dist 1.41 — same shape).

**Validation AL: λ=1.0 → 3.3372, λ=0.3 → 3.3164 — BOTH significantly
beat the current R_C** (vs C2 3.2132: Δ=+0.124, p=3.3e-4 and Δ=+0.103,
p=6.7e-4).  §17 gates: (1) FP basis equivalence — orthogonality 1.2e-4,
det +1, class-A identities are orthogonality-generic ✓; (2) draft
held-out CE improves ✓; (3) target untouched ✓; (4) H_t A4 NMSE 0.0232
(vs 0.069 for R1_D) ✓; (5) ctx K/V NMSE halves ✓; (6) validation AL
improves significantly ✓; (7) no test data in selection ✓.

The C9-vs-C8 decomposition attributes the gain: C8 (ctx-only training on
the same objective) merely TIES C2 (3.2025, p=0.68) — the extra +0.13 of
C9 comes from the *joint* draft-side co-adaptation (one matrix serving
both interfaces finds a better basis for both; the draft objective alone
was rotation-indifferent, VSQ's "R1D < random" echo).  Honest naming
(§32): C9 is a **jointly context-aware re-optimized shared rotation** —
NOT "R1_D reuse"; it lands ‖R1_DC−R1_D‖_F/√d = 1.41 away from its init.

## 7. P2 and QAT on the R1_D basis (§20–21, Q13/Q14)

`tables/p2_interaction.csv`.  P2 gain persists and is basis-agnostic:

| chain | rt basis (prior, bit-repro) | R1_D basis (fresh) |
|---|---:|---:|
| rotation only | 3.2132 | 3.1582 |
| + P2 | 3.4070 (+0.194) | 3.3515 (+0.193) |
| + P2 + QAT (Q5 recipe, lr 3e-2, 400 steps) | 3.6690 (+0.262) | 3.5783 (+0.227) |
| P2 alone (no ctx rotation) | 1.9023 (−0.114, destructive) | — |

P2 remains conditional-on-rotation (destructive alone), and QAT remains
complementary — both consistent with the VSQ-study attribution.

## 8. Final 4-dataset results (§22)

`tables/final_4dataset_al.csv` (micro-tau per dataset; mean4 =
dataset-level mean; F0/F1/F2/F4/F5/F6 reused bit-repro from the VSQ run
with provenance manifest; F3*/F5b/F6b fresh).

| Method | MT | GSM8K | HumanEval | ShareGPT | mean4 | full-blk | P(L≥6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 FP16 | 3.8619 | 4.2129 | 4.8920 | 3.8391 | **4.2015** | .098 | .247 |
| F1 naive W4A4 RTN | 1.8740 | 1.6681 | 1.6435 | 1.6105 | 1.6990 | | |
| F2 vanilla SpinQuant (no ctx rot) | 1.6749 | 2.0050 | 2.0990 | 1.5643 | 1.8358 | | |
| F3 + H_t@R1_D | 3.1703 | 3.2898 | 3.9186 | 2.9902 | 3.3422 | .052 | .154 |
| F3b + H_t@(R1_D·ΔR) (C8) | 3.2309 | 3.3208 | 4.0109 | 3.0204 | 3.3957 | .055 | .160 |
| **F3c shared R1_DC (C9)** | 3.3126 | 3.4570 | 4.1998 | 3.1134 | **3.5207** | .062 | .177 |
| F4 + current R_C (R1_T) | 3.2030 | 3.3123 | 3.9836 | 3.0313 | 3.3825 | | |
| F5b R1_D + P2 | 3.3851 | 3.5145 | 4.4590 | 3.1569 | 3.6289 | .068 | .193 |
| F5 R1_T + P2 | 3.4395 | 3.5510 | 4.5042 | 3.2089 | 3.6759 | | |
| F6b R1_D + P2 + QAT | 3.4936 | 3.7198 | 4.5843 | 3.2859 | 3.7709 | .073 | .208 |
| F6 R1_T + P2 + QAT (deployed) | 3.5420 | 3.7877 | 4.6500 | 3.3121 | **3.8230** | | |

Test-level paired bootstrap (3000, per dataset; `stats/bootstrap_final_*`):

- **C9 vs current R_C: significant on 4/4 datasets** (Δ +0.082…+0.216,
  p ≤ 3.3e-4 each).  At pure-structural-PTQ level C9 recovers 71.2% of
  the FP16 gap vs 65.4% for the current R_C.
- R1_D vs current R_C: rt better on 3/4 (mt p=.002, he p=6.7e-4,
  sg p=3.3e-4; gsm n.s. p=.085) — the −0.04 deficit is real and follows
  the chain (F5b vs F5, F6b vs F6: rt-chain better, 3/4 sig).
- C8 vs current R_C: tie (n.s. on 3/4; mtbench +0.028 in C8's favor).

## 9. RCAL (§23)

`tables/rcal_final.csv` (mtbench matched-reference replay):

| Method | AL_q | RCAL | SAL | LAL | AFS |
|---|---:|---:|---:|---:|---:|
| F3 R1_D | 2.1703 | 1.9518 | 0.2185 | 0.1358 | 0.9168 |
| F3b C8 | 2.2309 | 2.0131 | 0.2177 | 0.1426 | 0.9179 |
| F3c C9 | 2.3126 | 2.0815 | 0.2312 | 0.1565 | 0.9148 |
| F5b R1_D+P2 | 2.3851 | 2.1408 | 0.2443 | 0.1494 | 0.9158 |
| F5 rt+P2 (prior) | 2.4395 | 2.1910 | 0.2486 | 0.1570 | 0.9153 |
| F6b R1_D+P2+QAT | 2.4936 | 2.2313 | 0.2622 | 0.1662 | 0.9124 |
| F6 rt+P2+QAT (prior) | 2.5420 | 2.2655 | 0.2765 | 0.1732 | 0.9097 |

AFS 0.91–0.92 across every new arm, matching the deployed M5/M6 band —
**no arm's AL recovery is deceptive acceptance** (C9's +0.14 gain keeps
AFS 0.9148, within 0.001 of the current R_C chain).

## 10. Runtime & memory (§25)

`tables/runtime_context_rotation.csv`, `tables/memory_cost.csv` (RTX 4090):

- Dense `Ht@R` (identical cost for R1_D / R1_T / any dense R_C):
  341.5 µs/fwd @ S=512 fp32 (0.67 µs/token), 121.0 µs bf16;
  16.78 MMAC/token.  Matches the basis-audit measurement (334.7 µs).
- Structured fast-Hadamard: 14.4 µs @ S=512 (0.03 µs/token) — 8–24×
  cheaper, and Hadamard TIES all other rotations on validation AL.
- Storage: reusing R1_D at H_t saves the 64 MiB fp32 (32 MiB bf16)
  separate R_C matrix — `R1_frozen` is already resident for the draft
  path and `rc_matrix_buf` can alias the same buffer.  **But
  `H_t @ R1_D` is still a runtime rotation** (CLASS A), NOT zero-cost.
- ctx K/V weight views (~84 MiB bf16) are mandatory in M3 already
  (γ-contract), rotation folds into the same tensors offline.

## 11. Weight-view unification audit (§19)

`tables/weight_view_unification.json`.  Under C1 both branches read the
same R1_D basis, but the γ contracts differ:
`W_K,ctx = (W_K D_γhid) R1_D` vs `W_K,draft = (W_K D_γin,i) R1_D`.
cos(γ_hid, γ_in,i) ≈ 0.987–0.992 (close but NOT equal; elementwise ratio
spread is large), so the views are distinct tensors **mathematically** —
physical sharing would require γ_hid == γ_in,i.  Sharing impossible; the
~84 MiB ctx views stay (already required in M3).

## 12. Interpretation (§28) — primary outcome

**CASE B** (R1_D helps but is not enough), with the mechanism resolved as
**generic orthogonal mixing** (CASE E flavor): R1_T, Hadamard, random,
R1_D@R1_T, R1_D@ΔR, and the DFST-learned RC_L0-follow-ups all tie at the
top; the ONLY structural requirement is a *well-mixing* orthogonal
rotation at the H_t quantization boundary.  R1_D specifically is a
sub-par mixer (trained for draft-residual geometry, it leaves ~2.6% of
H_t energy in the top 0.1% of channels vs 0.3–0.6% for generic
rotations), which costs a small but Holm-significant −0.055 validation
tau.

The C9 result adds an important refinement to CASE E: matrix semantics
are weak only within the *training-free* tier.  A **jointly trained**
shared R1_DC significantly beats every training-free rotation (4/4
datasets) — mixing quality is the first-order requirement, and joint
context-aware optimization contributes a real second-order gain (+0.14
4-ds) on top.  So the final classification is: **CASE B for the
literal R1_D-extension hypothesis; CASE D achieved by the §16 optional
stage** (one shared matrix replaces R1_D + R_C and improves AL), with
CASE E limited to the training-free tier.

Terminology contract (§32): the correct description of C1 is
"**draft-R1 context extension** / reuse of the draft global rotation at
the context boundary" — NOT a newly learned context rotation.  C7 is one
combined orthogonal transform.  Matrix-semantics claims are NOT
supported (random/Hadamard tie).

## 13. Answers to §34 (1–20)

1. **Does H_t @ R1_D remove the H_t channel outliers?**  Partially.
   Kurtosis 240→4.3, absmax 39.6→12.0, but two saturated channel ridges
   survive (figs FIG_Ht_noRot_vs_R1D_sameZ) — R1_T/Hadamard/random reach
   kurtosis ~2.9–3.0 with no ridges.
2. **Kurtosis / top-0.1% energy / A4 NMSE / code entropy:**
   240.1→4.26 / 37.8%→2.6% / 0.482→0.069 / 0.31→2.27 bits-per-token.
   Current R_C: 2.89 / 0.31% / 0.0179 / 3.17 bits.  R1_D lands 3.9×
   worse NMSE and ~0.9 bit lower code entropy than any generic rotation.
3. **ctx K NMSE:** 0.274→0.0239 under R1_D (current R_C: 0.0108 — R1_D
   is 2.2× worse).  Layer-uniform (tables/ctx_kv_layerwise_nmse.csv).
4. **ctx V NMSE:** 0.597→0.0688 (current R_C 0.0309; same 2.2× ratio —
   no V-specific failure; §11 outcome = incomplete flattening (A), not
   B/C/D/E-of-§11).
5. **AL recovered by R1_D alone:** validation +1.1416 of the +1.1966
   ceiling (95.4% of the R_C effect), p=3.3e-4; 4-ds test 3.3422 vs
   naive-SQ 1.8358 and current-R_C 3.3825 (63.7% vs 65.4% of FP16 gap).
6. **R1_D tied to current R_C?**  NO — R_C better: validation Δ+0.0550
   [+0.017,+0.094], p=0.0047 (Holm-SIG); test 3/4 datasets SIG.
7. **R1_D tied to R1_T?**  Same as (6): current R_C == R1_T exactly.
8. **R1_D tied to Hadamard/random?**  Statistically yes (p=0.19/0.15;
   all n.s. incl. 3 extra seeds), but R1_D is nominally LOWEST of all
   rotation arms — a trained draft rotation is no better (slightly
   worse) than a random one at the context boundary.
9. **Separate context rotation still necessary?**  A separate *well-
   mixing rotation at the H_t boundary* is necessary (Gate A failed).
   A separate *matrix* is not: the C9 shared R1_DC serves both draft
   residual and H_t from ONE matrix and beats the separate-pair
   deployment (4/4 SIG).
10. **If insufficient, is the required residual ΔR small?**  NO.
    ‖ΔR−I‖_F/√d = 0.82 (mean eigen-angle 40°, p99 164°) —
    context geometry is structurally different from draft-residual
    geometry; R1_D is NOT near the right context basis.
11. **Can one jointly optimized R1_DC serve both?**  YES — the
    strongest result of the study: validation 3.3372 (λ=1.0) and 3.3164
    (λ=0.3) vs 3.2132; test 3.5207 vs 3.3825 (4/4 SIG); draft CE
    *improves* (2.80→2.55); all §17 gates pass.
12. **Can ctx/draft K/V weight views be physically unified?**  NO —
    same basis under C1/C9, but γ contracts differ
    (W γ_hid R vs W γ_in,i R; cos 0.987–0.992 ≠ 1), so the tensors are
    distinct; the ~84 MiB bf16 ctx views remain (required since M3).
13. **Does P2 still help after R1_D?**  YES: +0.193 on R1_D basis
    (3.1582→3.3515, p=3.3e-4) vs +0.194 on rt basis — basis-agnostic,
    and still destructive without any ctx rotation (1.9023).
14. **Does QAT still help after R1_D?**  YES: +0.227 (3.3515→3.5783,
    p=3.3e-4) with the exact matched Q5 recipe; the rt chain stays
    ahead at every stage (F6 3.8230 vs F6b 3.7709, 3/4 SIG).
15. **Final 4-dataset AL:** table §8.  Headlines: deployed F6 3.8230;
    R1_D-chain F6b 3.7709; structural-PTQ-only C9 3.5207 > current R_C
    3.3825.
16. **RCAL/AFS:** table §9.  AFS 0.9097–0.9179 across all arms — AL
    differences are genuine acceptance, not deceptive.
17. **Runtime overhead of H_t@R1_D:** identical to any dense R_C
    (CLASS A): 341.5 µs/fwd @ S=512 fp32 (0.67 µs/tok), 121 µs bf16,
    16.78 MMAC/tok; S∈{1,8,32,128,512} in
    tables/runtime_context_rotation.csv.  NOT zero-runtime.  (The
    Hadamard candidate would be 8–24× cheaper via fast-Hadamard and
    ties the training-free tier.)
18. **Storage saved vs separate R_C:** 64 MiB fp32 (32 MiB bf16) matrix
    + no extra checkpoint; R1_frozen can be aliased.  Same saving
    applies to C9 (one matrix instead of two).
19. **Final scientific story:** a *well-mixing orthogonal rotation at
    the H_t→ctx-K/V quantization boundary* is the mandatory mechanism
    (placement, not matrix identity); training-free mixers tie and
    R1_T-reuse stays the zero-training default; a single jointly
    context-aware trained rotation (shared R1_DC) both eliminates the
    second matrix and adds +0.14 4-ds AL.  "Draft-R1 context extension"
    alone is a worse mixer and is rejected.
20. **Does this simplify or weaken the SEAGLE novelty claim?**  It
    SHARPENS it.  The placement novelty survives every attack here
    (Gate A failed for the simplest alternative).  The R_C matrix-
    semantics claim stays retired (random/Hadamard tie, as in FIDI),
    and the new C9 result upgrades the story: the context interface is
    best served by joint draft+context rotation optimization — a
    simplification (one matrix) AND an improvement, not a concession.

## 14. Actual cost / §33 items 17–19

8×RTX 4090, 12:56–15:16 UTC 2026-08-14 wall ≈ **~19 GPU-h** fresh
compute (validation arms ~10 min each; 4-ds fresh arms 25–60 min each;
C8 6 min; C9 2×22 min; QAT 8 min; RCAL 5×7 min; mech batteries ~15 min)
+ 24 reused prior shards (bit-repro proven: two Δ=0.0000 anchors).
Storage: run dir 3.5 GB (3.3 GB rotation/QAT ckpts, 58 MB captures,
92 MB figs, 36 MB cycles).  Scheduler: GPUs 0–6 + GPU 7 hand-lane
(artifact-skip pattern; one dispatcher at all times).
