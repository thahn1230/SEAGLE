# Draft-aware SpinQuant rotations for a W4A4 EAGLE-1 draft: does R2 need to be learned too?

Confirmatory 2x2 factorial study, 2026-08-10, host `gpusystem` (8x RTX 4090 24GB).
Run directory: `runs/eagle1_draft_aware_r1_r2_gs_20260810_065036/`.

Terminology (used throughout): **GS** = Global Scaling (legacy EP3-G), **LS** =
Local Scaling (legacy EP3-P). **R_{1,T}** = target SpinQuant residual-stream R1;
**R_{1,D}** = draft-aware residual R1; **R_{2,B}** = baseline draft attention R2;
**R_{2,D}** = learned draft-aware attention R2. Legacy names appear only in this
sentence and in the provenance table.

---

## 1. Research question

A previous study showed that the target-optimal SpinQuant residual rotation
R_{1,T} is not optimal for a quantized EAGLE draft, and that a draft-specific
residual rotation `R_{1,D} = R_{1,T} C(A)` (all weights frozen) buys substantial
acceptance length. That study held the draft's attention R2 fixed. This study
asks whether the attention-local basis also needs draft-aware optimization:

| Arm | R1 basis (draft-internal) | R2 basis (draft V/O) |
|-----|---------------------------|----------------------|
| A0 BASE            | R_{1,T} | R_{2,B} |
| A1 R1D_ONLY        | learned R_{1,D} | R_{2,B} |
| A2 R2D_ONLY        | R_{1,T} | learned R_{2,D} |
| A3 R1D_R2D_JOINT   | learned R_{1,D} | learned R_{2,D} |
| A4 COMPOSED (diagnostic) | A1's R_{1,D} | A2's R_{2,D}, no joint training |

All arms share one deployment contract: target = W4A4 SpinQuant (T4), draft =
W4A4, KV = FP16 (R3 OFF), R4 frozen as deployed, batch 1, greedy EAGLE-1,
official `mc_sim_7b_63` tree, **GS** scaling with the repository's canonical T4
`beta_GS = 0.42` (m = 4096^0.42 = 32.89964245299412) held FIXED, and the first
target->draft interface pinned to R_{1,T} in every arm. Target, draft, and
LM-head weights are frozen; only rotation generators receive gradients.

Pre-registration (frozen before any training on this server):
`runs/.../configs/preregistration.md`.

---

## 2. Repository / environment

| Item | Value |
|---|---|
| Repo | `github.com/thahn1230/SEAGLE`, base commit `a6a71da`, branch `exp/eagle1-draft-aware-r1-r2-gs` |
| third_party/EAGLE | `4a9cf3a1` (branch `v1`) |
| third_party/SpinQuant | `8f47aa3f` |
| Target | `meta-llama/Llama-2-7b-chat-hf` @ `f5db02db724555f92da89c216ac04704f23d4590` |
| Draft | `yuhuili/EAGLE-llama2-chat-7B` @ `44e37ec383348306fe9b1dfe7c79e145c96db3d0` |
| GPUs | 8x NVIDIA GeForce RTX 4090 24GB, driver 550.163.01, CUDA 12.4 |
| Env | conda `seagle`: Python 3.10.20, torch 2.6.0+cu124, transformers 4.51.3, `fast_hadamard_transform` 1.1.0 (built from source against conda CUDA 12.4) |

The Llama-2 repository is gated; access was provided by the user's HF token.
No silent Vicuna substitution was made — the primary experiment runs on the
declared Llama-2 pair.

### 2.1 R_{1,T} was rebuilt from scratch

This server had no prior `outputs/` artifacts, and SpinQuant's published
rotations cover base models only, so the chat-model rotation was re-optimized
with the official recipe (`scripts/_run_optimize_rotation_gpusystem.sh`, a
host-corrected copy of `_run_optimize_rotation.sh`): wikitext-2 train, seqlen
2048, 100 steps at global batch 8 (8 GPUs x batch 1 x accum 1), lr 1.5 cosine,
W4A4-KV16 in the loop with `--w_clip --a_asym --k_asym --v_asym`. Wall clock
5h51m on 8x4090. Artifact:
`outputs/rotations/learned_chat_w4a4kv16/R.bin`, sha256
`0a2d19979190ba97d4f1d024151393ff2679ef0542b939932d2d8788b1fdbbaa`, 33 keys
(`R1` 4096x4096 + 32 per-layer `self_attn.R2` 128x128).

PPL sanity anchor (official contract, wikitext-2 test, in-process build,
`runs/.../audit/ppl_sanity.json`): FP16 **6.9436** (prior-server anchor 6.9452),
W4A4 + learned R_{1,T} **7.1062** (prior-server anchor 6.9629; sanity band
6.0-8.5 → PASS). The rebuilt rotation is a valid but not bit-identical
re-derivation of the original — absolute AL numbers are therefore comparable
*within* this study, not against the earlier server's tables.

---

## 3. R2 contract audit (what the baseline R2 actually is)

Full audit: `docs/EAGLE1_DRAFT_AWARE_R1_R2_GS_AUDIT.md`; executable version
`scripts/audit_r2_contract.py` (CPU + model levels, all checks PASS,
`runs/.../audit/r2_contract_audit.json`).

Findings that mattered for the design:

1. **Provenance.** R_{2,B} is a deterministic **seed-0 Haar-random orthogonal**
   matrix (QR of a Gaussian), regenerated at every adapter install — *not*
   loaded from `R.bin`, *not* the target's learned per-layer R2, *not*
   transferred. sha256 of its fp64 bytes: `145e00cfe8417ed6…`. It is therefore
   called "baseline R2", never "target R2".
2. A pre-existing code comment described it as a "random Hadamard-128"; the
   audit disproves that (element magnitudes span 0.0000-0.3283 vs a Hadamard's
   constant 0.0884). The *target's* random-mode R2s are Hadamard — the two must
   not be conflated. The comment is corrected in this branch.
3. **Shape/granularity.** One `[128,128]` fp64 matrix **shared across all 32
   heads** of the draft's single decoder layer (`num_hidden_layers = 1`,
   verified against the draft config). Applied block-diagonally.
4. **Fold equations.** Per head h: `V_h' = R2 · V_h` (v_proj output rows) and
   `O_h' = O_h · R2ᵀ` (o_proj input columns), i.e. `v' = blkdiag(R2)·v`,
   `o' = o·blkdiag(R2)ᵀ`. (An earlier internal note said `O_h · R2`; the
   executable audit caught the transpose and it is corrected here.)
5. **Quantization boundaries.** v/o weights are W4-quantized **after** the R2
   fold (quantizing before the fold differs by 2.81 in max element). The o_proj
   input activation is A4-quantized **in the R2 basis**, quantize-then-GEMM;
   the audit further shows the A4 grid is genuinely basis-dependent
   (`Q(x R2ᵀ) != Q(x) R2ᵀ`, max deviation 0.7361) — which is exactly why a
   draft-aware R2 can matter at all.
6. **Foldability.** R2 is F0 (fully offline-folded): no `[128,128]` buffer
   exists in any installed runtime module; the only online draft operators are
   the R4 down_proj Hadamard and the PostProjectionR1 GEMM.

### 3.1 R_{2,D} parameterization

`ResidualR2Rotation` (`src/eagle_spinquant/residual_rotation.py`):
`R_{2,D} = R_{2,B} C(B)`, `B = W − Wᵀ` skew by construction,
`C(B) = (I − B/2)^{-1}(I + B/2)`, `W = 0` at init so step 0 reproduces R_{2,B}
**bitwise**. Granularity matches the baseline exactly (one shared `[128,128]`);
no new sharing policy was invented. R_{1,D} keeps the canonical residual-Cayley
form `R_{1,T} C(A)` on the dense 4096x4096 residual basis.

---

## 4. Correctness gates (all passed before training)

| Gate | Script / artifact | Result |
|---|---|---|
| R2 contract | `audit_r2_contract.py` → `audit/r2_contract_audit.json` | PASS (10 checks, CPU + model level) |
| **R2-A** FP gauge equivalence | `check_r2_gates.py` → `gradchecks/gate_r2_ab.json` | PASS |
| **R2-B** quantization sensitivity | same | PASS |
| **R2-C** trainer/runtime parity | `check_r1r2_gs_parity.py` → `gradchecks/gate_gs_r1r2_parity.json` | PASS |
| **GS-R1** parity + interface pinning | same | PASS |
| **R2-D** orthogonality | asserted every training step, logged every validation | PASS |
| Parameter audit + weight freeze | `gradchecks/param_audit__*.json` in every run | PASS |

**Gate R2-A (FP gauge, quantization OFF).** The learnable object at init
reproduces the baseline forward bitwise (max Δlogits = 0.0). For a perturbed
R_{2,D}, the *correct* V/O pair refold leaves the FP function at the
fp16-roundoff floor — and the gate is discriminative rather than
threshold-fitted: under a 5x larger generator the correct fold's deviation does
*not* grow (Δlogits 0.0156 → 0.0117, x0.75, greedy tokens preserved at both
scales) while a deliberately **broken** fold (v refolded, o left at R_{2,B} — a
genuine function change) grows 0.3721 → 4.9902 (x13.4). Correct/broken ratio
0.042 and 0.002. The fp64 zero-deviation gauge identity is covered separately by
the contract audit (relative deviation 1.3e-15, and it holds for an arbitrary
orthogonal R2', so this is gauge freedom, not a property of the seed-0 draw).

**Gate R2-B (W4A4 ON).** A perturbed R_{2,D} changes the quantized v/o weights
and *only* v/o among the nine quantized sites; quantized draft logits move
(max Δ 4.04); the LK hybrid loss moves (6.8290 → 7.1713); and `dL/dB` is finite
and nonzero on multiple independent batches (‖grad‖ = 1.38, 3.01, 1.43, 2.26).
In joint mode `dL/dA` also flows (65.6, 102.6, 79.2, 100.8). The loss is
therefore *not* mathematically independent of R2 — the transform is on the
correct side of the quantizer.

**Gate R2-C / GS-R1 (trainer ↔ runtime).** Four legs under the GS fold
(m = 32.899642), each comparing the differentiable trainer against the deployed
`ConcatSelectiveDraftAdapter`: baseline, perturbed R_{1,D} only, perturbed
R_{2,D} only, and joint. All four are **bitwise identical** — 10 fp16 quantized
tensors with max|Δ| = 0.0, and a K=4 teacher-forced chain with max Δlogits = 0.0
and identical greedy tokens at every depth. Three structural cross-leg checks
also pass: the first target->draft interface fold (`W_first`) is **hash-identical
across all four legs** (pinned to R_{1,T}; neither rotation may move it), a
perturbed R_{2,D} changes only the v/o folds, and a perturbed R_{1,D} moves the
recurrent fold.

**Parameter audit / weight freeze.** Every training run writes the full
parameter table and asserts that the only trainable tensors are the rotation
generators (`rot.W` for A1/A3, `r2_rot.W` for A2/A3); SHA-256 of every model
weight buffer is recorded before and after training and asserted equal. In the
first completed run the trainable set was exactly `['rot.W']` for A1 and the
weight hashes matched before/after.

---

## 5. Training

**Objective.** The repository's validated LK hybrid loss, reused unchanged: at
recurrent depth k, `alpha_k = sum_v min(p_k(v), q_k(v))`, `TV_k = 1 − alpha_k`,
`lambda_k = exp(−3 · stopgrad(mean alpha_k))`, and
`L = sum_k w_k [lambda_k KL(p_k‖q_k) + (1−lambda_k) TV_k] / sum_k w_k` with
K = 4 teacher-forced depths and `w_k = 0.8^(k−1)`. Teacher = the deployed
**W4A4 (T4)** target, matching deployment; FP16-target logits are not used as
the primary teacher.

*Scientific wording.* This is an **acceptance-aware surrogate** under our greedy
EAGLE evaluation — full-vocabulary overlap is not mathematically identical to
greedy AL. The loss is not novel; it is an LK-Loss-inspired hybrid objective
reused here for quantization-basis optimization. The novelty under test is
acceptance-aware optimization of the **draft quantization basis** (R_{1,D} and
R_{2,D}) with all model weights frozen.

**Corpus.** Regenerated with the canonical builder
(`build_target_generated_lk_corpus.py`), T4 teacher, greedy, 5 domains x 1000 =
**5000 windows** (wiki / c4 / sharegpt / gsm8k / code-mbpp), T_WIN 48,
K_STORE 6, GEN_LEN 64, full-vocabulary fp16 teacher logits. Train-pool
disjointness from every evaluation pool is the builder's Gate-F guarantee
(sharegpt/gsm8k offsets ≥ 2000, wiki/c4 TRAIN vs eval test/validation, mbpp vs
HumanEval). Manifest with per-shard SHA-256:
`runs/.../manifests/lkcorpus__rd_gs_all.json`.

*Documented deviation:* the canonical merge concatenates domains in order, which
would make the trainer's last-10% validation split code-only. Because this study
selects the **true best-validation checkpoint**, the merge was replaced by a
deterministic seed-0 stratified shuffle before re-sharding
(`scripts/_r1r2gs_merge_corpus.py`), giving a domain-balanced held-out tail
(gsm8k 107 / c4 101 / wiki 101 / code 99 / sharegpt 92).

**Schedule.** 3000 steps, batch 32, accum 1, AdamW(0.9, 0.95, wd 0), 100-step
warmup + cosine, grad clip 0.5, K = 4, `--eval-every 100`, GS alpha fixed at
32.89964245299412, draft KV16. Seeds 1001 / 1002 / 1003 per trained arm.

**LR selection.** A1 used the previously validated anchor 3e-4. For A2 and A3 a
held-out LR pilot ran {1e-4, 3e-4, 1e-3} at a short fixed budget (600 steps),
selected on **held-out validation loss only** — never on MT-Bench, GSM8K,
ShareGPT, or HumanEval — before any confirmatory evaluation
(`runs/.../tables/lr_selection.json`):

| Arm | 1e-4 | 3e-4 | 1e-3 | chosen |
|---|---|---|---|---|
| A2 (R2_D only) | 5.8672 | 5.7397 | **5.3880** | 1e-3 |
| A3 (joint)     | 2.9917 | **2.7950** | 2.8120 | 3e-4 |

A null result for R2_D therefore cannot be attributed to an untuned learning
rate.

**Checkpoint selection.** The trainer was extended with `--save-best-val`: the
evaluated artifact is the **true best-validation checkpoint**, not the final
step (the final-step rotation is still saved alongside as `<out>.final.pt` for
provenance). This closes a documented weakness of the earlier studies, and it
matters here: in the first completed run the best validation loss occurred at
step 799 (5.3372) while the final step 2999 was 5.7386 — the old policy would
have evaluated a materially worse rotation.

---

## 6. Main 2x2 results

Official micro-tau (tau = accepted draft tokens + 1, AL = sum(tau)/cycles,
pooled over cycles per dataset). Each trained arm is represented by its
pre-registered MEDIAN-VALIDATION seed: A1 = s1001, A2 = s1003, A3 = s1002.

| Method | R1 basis | R2 basis | MT | GSM8K | ShareGPT | HumanEval | Mean4 |
|--------|----------|----------|------|-------|----------|-----------|-------|
| A0 GS Base        | R_{1,T}  | R_{2,B}  | 2.9413 | 3.5647 | 3.0970 | 3.6320 | 3.3088 |
| A1 GS + R1_D      | learned  | R_{2,B}  | 3.1039 | 3.7829 | 3.2112 | 3.7597 | 3.4644 |
| A2 GS + R2_D      | R_{1,T}  | learned  | 2.9788 | 3.6308 | 3.1295 | 3.6397 | 3.3447 |
| A3 GS + R1_D+R2_D | learned  | learned  | 3.1746 | 3.7663 | 3.2609 | 3.7640 | 3.4915 |
| A4 COMPOSED (diag)| A1's     | A2's     | 3.0508 | 3.7525 | 3.2202 | 3.7114 | 3.4337 |

Delta table (vs A0 unless stated; mean4 is descriptive only — cycles are
never pooled across datasets):

| Contrast | MT | GSM8K | ShareGPT | HumanEval | Mean4 |
|----------|------|-------|----------|-----------|-------|
| R1 effect (A1-A0)        | +0.1625 | +0.2182 | +0.1142 | +0.1277 | +0.1556 |
| R2 effect (A2-A0)        | +0.0375 | +0.0661 | +0.0325 | +0.0076 | +0.0359 |
| R2 after R1 (A3-A1)      | +0.0708 | -0.0166 | +0.0497 | +0.0043 | +0.0271 |
| R1 after R2 (A3-A2)      | +0.1958 | +0.1356 | +0.1314 | +0.1243 | +0.1468 |
| Interaction (A3-A1-A2+A0)| +0.0333 | -0.0827 | +0.0173 | -0.0033 | -0.0089 |

Per-seed AL is in `tables/al_by_dataset.csv` and figure 7; all three seeds of
every arm were evaluated on all four datasets (no median-only shortcut).

## 7. Statistics

Paired prompt-cluster bootstrap, 3000 resamples, prompts resampled with
replacement and the same resample indices applied to both arms; micro-tau
recomputed per resample. Holm step-down WITHIN each pre-registered family.
Empirical p of 0 is reported as `< m/3000` (Holm bound), never as 0.

**Family F1 — A1 vs A0 (effect of draft-aware R1): 4/4 significant.**

| Dataset | delta | 95% CI | Holm p | verdict |
|---|---|---|---|---|
| MT-Bench  | +0.1625 | [+0.1072, +0.2204] | < 4/3000 | SIG |
| GSM8K     | +0.2182 | [+0.1757, +0.2607] | < 4/3000 | SIG |
| ShareGPT  | +0.1142 | [+0.0372, +0.1944] | 0.0040   | SIG |
| HumanEval | +0.1277 | [+0.0777, +0.1774] | < 4/3000 | SIG |

**Family F2 — A2 vs A0 (effect of draft-aware R2): 1/4 significant.**

| Dataset | delta | 95% CI | Holm p | verdict |
|---|---|---|---|---|
| MT-Bench  | +0.0375 | [-0.0147, +0.0891] | 0.516  | n.s. |
| GSM8K     | +0.0661 | [+0.0250, +0.1071] | 0.0107 | SIG |
| ShareGPT  | +0.0325 | [-0.0291, +0.0914] | 0.603  | n.s. |
| HumanEval | +0.0076 | [-0.0423, +0.0587] | 0.741  | n.s. |

**Family F3 — A3 vs A1 (incremental R2 after R1_D): 0/4 significant.**

| Dataset | delta | 95% CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|
| MT-Bench  | +0.0708 | [+0.0046, +0.1360] | 0.0407 | 0.163 | n.s. |
| GSM8K     | -0.0166 | [-0.0624, +0.0317] | 0.479  | 0.957 | n.s. |
| ShareGPT  | +0.0497 | [-0.0207, +0.1210] | 0.187  | 0.562 | n.s. |
| HumanEval | +0.0043 | [-0.0490, +0.0519] | 0.911  | 0.957 | n.s. |

**Exploratory (outside the confirmatory families, labeled as such).**
A3 vs A2 (R1 after R2_D) is significant on all four datasets: +0.1958,
+0.1356, +0.1314, +0.1243, every CI excluding 0 with p < 1/3000. A3 vs A0:
+0.2333, +0.2017, +0.1639, +0.1320, all p < 1/3000.

**Seed-matched secondary analysis (pre-declared in Amendment 3, before any
p-value was computed).** Because A3's between-seed spread on MT-Bench is
about twice A1's (range 0.108 vs 0.046), the median-seed F3 contrast carries
between-seed noise. Computing A3_sX - A1_sX within each seed and averaging,
with the same paired prompt-cluster bootstrap:

| Dataset | mean delta | per-seed | 95% CI | p |
|---|---|---|---|---|
| MT-Bench  | +0.0137 | +0.042, +0.031, -0.032 | [-0.0210, +0.0454] | 0.420 |
| GSM8K     | +0.0015 | +0.026, -0.023          | [-0.0309, +0.0324] | 0.947 |
| ShareGPT  | -0.0055 | -0.005, +0.006, -0.017 | [-0.0498, +0.0393] | 0.812 |
| HumanEval | +0.0230 | +0.053, -0.004, +0.020 | [-0.0056, +0.0507] | 0.118 |

None significant, and the sign flips across seeds. This confirms that the
median-seed F3 point estimate on MT-Bench (+0.0708, raw p 0.041) does not
survive either multiplicity correction or seed matching.

**Interaction.** The factorial interaction A3 - A1 - A2 + A0 is small and
sign-inconsistent across datasets (+0.033, -0.083, +0.017, -0.003; mean4
-0.0089). It is descriptive only. Under the prompt's interpretation rule
this is closest to "approximately additive-to-slightly-overlapping", but the
per-dataset sign inconsistency means the interaction is not resolved by this
experiment; the seed-matched F3 result (no incremental benefit) is the
load-bearing evidence. We do NOT claim R1_D and R2_D occupy "the same basin"
— no landscape or connectivity evidence was collected.

## 8. RCAL (same-proposal replay, T4 + MT-Bench 80)

Exact same-proposal replay against an FP16 reference verifier held in
lockstep on a second GPU; metrics on accepted PROPOSAL tokens only (so
AL_q = tau - 1). Paired prompt-cluster bootstrap, 3000 resamples.

| Arm | AL_q | AL_0 | RCAL | SAL | LAL | AFS |
|---|---|---|---|---|---|---|
| A0     | 1.9413 | 1.8184 | 1.5754 | 0.3659 | 0.2430 | 0.8380 |
| A1_MED | 2.1039 | 1.9786 | 1.6868 | 0.4171 | 0.2918 | 0.8264 |
| A2_MED | 1.9788 | 1.8808 | 1.6197 | 0.3591 | 0.2612 | 0.8393 |
| A3_MED | 2.1735 | 2.0308 | 1.7284 | 0.4451 | 0.3025 | 0.8222 |

| Pair | dAL_q [CI] | dRCAL [CI] | deceptive-AL flag |
|---|---|---|---|
| A0→A1 | +0.1625 [+0.109, +0.219] | +0.1114 [+0.056, +0.164] | no |
| A0→A2 | +0.0375 [-0.014, +0.092] | +0.0442 [-0.006, +0.095] | no |
| A1→A3 | +0.0696 [+0.006, +0.133] | +0.0416 [-0.023, +0.106] | no |
| A0→A3 | +0.2321 [+0.173, +0.297] | +0.1529 [+0.092, +0.214] | no |

No arm triggers the deceptive-AL flag. R1_D's gain is substantially
reference-consistent: 69% of its AL_q increase is matched by an RCAL
increase whose CI excludes zero, with the remaining 31% being
verifier-specific acceptance. For A2, the point estimate of dRCAL (+0.044)
slightly EXCEEDS its dAL_q (+0.038) and its verifier-specific acceptance
(SAL) actually falls (-0.007) with AFS rising — so the concern that a
learned R2 might buy compatibility with the W4A4 verifier without agreeing
better with the FP16 reference is not what happened; but neither delta is
statistically significant, so this is a direction, not a finding. For the
A1→A3 increment, dAL_q's CI excludes zero while dRCAL's does not, i.e. the
reference-consistency of that small increment is unproven.

## 9. Mechanism

**R2 geometry (median-val seeds).** Learned R_{2,D} vs baseline R_{2,B}:

| Arm | geodesic | Frobenius | max elem change | median eigenangle | planes > 0.1 rad | generator ‖B‖_F |
|---|---|---|---|---|---|---|
| A2 (R2 alone)  | 11.502 | 14.257 | 0.4498 | 1.381 rad (79°) | 124/128 | 23.25 |
| A3 (joint)     | 3.845  | 5.321  | 0.1981 | 0.359 rad (21°) | 108/128 | 5.69  |

Across all three seeds the pattern is the same: trained alone, the R2
generator norm is 16.3-23.2; trained jointly with R1_D it is 5.62-5.78 —
about four times smaller and remarkably consistent. **Once R1_D is free to
adapt the residual basis, the optimizer needs far less attention-local
correction.** This is direct, AL-independent evidence that the two rotations
address overlapping degrees of freedom.

**Is the R2 gain explained by conventional quantization geometry? No.**
W4 NMSE is essentially unchanged at every site — v_proj 0.012085 (A0) →
0.012081 (A2) / 0.012103 (A3); o_proj 0.013300 → 0.013250 / 0.013314;
W_rec 0.017576 → 0.017576 / 0.017915. A2 does slightly improve the A4
activation error at its own boundary (o_proj input a4_qnmse 0.03603 →
0.03484, -3%), while A3 makes it worse (0.04306) yet has the higher AL.
**AL does not track the quantization proxies**, replicating the earlier
R1_D finding. We therefore do NOT claim NMSE explains any of the gain.

**Per-depth full-vocabulary overlap (held-out corpus tail, K=1..4):**

| Arm | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| A0     | 0.348 | 0.299 | 0.196 | 0.186 |
| A1     | 0.498 | 0.523 | 0.484 | 0.437 |
| A2     | 0.373 | 0.318 | 0.238 | 0.221 |
| A3     | 0.512 | 0.540 | 0.495 | 0.430 |
| A4 comp| 0.396 | 0.396 | 0.326 | 0.277 |

**Composition control (A4).** Bolting the independently-trained R_{2,D}
onto the independently-trained R_{1,D}, with no joint fine-tuning, *reduces*
AL relative to R_{1,D} alone (mean4 3.4337 vs 3.4644, -0.031) and markedly
degrades per-depth overlap (0.396/0.396/0.326/0.277 vs A1's
0.498/0.523/0.484/0.437). Each rotation was optimized assuming the other
sat at its baseline, so they do not compose. Joint training recovers and
slightly exceeds A1. This is a diagnostic arm, not a confirmatory one.

**Random-R2 control (matched generator Frobenius norm).** Three random
residual Cayley perturbations were built with ‖B‖_F matched *exactly* to
the learned A2 generator (23.2495 in all three, by construction). Their
geodesic distances (11.427/11.431/11.445) also landed within 0.7% of the
learned rotation's (11.502), although geodesic magnitude was NOT explicitly
matched — only the generator norm was, and only that is claimed. MT-Bench:

| | delta vs A0 | mean |
|---|---|---|
| Learned R_{2,D} (3 seeds) | +0.0798, +0.0354, +0.0375 | +0.0509 |
| Random R2, matched ‖B‖_F  | +0.0140, +0.0176, +0.0274 | +0.0197 |

The learned direction outperforms matched random perturbation on average
(2.6x), but the distributions overlap (learned min +0.0354 vs random max
+0.0274) and, notably, *arbitrary* R2 perturbation of this magnitude is
also mildly positive. Since neither the learned A2 effect nor this gap is
significant on MT-Bench, the honest reading is that the baseline seed-0
Haar R_{2,B} is not a specially good attention basis, and that most of the
R2 gauge is degenerate with respect to acceptance.

## 10. Runtime and foldability

**Operator-level proof (the load-bearing evidence).** A torch profiler pass
over real MT-Bench decoding records the same **10 distinct CUDA kernels** in
all four arms, with identical identities; residual call-count differences
(max 34 out of ~3000 calls) track the number of generation cycles rather
than any added operator. Neither R_{1,D} nor R_{2,D} introduces a kernel:
both are folded offline into weight buffers that already exist (F0). R2_D in
particular reuses the same `conjugate_v_o` V/O fold as the baseline, and the
model-level audit confirms no `[128,128]` buffer survives in any installed
runtime module. Evidence: `runtime/operator_audit.json`,
`tables/profiler__*.txt`, `audit/r2_contract_audit.json`.

**Wall-clock, measured on a quiesced host** (all other jobs finished; four
arms benchmarked sequentially on GPU 0; fake-quant, batch 1, greedy,
official tree):

| Arm | ms/token | draft ms/cycle | verify ms/cycle | postProjR1 ms/cycle | draft/verify | vs A0 |
|---|---|---|---|---|---|---|
| A0 | 51.69 | 25.871 | 109.868 | 0.5913 | 0.23547 | — |
| A1 | 51.03 | 25.778 | 108.346 | 0.5977 | 0.23792 | +1.04% |
| A2 | 63.70 | 30.514 | 130.691 | 0.6454 | 0.23348 | -0.85% |
| A3 | 53.38 | 27.470 | 117.104 | 0.6139 | 0.23458 | -0.38% |

The absolute numbers still spread by up to 23% (A2 vs A0), but this is
measurement noise, not an arm effect, and the data say so internally: the
**verify** cost is a provably arm-invariant workload — every arm verifies
with the same target model and the same weights, and no draft-side rotation
can touch it — yet it spans 108.3-130.7 ms/cycle (21%). Normalizing draft
cost by that internal control collapses the spread to **within 1.9% across
all four arms**. Combined with the identical kernel sets, this supports the
operator-level claim.

An earlier set of measurements taken while 6-7 other GPU jobs were running
is retained in `tables/runtime__A*.json` and is NOT used for any claim; the
quiesced re-measurement (`runtime__QUIESCED_*.json`) supersedes it. That
first attempt is what surfaced the contention problem: A1 appeared 33%
slower than A0, with verify inflated by the same factor.

**Preparation cost (separate from inference cost).** Each rotation is
trained once, offline: 3000 steps at ~2.1 h on one RTX 4090 per seed, plus
the one-time R_{1,T} SpinQuant optimization (5 h 51 min on 8 GPUs) and the
5000-window corpus build (~42 min x 5 GPUs). Deployment then loads a folded
state dict — R_{2,D} adds 128x128 fp64 (128 KB) to a checkpoint, and nothing
to the runtime.

**Scope limit.** All quantization here is fake (quantize-dequantize with
FP16 matmuls). None of these timings is an INT4 deployment throughput claim.

## 10.5 Relation to how SpinQuant itself trains rotations

Worth stating explicitly, because it changes how the arms should be read.
SpinQuant optimizes **R1 and every per-layer R2 jointly** — one parameter
list, one optimizer, one loss (`optimize_rotation.py:103-108`:
`trainable_parameters = [model.R1.weight] + [layers[i].self_attn.R2.weight
for i in range(num_hidden_layers)]`, handed to a single `SGDG(...,
stiefel=True)`). There is no alternating or staged schedule. Its objective is
plain next-token cross-entropy on wikitext-2 with W4A4 fake-quant in the
loop, over 100 steps at global batch 8, with rotations initialized to random
Hadamard and kept on the Stiefel manifold by a 5-iteration Cayley loop plus
occasional (~1%) QR re-normalization.

Mapping that onto this study:

| Arm | Relation to SpinQuant's own procedure |
|---|---|
| A1 (R1_D only) | not a SpinQuant configuration — R2 held fixed |
| A2 (R2_D only) | not a SpinQuant configuration — R1 held fixed |
| **A3 (joint)** | **SpinQuant's procedure, with the objective swapped from target CE to the draft's acceptance-aware surrogate** |
| A4 (composed)  | something SpinQuant never does |

Two consequences. First, the A4 result (composition falls below A1) is an
independent confirmation of SpinQuant's design choice: separately-optimized
rotations do not compose, so joint optimization is the right default.
Second, and more pointed: **even when we follow SpinQuant's joint procedure,
the R2 component contributes nothing measurable for the draft.** A3 vs A1 is
non-significant on 4/4 datasets, and the jointly-trained R2 generator settles
at 5.62-5.78 versus 16.3-23.2 when trained alone — the optimizer, free to
move both, largely declines to move R2.

A plausible structural reason (stated as interpretation, not measurement):
the target has 32 independent per-layer R2 matrices, each with its own share
of the CE objective, whereas this draft has a single decoder layer and one
R2 shared across all 32 heads. In that geometry the residual-stream basis
R1 can absorb what the single attention-local rotation would otherwise do.
Whether the conclusion would change for a multi-layer draft is untested here
and is the natural follow-up.

## 11. Scientific conclusion

> **"Draft-awareness is primarily required for R1; R2 can remain fixed."**

The evidence, in the order it constrains the conclusion:

1. R_{1,D} is significant on 4/4 datasets (family F1), replicating the prior
   LS finding under the GS scaling policy for the first time.
2. R_{2,D} alone is significant on 1/4 (family F2, GSM8K only), with a mean4
   effect roughly one fifth of R_{1,D}'s.
3. **Adding R_{2,D} on top of R_{1,D} is significant on 0/4** (family F3),
   and the pre-declared seed-matched analysis agrees with sign flips across
   seeds.
4. The converse — adding R_{1,D} on top of R_{2,D} — is significant on 4/4.
5. Mechanistically, when R_{1,D} can adapt the residual basis, the optimizer
   requests four times less attention-local rotation (generator norm
   16.3-23.2 alone vs 5.62-5.78 jointly, consistently across seeds).

What we do **not** conclude: that R2 is unnecessary in general. R_{2,D} was
implemented correctly (six executable gates including four-leg bitwise
trainer/runtime parity), tuned (LR selected on held-out validation loss
only), and evaluated at full confirmatory scale; it does move acceptance a
little on its own. The defensible statement is narrower and is the one this
study supports: **in a deployment that already learns a draft-aware residual
rotation, learning the attention-local rotation buys no measurable
additional acceptance length.** The attention-local V/O basis transfers
adequately from the target-style construction; the residual-stream basis,
which governs both the target->draft interface and the recurrent trajectory,
does not.

Secondary result worth reporting: independently-optimized R_{1,D} and
R_{2,D} do **not** compose (arm A4 falls below A1 on mean4 and markedly on
per-depth overlap). If both are ever learned, they must be co-trained.

## 12. Artifacts

| Item | Path |
|---|---|
| Run directory | `runs/eagle1_draft_aware_r1_r2_gs_20260810_065036/` |
| Pre-registration (+3 amendments) | `runs/.../configs/preregistration.md` |
| R2 contract audit | `docs/EAGLE1_DRAFT_AWARE_R1_R2_GS_AUDIT.md`, `runs/.../audit/r2_contract_audit.json` |
| Gates | `runs/.../gradchecks/{gate_r2_ab,gate_gs_r1r2_parity,param_audit__*}.json` |
| R_{1,T} rotation | `outputs/rotations/learned_chat_w4a4kv16/R.bin` (sha256 `0a2d1997…bbaa`) |
| Selected rotations | `runs/.../rotations/RD_GS_{A1_s1001,A2_s1003,A3_s1002}.pt` (+ `.sha256`) |
| Controls | `runs/.../rotations/{COMPOSED_A1R1_A2R2,RANDR2_s700{0,1,2}}.pt` |
| Corpus manifest | `runs/.../manifests/lkcorpus__rd_gs_all.json` |
| Tables | `runs/.../tables/{final_summary.json,al_by_dataset.csv,bootstrap_summary.csv,rcal_summary.csv,runtime_foldability.csv,rotation_geometry.csv}` |
| Statistics | `runs/.../stats/{bootstrap_pairs_*.json,holm_adjusted.json,rcal_bootstrap_mtbench.json}` |
| Mechanism | `runs/.../geometry/r2_mechanism.json` |
| Runtime | `runs/.../runtime/{operator_audit.json,runtime_summary.json}` |
| Figures (PDF+PNG+CSV) | `runs/.../figures/fig1..fig8` |
| Reports | `runs/.../reports/{PAPER_CLAIMS.md,KOREAN_SUMMARY.md}` |
