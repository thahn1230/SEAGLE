# Adversarial self-audit — DFlash × SpinQuant novelty study (10 questions)

Companion to `docs/DFLASH_SPINQUANT_NOVELTY_STUDY.md`. Each question is posed
as hostilely as we know how; answers cite artifacts.

**Q1. Did any test-set number influence any selection?**
No. Every method/LR/checkpoint choice traces to gsm8kvalid or trainer-side
held-out CE, with timestamps: the M6 configuration was frozen and queued
(09:14, `RD/manifests/QAT_SELECTION_NOTE.md`) before any M6 test shard
existed; the Q5 LR amendment used trainer CE only. The component/I7 arms are
validation-only. Residual exposure: canonical M-arm test numbers existed
before FIDI began — but FIDI added no M-arm re-selection; its only new test
runs are the four M6 shards under the frozen config.

**Q2. Is the "vanilla SpinQuant fails" claim an artifact of applying it
wrongly?**
The opposite arm exists: M2 (RAW, no interface folds) collapses to τ=1.0,
while M3 applies the mandatory folds and passes Gate C/D/E plus the FIDI R1f
FP-parity gate (rot_fp16 H_t == fp16 H_t, kurt 80.40 vs 80.37). The failure
we report survives *correct* application: H_t kurtosis/NMSE are unchanged by
construction because `fold_wc` must cancel R1_T for FP correctness — an
algebraic statement (§20 counterfactual), verified on two independent data
sources (hcache 69.5→69.4; deployed forward 180–210 → unchanged).

**Q3. Could the R_C gain be a generic effect of ANY orthogonal transform
(regularization), not target-basis reuse?**
Now directly measured (study §8): a fresh random orthogonal R_C scores
3.1781 and a random Hadamard 3.1950 vs R1_T-reuse 3.2132 — both
statistically indistinguishable (p=0.131 / 0.543). So YES: the benefit is
generic outlier diffusion by any orthogonal rotation at this interface.
The report therefore claims placement novelty (a rotation must be inserted
at H_t→ctx-K/V, which vanilla SpinQuant structurally cannot provide) and
explicitly does NOT claim target-basis-specific structure; R1_T reuse is
kept as the natural zero-training choice. The draft-residual rotations are
the contrast case where the learned basis DOES matter (random CE 4.83 vs
learned 4.47; G1 significantly worse).

**Q4. Are the QAT gains (P5) just training-on-more-data leakage into eval?**
The QAT corpus is test-disjoint by construction (gsm8kcalib trajectories +
sharegpt-calib offset-500 continuations; committed at
`RD/tables/train_corpus.jsonl`), identical for Q2/Q3/Q5, and P5 is a paired
comparison against M5 which shares every other component. The gain replicates
across all four datasets (p≤6.7e-4 each). Residual risk: distributional
similarity between sharegpt-calib and the sharegpt eval pool (disjoint rows,
same source) — flagged, affects only the sharegpt column (+0.087, smallest).

**Q5. Was the Q5 LR amendment (3e-2) p-hacking?**
It is a documented deviation with a training-side criterion only
(`QAT_SELECTION_NOTE.md`): pilot LR was Q2-init-scaled; at Q5's far lower
init CE, 3e-1 and 1e-1 moved nothing (best_step 0) — a null result, not a
choice among improvements; 3e-2 was the single additional probe and its CE
gain (2.644→2.389) preceded any AL evaluation. Validation then confirmed
independently (V_Q5b 3.501 > V_Q5 3.148).

**Q6. Is M6 invalid because P2 was not part of Q5's training?**
It is disclosed as eval-composed (train/eval granularity mismatch). The
mismatch could only hurt M6 — it wins anyway (V_Q5bp2 3.669 > V_Q5b 3.501,
p=3.3e-4; P5 positive on all four test sets). A Q6 arm trained with P2 is
listed as future work; its absence cannot flip any reported sign.

**Q7. The sharegpt file was lost and rebuilt — is the frozen-prompt contract
broken?**
Byte-identity: no (33 recipe variants failed to reproduce the old sha).
Functional identity: proven — the rebuilt file drives a greedy fp16 eval that
reproduces the canonical shard **bit-identically** (80/80 prompt tau vectors,
r=1.0000; `M0shk` vs `al__M0_fp16__fp16__sharegpt.csv`). Every sharegpt
pairing in the battery is therefore prompt-exact. The new sha is pinned.

**Q8. Do the FIDI captures measure the deployed system, or an instrumented
variant?**
The taps are regression-gated: old-code vs new-code forward outputs are
bitwise identical for bits∈{4,16}, with and without taps
(`FIDI/tables/fidi_gate_d.json`). Capture configs reuse the frozen deployed
checkpoints (s1 R.bin sha-verified, R1D_s1r1.pt.best) and replay a frozen
fp16 trajectory; R2/R3 stats therefore describe exactly the M3/M5 deployment
paths. The only synthetic element is the R0q arm, used interpretively and
flagged as such (report §6.4).

**Q9. Is "compensatory alignment" (§15) overclaimed as training causation?**
No causal wording is used: only Tier-1 (column-norm anti-correlation, W_c
Spearman −0.835; V ctx ≈ −0.25) and Tier-2 (ablation sensitivity ∝ channel
RMS, r 0.90–0.97) evidence exists; the DFlash initialization/trajectory
checkpoints required for Tier-3/4 are not public, and the report's §5.4/§5.13
answers say "exhibits compensatory alignment" exactly as §27 prescribes.

**Q10. Does the headline CASE D hold if any single pending/weak cell flips?**
The three legs are independently significant on all datasets where run:
R_C (P4a, 4/4), P2-on-RC (P3, 4/4), QAT (P5, 4/4). The two designed
non-significances (S2 seed equivalence; humaneval M6-vs-FP16 gap) support,
not carry, the conclusion. Weakest cells: restore-Q (+0.023, Holm-n.s.) and
restore-O (−0.008) — both are *null* claims in the report ("Q/O minor").
Removing them changes nothing. The single strongest attack surface remains
Q3 (rotation-genericity), answered above with its honest scope limitation.
