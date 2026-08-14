# R1DCE — Draft R1_D Extension to H_t: PREREGISTRATION (frozen 2026-08-14)

Study: can the existing Draft global rotation R1_D, applied as
`H_t -> H_t @ R1_D` BEFORE the ctx A4 quantizer (with paired ctx K/V weight
views), replace the separate context rotation R_C?  Structural
simplification / mechanism study; follow-up to VSQ+FIDI (CASE D, @682f666).

## Source of truth (validated, unchanged)

- Basis contract: `seagle_port/fidi_basis_audit.py` verdict D (@70cc502).
  H_t (bare-RMS of folded-W_c output) is in the ORIGINAL context basis;
  deployed R_C is CLASS A (`Ht = Ht @ rc_matrix_buf`,
  vsq_draft_rot.py:252) BEFORE the per-layer ctx A4 sites
  (`xc_k/xc_v = act_fake_ste(Ht, cab)`, vsq_draft_rot.py:277-278); ctx
  views = `quant((W_K γ_hid) @ R_C)` / `quant((R2ᵀ∘W_V γ_hid) @ R_C)`.
- Current deployed R_C = `--vsq-rc rt` = R1_T (s1 R.bin, sha f433a88b),
  confirmed in every M5/M6 scheduler command of the VSQ run.
- Draft rotation ckpt: `VSQ_RD/rotations/draft/R1D_s1r1.pt.best`
  (keys R1_D [4096²], R2_D 5×[128²]).  R2/O pairing UNTOUCHED (§18).

## Frozen protocol

- Validation (ALL selection): gsm8kvalid n=60, max_new 512, greedy, B=10,
  cycle-pooled tau (micro-tau); prompt macro AL reported secondary.
- Test (final only): mtbench80(2turn)/gsm8k200/humaneval164/sharegpt80,
  max_new 1024, greedy, B=10; 4-ds mean = dataset-level mean, no pooling.
- Stats: paired bootstrap n=3000 seed 0 keyed (prompt_id, turn), Holm over
  preregistered primaries; p floored at 1/n (never 0).
- Everything frozen: target weights, draft weights, W_c fold, R1_T, R1_D,
  R2_D, quantizer policy (W4 per-ch sym RTN no-clip; A4 per-token asym);
  P2 OFF and QAT OFF for the entire Phase-1 rotation-placement study.

## Phase-1 arms (validation AL; identical except the ctx rotation)

Common: `--target-mode w4a4 --rbin s1 R.bin --vsq-draft R1D_s1r1.pt
--interface stock --dataset gsm8kvalid --max-samples 60
--max-new-tokens 512` in the NEW run dir.

| Arm | tag | ctx rotation (`--vsq-rc`) |
|---|---|---|
| C0 | C0_none | (absent) — model-local SQ only |
| C1 | C1_r1d | rotations/rc_R1D.pt  (PRIMARY NEW ARM) |
| C2 | C2_r1t | rt (R1_T) |
| C3 | ≡ C2 | current R_C == R1_T proven in Phase-0; single run reported as C2/C3 |
| C4 | C4_had | FIDI tables/rc_hadamard.pt (seed 1234, validated) |
| C5 | C5_rnd_s101/s102/s103 | fresh QR-orthogonal, PREREGISTERED seeds 101/102/103; FIDI rc_random.pt (seed 1234) reused as supplementary C5_rnd_s1234 |
| C6 | C6_learned | DFST RC_L0.pt (only existing learned R_C; CAVEAT: trained in DFST era under w16a4kv16 target rbin + RCDraft pipeline — diagnostic only) |
| C7b | C7_combo | rotations/rc_combo_c7b.pt = R1_D @ R1_T (combined; C7a sequential proven ≡ C7b in GATE-6, no separate long run) |

No test dataset is used for any selection.  Random-seed selection on test
is forbidden; C5 is reported as mean over the 3 preregistered seeds.

## Preregistered primary comparisons (Holm family, validation)

P1: C0 vs C1 (does R1_D help at H_t?)
P2: C1 vs C2 (R1_D vs R1_T reuse ≡ current R_C)
P3: C1 vs C4 (R1_D vs Hadamard)
P4: C1 vs C5_s101 (R1_D vs random; seeds s102/s103 supplementary)
P5: C1 vs C7b (single vs combined rotation)
Secondary (no Holm): C2 reproduction vs VSQ-run V_M5_rconly shard
(bit-repro check), C6 vs C2.

## Decision Gate A (§10, frozen thresholds)

R1_D is "sufficient" iff BOTH:
 (a) validation tau(C1) non-inferior to tau(C2): bootstrap 95% CI of
     tau(C2)-tau(C1) upper bound < 0.10 AND Holm-p(P2) > 0.05 for
     superiority of C2; and
 (b) H_t A4 NMSE(C1) <= 1.10 × NMSE(C2)  (10% relative).
If Gate A passes: NO separate R_C; structural method := "draft-R1 context
extension"; proceed to P2/QAT with R1_D.  C8 (residual ΔR, Cayley,
R_context = R1_D·ΔR, ctx K/V W4A4 NMSE objective on calib split) and C9
(shared R1_DC, dual objective) trigger ONLY if Gate A fails.

## Phase order (§35 — binding)

1) C1 first, 2) C1 vs {C2/C3, C4, C5}, 3) C8 only if Gate A fails,
4) C9 optional after C8, 5) freeze structure -> P2 arms
(best, best+P2, P2-alone) -> QAT (Q5-recipe: SGD-mom 0.9, 400 steps,
lr 3e-2 primary with 3e-1/1e-1 pilots if CE stalls, same corpus
`VSQ_RD/tables/train_corpus.jsonl` + `hcache_w4a4_s1`, trainer-CE best
selection, validation confirm) -> final 4-ds F0..F6 -> RCAL.

Final 4-ds reuse (bit-reproducible greedy protocol, same host/env):
F0=M0, F1=M1, F2=M3, F4=M5a(+rt) shards are REUSED from the VSQ run with
provenance manifest; F3 (=selected C1 config), F5 (best+P2), F6 (+QAT)
run fresh.  If Gate A fails and R_C stays, F3 still runs (as the honest
R1_D result) and F5/F6 use the winning rotation.

## Mechanistic measurements (no selection role)

- §6 H_t candidate battery from hcache_w4a4_s1 (112 rows, FP-path H_t per
  FIDI convention; same paired tokens for every candidate).
- §8 ctx K/V A/W/A+W decomposition per layer (fidi_awdecomp conventions).
- §19 γ-contract audit (γ_hid vs per-layer γ_in) for view unification.
- §25 runtime: Ht@R matmul S∈{1,8,32,128,512} + fast-Hadamard if importable.
- GATES 1-9 must pass before any headline AL claim.

## GPU / cost plan

Host gpusystem, 8×4090; scheduler = seagle_port.scheduler on THIS run dir
only (old VSQ scheduler killed first; disjoint GPU rule).  Estimates:
Phase-1 validation ~9 jobs × ~40-70 min; mech captures ~1-2 GPU-h; final
4-ds fresh ~12 jobs × 1-4 h; QAT ~10 min.  Total ≈ 45-60 GPU-h.
Storage: ~3 GB captures + shards (<1 GB).
