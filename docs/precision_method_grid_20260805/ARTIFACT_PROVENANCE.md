# Artifact Provenance — reuse vs fresh (2026-08-05)

## REUSED (implementation / checkpoints / baselines)

| artifact | provenance | use |
|---|---|---|
| eval_eagle_acceptance_length.py / capture / compute_rcal / bootstraps | @3cb3fb2 R_D study | primary evaluators (w8a8 target option ADDED this study) |
| ConcatSelectiveDraftAdapter + QUANT_BITS fake_w8a8/fake_w4a4 | GQ/TLDR era | draft precision policies |
| study.build_study_target + QUANT_CFGS w8a8/w4a4 | canonical | target builds |
| R_T learned_chat_w4a4kv16/R.bin sha 4b7e91d2 | SpinQuant opt | all rotated targets |
| RD_HYB_s2.pt sha 20c03f00 | @3cb3fb2 finalist | B7/B8 @ T4 primary |
| LK2_HYBRID_s2.pt | LK study | transfer-ablation only |
| anchor.pt sha a7c6ccc8 | OF study | all QAT inits |
| EP3-P (0.40,0.45) int4 / (0.46,0.46) fp16; EP3-G β 0.42/0.46 | GQ ep3sel + ep3g_beta | B3-B8 scales (T8 fresh) |
| train_eagle_draft_int4_qat.py (+ new --alpha-rec) | OF/GQ | all 12 QAT conditions |
| train_eagle_lk_rotation.py (+EP3-P fold, @3cb3fb2) | R_D study | T16/T8 R_D training |
| build_target_generated_lk_corpus.py | LK/R_D | T16/T8 teacher corpora |
| check_rd_ep3p_parity.py Gate-P | @3cb3fb2 | gate re-run |
| BWAL/LRAS 3×3 CSVs + TLDR 4×4 table | 7/15-7/18 | regression baselines ONLY (different runner/metric — never merged into new tables) |
| OF Q0/Q1, GQ GQAT_T0/T1 ckpts | OF/GQ | regression baselines; NOT reused as new-arm results (fold/alpha contract differs: LP3/alpha-1 vs this study's EP3-*) |

## FRESH (built/trained/measured this study)

1. `--target w8a8` in eval + capture; fake_w8a8 D8 cells via existing
   QUANT_BITS.
2. T8 EP3-G/EP3-P calibration (capture → grid → calib-AL select).
3. T16/T8 target-matched corpora + R_D trainings (2 runs).
4. 12 QAT conditions × 3 seeds (Generic/EP3-G/EP3-P/EP3-P+R_D ×
   T16/T8/T4) — EP3-P-fold QAT is a first (never existed).
5. LR pilot (fresh, shared across methods).
6. All 9 A-grid cells + all 24 B arms × 4 datasets with the official
   evaluator (old grids not contract-compatible).
7. compute_macro_al.py, Holm adapter, target-quality mini-harness,
   latency instrumentation (PostProjectionR1 timer etc.),
   folding_audit.json generator.

## Rules honored

- Nothing deleted/overwritten; new run dir + new branch.
- Old-vs-new number mismatches → contract-diff analysis first
  (BWAL/LRAS per-prompt runner vs official micro tau; TLDR micro-AL
  is proposal+1 same-class but 20-prompt-era pools differ).
