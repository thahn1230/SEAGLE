# Baseline Audit — Target/Draft Precision × Method Grid (2026-08-05)

Branch `exp/eagle1-target-draft-precision-method-grid` @ 3cb3fb2.
Run dir: `runs/eagle1_target_draft_precision_method_grid_20260805_184316`
(pointer `runs/PMG_RUN_DIR`). Worktrees: main only. GPUs 0-7 all free at
audit time (operator: use all 8).

## Canonical assets (verified on disk)

| asset | path | id |
|---|---|---|
| Target | meta-llama/Llama-2-7b-chat-hf snapshot f5db02db | HF cache |
| Draft | yuhuili/EAGLE-llama2-chat-7B snapshot 44e37ec3 | HF cache |
| R_T | outputs/rotations/learned_chat_w4a4kv16/R.bin | sha 4b7e91d2… 69.2MB |
| R_D (T4-matched finalist) | <RDROT run>/rotations/RD_HYB_s2.pt | sha 20c03f00… |
| FP16 anchor (QAT init) | checkpoints/eagle1_fresh_fp16_anchor/anchor.pt | sha a7c6ccc8… 3.35GB |
| Tokenized ShareGPT cache | /data/…/eagle_tok_cache_v1.pt | int16 rows, 12000+400 |
| EP3-P int4 scales | tables/ep3_selection_int4.json (OF/GQ runs) | (β_f,β_r)=(0.40,0.45) |
| EP3-P fp16 scales | GQ study | (0.46,0.46), AFS 1.0 |
| EP3-G betas | eval_eagle1_generic_qat_vs_p3exp.py ep3g_beta() | fp16 0.46 → m 45.887; int4 0.42 → m 32.900 |
| LP3 legacy alphas | alpha_selected.json / alpha_calibration.json | fp16 45.254834, int4 32.0 |

## Existing precision-grid experiments

1. **BWAL 3×3** (random-Hadamard R1, naive pre-P3 D4):
   `artifacts/bitwidth_al_component_causality/final_matrix/analysis/matrix_al.csv`
   — T16D16 3.6276 … T4D4 1.2443. Own runner + per-prompt metric.
2. **LRAS 3×3** (learned_chat_w4a4kv16 rotation, same runner):
   `runs/eagle_learned_rotation_al_sensitivity_20260716_192113/precision_matrix_summary.csv`
   — T4D4 1.2557. Run-to-run stock-row noise ≈ 0.015 AL.
3. **TLDR-KV4 4×4** (T16/T8/T4kv16/T4kv4 × D16/D8/D4P3kv16/D4P3kv4,
   micro-AL, 80 mtbench): T8 row D16 3.5396 / D8 3.3539 / D4P3 3.3588.
4. SEAGLE INT4 e2e "validated taus" are IMPORTED constants from the
   fake-quant studies (PTQ-vs-QAT C1/C4 + GQ EP3-P), not measured there.

**Gap driving Experiment A re-run**: no grid with (a) the official
micro-tau evaluator (`eval_eagle_acceptance_length.py`), (b) 4 datasets,
(c) one consistent modern contract across all 9 cells. Old grids serve
as regression baselines only (BWAL/LRAS D4 cells are pre-P3 naive —
matches our A-grid D4 definition; TLDR D4P3 matches B5-lite).

## Quantizer policies (canonical, from code)

- **Target W4A4**: SpinQuant full (R1+R2+R4; R3 iff KV<16 — here KV16),
  RTN w4 per-channel sym + clip, per-token asym A4, k/v16,
  KIND=learned_chat_w4a4kv16 (study.py:328-347, 393-411).
- **Target W8A8**: SAME pipeline, w8a8 key (w8 RTN clip, a8 asym,
  KV16). Never plain-RTN-without-rotation — canonical is rotated.
- **Draft W4A4/W8A8** (fake_w4a4/fake_w8a8, concat_selective QUANT_BITS):
  projection first/rec + AR q/k/v/o/gate/up/down quantized (weights
  per-out-channel sym RTN + MSE clip; acts per-token asym);
  PostProjectionR1 fp32; embedding, draft lm_head, norms, softmax,
  RoPE, residual, KV cache fp16. "Draft W4A4" ≠ whole-draft-4-bit.

## QAT infrastructure

- Canonical trainer `scripts/train_eagle_draft_int4_qat.py`: official
  EAGLE loss (SmoothL1 + 0.1 softCE), frozen rotations/scales/embed/
  head, trainable = 11 draft-core tensors, anchor init, STE at deployed
  sites, leaf-grad accumulation. Arms C3 (fp16 teacher, identity) / C7
  (W4A4 teacher, gamma_R1).
- OF-study protocol (canonical fairness template): LR pilot 1000 steps
  × {1e-6, 3e-6, 1e-5} → 3 seeds × 3000 steps, ckpt every 500,
  offline best-val selection on c4 calib-20, then mtbench. Existing
  ckpts: Q0/Q1 3 seeds (lr 1e-6, LP3 alphas), GQAT_T0/T1 3 seeds
  (lr 1e-5, alpha=1) — ~146GB total.
- **EP3-P-fold QAT never run** (gap). ExactQuantizedRotationForward
  already supports alpha_rec_init + train_draft_core → plumbing only.
  Caveat: single-step loss gives alpha_rec only indirect gradient via
  shared W_e/W_h masters (documented; same for all arms).
- **Joint R_D+core QAT never run** (optional ablation only).

## Evaluator contracts

- Official tau = micro cycle-pooled (accept_length+1), mc_sim_7b_63,
  greedy, max_new_tokens 128, prompt truncate 1024/512, pools:
  mtbench 80 single-turn requests (turn 1 only), gsm8k eval 200,
  sharegpt eval 80, humaneval 164; calib pools disjoint (offset-500 /
  c4 calib).
- Macro-AL computed NOWHERE → new small script required (from
  shards CSV acceptance_list).
- RCAL proposal-only (AL_q = tau−1 exactly), lockstep FP16 reference,
  paired prompt-cluster bootstrap 3000 (seeds 20260723/20260731).
- Holm: scripts/holm_adjust.py exists but ORPHANED + schema-mismatch →
  new adapter needed.
- `--target` in official eval/capture = {fp16,int4} only → **w8a8
  option must be added** (build_study_target already supports it).
- **No target-quality harness** (gsm8k EM, humaneval pass@1) → build
  minimal fresh (with honest max_new_tokens caveat).

## Interface / folding facts (details in RUNTIME_FOLDING_CONTRACT.md)

- Restored stock interface h=(a_t@R_Tᵀ)⊙γ is F0-foldable and the fold
  EXISTS (first_hidden_mode="gamma_R1" is exactly that fold); measured
  equivalence 3.5703 == 3.5703 (fp16 draft). GQ trap (missing restore
  wrapper → AL 0.15 collapse) documented only in shell header + log —
  written up here.
- Explicit runtime ops today: PostProjectionR1 dense fp32 GEMM (every
  draft forward), m_rec e-slice multiply, R4 online Hadamard,
  R-EP3-P StructuredRotation.apply (not used in this study's arms),
  restored-interface fp32 restore (stock arms).
- No latency instrumentation exists for ConcatSelective arms → Phase 8
  adds timers.
