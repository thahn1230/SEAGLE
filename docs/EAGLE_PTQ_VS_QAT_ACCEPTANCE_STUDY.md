# Training-Free PTQ versus Draft QAT for Quantized EAGLE-1

**Status: IN PROGRESS — skeleton with completed PTQ-family results; QAT
trainings running. Final numbers, verdicts, and the executive summary are
filled in by the finalization pass.**

Run: `runs/eagle_ptq_vs_qat_al_20260723_143129` ·
Branch: `exp/eagle1-ptq-vs-qat-al-study` ·
Models: Llama-2-7b-chat (f5db02db) + yuhuili/EAGLE-llama2-chat-7B (44e37ec) ·
R_T: `outputs/rotations/learned_chat_w4a4kv16/R.bin`
(sha256 4b7e91d2…a48fa6e) · GPUs 0–5 (6 reserved, 7 lost).

## 0. Research question

When applying INT4 to EAGLE-1, is QAT on the draft necessary, or does a
training-free structural PTQ method (rotation-consistent interface + P3
projection migration, all weights frozen) retain enough Average Acceptance
Length (tau) to be statistically non-inferior within a pre-registered
margin?

Pre-registered criteria (section 14 of the spec; fixed before any QAT
result was seen):

- **Criterion A** — non-inferiority: lower 95% CB of
  (tau_strictPTQ − tau_QAT) > −0.05, paired prompt-cluster bootstrap.
- **Criterion B** — recovery: recovery_ratio =
  (tau_strictPTQ − tau_naive)/(tau_QAT − tau_naive) ≥ 0.90 AND
  |tau_QAT − tau_strictPTQ| ≤ 0.10 AND no target-quality regression.

## 1. Study design

19 cells (C1–C14 + N1–N5; N4≡C5, N5≡C12 definitional aliases), two target
precisions (FP16; INT4 = learned-R_T SpinQuant W4A4 KV16 RTN+w_clip), one
draft INT4 contract (**D4P3**: first/recurrent projection + AR linears
W4A4, P3 α-migration, embedding+LM-head FP16, KV16). Metric: official
micro-AL tau = Σ(accepted+1)/cycles, MT-bench 80, greedy, tree
mc_sim_7b_63. Cross-domain: sharegpt:80, c4:200, gsm8k:200, humaneval:all.

Interface structures (section 7): identity (FP16 target),
gamma_R1 shared-basis (INT4 target), restored (INT4 target + stock FP16
draft). The runtime draft is **internally R1-rotated in every mode** —
"identity" refers only to the first-path input fold; W_rec=[W_e|W_h·R1],
post-projection R1, R1-conjugated AR linears, head W_lm·R1. QAT trains in
exactly this deployed basis (Gate D).

### QAT implementation (audited: `tables/qat_pipeline_audit.json`, PASS)

Original EAGLE objective preserved exactly:
`loss = 1.0·SmoothL1(pred_hidden, target_hidden) + 0.1·softCE(head)` with
loss-mask weighting; AdamW lr 3e-5 β(0.9,0.95), clip_grad_VALUE 0.5,
linear warmup 2000 → linear decay; uniform input noise (rand−.5)·0.2·512/L;
max_len 2048; head + embedding frozen (as in original cnets); all other
draft params trainable. STE W4A4 fake-quant only at the deployed INT4
sites via the official SpinQuant quantizers with runtime cast order.
Teacher = fused frozen-target forward per batch (identical signal to
pre-generated hiddens). Budget: 6000 optimizer steps (bs1×accum4 ≈ 2
epochs over 12,000 pinned ShareGPT conversations), identical across every
trained arm. Deviations from the original pipeline are enumerated in
`manifests/study_config.json`.

Arms: C3 (FP16-target QAT, seeds 0/1/2) · C7 (INT4-target QAT, seeds
0/1/2) · C7b (C6-initialized QAT) · C6 (INT4-target-adapted FP16 retrain)
· C8 (FP16 retrain control). Derived: C9=C8+PTQ, C10=C6+PTQ,
C13=C8@INT4-target, C14=C3@INT4-target.

### Alpha calibration (section 8)

Pre-registered grid {8, 11.31, 16, 22.63, 32, 45.25, 64, 90.51, 128},
selected on the **c4 calib pool (offset-500, 20 prompts — disjoint from
all evaluation and training data)**. Central sweep results (tau):

| α | fp16 target | int4 target |
|---|---|---|
| 16 | 1.7333 | (grid completing) |
| 22.627 | 2.0801 | (grid completing) |
| 32 | 2.4116 | (grid completing) |
| **45.255** | **2.6701** | **best of swept** |
| 64 | 2.5899 | 2.9589 |

**Selected: α\* = 45.254834 for BOTH targets**, used identically for
strict PTQ and QAT deployment (fairness, section 17). The historical
default α=32 int4 run is retained as sensitivity record `C5a32sens`
(tau 2.9973 — the calibrated α is confirmed on mtbench by the C5 rerun).

## 2. Completed results (MT-bench 80, greedy, micro-AL tau)

| ID | Target | Draft | tau |
|---|---|---|---:|
| C1 | FP16 | stock FP16 draft (**Gate B: reproduces 3.5762 exactly**) | 3.5762 |
| C2 | FP16 | strict TF PTQ D4P3 @α\* | 2.9065 |
| C4 | INT4 | stock FP16 draft, restored interface | 3.2744 |
| C5 | INT4 | strict TF PTQ D4P3 (α32 sens: 2.9973; @α\* rerun pending) | TBD |
| C11 | FP16 | rotation-optimized PTQ (LK R_D) | 2.7888 |
| C12 | INT4 | rotation-optimized PTQ (LK R_D) | 3.1816 |
| N1 | FP16 | naive W4A4, no P3 | 1.0444 |
| N2 | INT4 | naive W4A4, no P3 | 1.2521 |
| N3 | FP16 | P2 branchwise scales, no P3 | 1.2012 |
| N3b | INT4 | P2 branchwise scales, no P3 | 1.9818 |
| C3/C6/C7/C7b/C8/C9/C10/C13/C14 | | training in flight | TBD |

Already-visible structure:

- **P3 recovery is enormous**: naive → P3 is +1.86 (fp16) / +1.75 (int4).
  Whatever QAT adds sits on top of a mostly-structural recovery.
- **C11 < C2 (−0.12)**: the LK acceptance-aware R_D was optimized for the
  gamma_R1 (INT4-target) interface; deployed under the FP16-target
  identity interface it *hurts*. Rotation optimization is
  interface-specific — reported as measured.
- **C4 = 3.2744**: target INT4 costs −0.30 tau with a stock FP16 draft.
- Curious inversion C5(α32) 2.997 > C2 2.907 despite the harder target —
  the rotated interface + rotated target pair is a better-matched
  quantization geometry than identity D4P3 under FP16.

## 3. Gates

| Gate | Status | Evidence |
|---|---|---|
| A provenance | **PASS** | Provenance.txt; R_T sha match; models pinned |
| B C1 reproduced | **PASS** | 3.5762 == known stock baseline |
| C strict PTQ from validated P3 code | **PASS** | C2/C5 run through the P2–P3-audit-validated adapter |
| D QAT forward == deployment | **PASS** | `tables/gateD_qat_parity.json`: bitwise weight hashes + K=4 chain logits/tokens equal, BOTH structural modes, at α\*. Root-caused two 1-ULP parity breakers: (i) fp16 α-division rounding (CPU vs GPU) at non-power-of-2 α; (ii) STE reconstruction w+(qw−w) re-rounding. LK reference harness (α=32) also PASS |
| E objective preserved | **PASS** | qat_pipeline_audit.json |
| F trainable scopes | **PASS** | audit + test_qat_trainable_parameter_list |
| G teacher matches named target | **PASS (construction + algebra test)** | teacher transforms: x=interface input, t=(h-basis)·R1; test_teacher_target_transform_consistency |
| H no dataset overlap | **PASS** | 62 train convs excluded matching eval/calib/valid pools; calib pool offset-500; manifest sha |
| I QAT seeds complete | pending (3+3 seeds in flight) |
| J oracle validated on toys | **PASS** | oracle self-test + tests |
| K official tau aggregation | **PASS** | test_official_tau_aggregation vs aggregate_micro_al |
| L speculative fidelity | pending (jobs queued) |

Trainer validation: 30-step C3 smoke (val top-3 0.797), 8-step C7 smoke
(int4 teacher, top-3 0.868), leaf-gradient accumulation scheme verified
against full STE backward (cos=1.000000, relerr ≤ 4e-4 on every trainable
tensor), peak memory 21–22 GiB on 24 GiB cards.

## 4–14. (filled by finalization)

Primary table with CIs and oracle utilization; attainment metrics;
criteria A/B verdicts; decision logic A–F; ceilings (stock / empirical
trainable / oracle-draft / policy-theoretical); seed variance;
teacher-mismatch; retrain-then-PTQ vs QAT; cost accounting; memory;
speculative fidelity; cross-domain; 14 plots; answers to the 14 report
questions; executive recommendation.
