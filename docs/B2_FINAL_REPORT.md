# B2 SPLIT-PROJECTION STUDY — FINAL REPORT

Date 2026-07-14 · branch `exp/eagle1-spinquant-b2-split` · GPUs: physical 6,7 only
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=6,7`, asserted in every process)
· Models: Llama-2-7b-chat-hf + yuhuili/EAGLE-llama2-chat-7B (EAGLE **v1** @4a9cf3a)
· SpinQuant @8f47aa3, random-Hadamard R1, RTN · greedy, MT-bench, mc_sim_7b_63 tree
· Labels: FP16 / fake_W4A4(16) / real_packed_W4A4(16) — never conflated.
Evidence root: `artifacts/b2_split_study/` (CSV/JSON per claim below).

## 1-3. The two representations and why the projection must split (spec §24: 1-3)

- First draft forward of every verification cycle consumes the TARGET tail
  output. Stock target → `h_t = n_t·D_γ` (gamma INCLUDED, original basis);
  fused SpinQuant target → `a_t = n_t·R1` (gamma EXCLUDED — fused into the LM
  head; rotated basis). Verified from `ea_model.py:124-145`,
  `modeling_llama_kv.py:1074`, `cnets.py:593`.
- Recurrent forwards consume the draft's own AR output `h_d` (imitates the
  gamma-INCLUDED `h_t`; no draft final norm exists in the loop), rotated:
  `h_d_R = h_d·R1`.
- Because `D_γ R1 ≠ R1 D_γ` (measured commutator ‖C‖_F/‖D_γR1‖_F = **0.0925** with the real R1 and
  real target_final_rms_gamma; elementwise-γ misuse costs 9.4% rel-L2 per
  application, `commutator.json`), one fixed projection cannot serve both inputs.

## 4. Exact implemented formulas (row-vector, W [out,in])

    projection_first      = R1ᵀ·[W_e·R1 | W_f·D_γ·R1],  bias b·R1     (Arch B)
    projection_recurrent  = R1ᵀ·[W_e·R1 | W_f·R1],      bias b·R1
    Arch A folded first   = [W_e | W_f·D_γ·R1], recurrent = original fc
    draft head (Arch B)   = W_lm·R1 (no gamma);  fused target head = W_lm·D_γ·R1

gamma enters EXACTLY once, through projection_first. Concat order [embed|feature]
proven in source + runtime slice test (`test_b2_concat_order.py`).

## 5. FP numerical proof of equivalence (STOP GATE A — PASS)

`fp_equivalence.{csv,json}`, n=8 prompts × 48 tok, fp16, greedy:
- rotated fp16 target greedy == stock greedy: **8/8 exact**.
- FP00 stock 3.7208; **A-explicit = A-folded = B2-split = explicit-M_γ = 3.7538**,
  all with ea==naive token equality **1.0** (verifier preservation).
- tiny-fp64 chain through the real vendored decoder: split path rel-L2 < 1e-6 at
  every depth 1-5, top-1 agreement 1.0 (`tiny_chain_equivalence.json`).

## 6. Negative control: first projection reused recurrently (FP03)

- e2e: 3.2646 vs 3.7538 → paired drop **0.489, 8/8 prompts degraded**.
- algebraic (tiny fp64 chain): depth-0 error <1e-6 but depth≥1 rel-L2 > 1e-3 and
  growing — depth-dependent divergence proven (`test_b2_gamma_applied_once.py`).

## 7. Negative control: gamma omitted from first projection (FP04)

Collapse to **1.1295** (paired drop 2.62, 8/8). Also: N1 naive 1.1312,
N2 elementwise-γ 1.0384, N4 wrong orientation 1.1289, N5 γ-on-e-block 3.5132
(drop 0.241, 6/8 — mild because embedding-branch perturbation is second-order).

## 8. Dispatch proof

`projection_dispatch_trace.jsonl` + `projection_dispatch_summary.csv`:
projection_first fired **exactly once per verification round, always at
draft_forward_index 0, always feature_origin=target_first**; recurrent never at
index 0; per-prompt round counters reset (asserted in
`test_b2_tree_dispatch.py` / `test_b2_state_reset.py` — pass). Structural check
across the matrix: calls_recurrent = 4×calls_first in every config (tree has 4
recurrent levels); Q11 coverage: projection_first n=1119 = verification rounds entered
(1099 token-yielding rounds in the acceptance traces + 20 final truncated
rounds, one per prompt), recurrent 4476=4×1119, each AR linear 5595=5×1119,
all fake-quant counters live (`coverage__quant.csv`).

## 9. Real W4A4 dispatch (STOP GATE B — PASS)

`real_w4a4_dispatch.{txt,json}`, `real_w4a4_projections.csv`: BOTH packed
projections [4096,8192] execute through QuaRot CUTLASS INT4×INT4
(`quarot.sym_quant/matmul/sym_dequant`; per-channel sym weights absmax/7,
per-token sym int4 activations, INT32 accum; packed 16.8 MB each; KV fp16), real
output vs same-recipe fp emulation rel-L2 **1.8e-2**; real tinygemm W4A16
(`aten._weight_int4pack_mm`, group 128) rel-vs-fp16 0.101. Scale buffers are
independent per projection; measured coincidence of full-row scale VALUES is a
weight property: 100% of rows attain absmax in the shared embedding block
(e 0.308 vs h 0.129/0.073 mean |w|max), and D_γ amplifies the first h-block
≈1.76× — the h-block payload is quantized at e-block-dictated resolution.
No silent fallback: FakeW4A4Linear counters + module types asserted per run.

## 10-11. Acceptance results with paired 95% CIs (n=20 prompts, 64 tok, greedy)

ε = max(0.05, 1% baseline); paired prompt bootstrap 10k (`b2_effects.csv`):

| comparison | Δ accepted/round | 95% CI | decision |
|---|---:|---|---|
| rotation control FP02 vs stock | −0.004 | [−0.051, +0.032] | inconclusive at ε=0.05 / equivalent at ε=0.10* |
| **target-only** W4A16 | −0.311 | [−0.436, −0.184] | meaningful decrease |
| **target-only** W4A4 (A-explicit; B2-split identical) | **−0.408** | [−0.572, −0.255] | meaningful decrease |
| **target-only** W4A4KV4 | −0.493 | [−0.714, −0.287] | meaningful decrease |
| draft W4A16 first-only / recurrent-only | −0.732 / −0.725 | [−0.85,−0.59] / [−0.89,−0.57] | decrease (symmetric) |
| draft W4A4 **first-only** | **−2.000** | [−2.213, −1.797] | meaningful decrease |
| draft W4A4 **recurrent-only** | −1.439 | [−1.667, −1.220] | meaningful decrease |
| draft W4A4 both projections | −2.137 | [−2.381, −1.903] | meaningful decrease |
| draft W4A4 **AR head only** | −0.183 | [−0.273, −0.085] | meaningful decrease (mild) |
| draft W4A4 full path (Q01) | −2.143 | [−2.390, −1.906] | meaningful decrease |
| **both** Q11 vs stock | −2.247 | [−2.497, −2.012] | meaningful decrease |
| Q11 vs target-only Q10 | −1.839 | [−2.089, −1.597] | meaningful decrease |
| Q11 vs draft-only Q01 | −0.099 | [−0.136, −0.066] | meaningful decrease |

*CI slightly exceeds the −0.05 edge → formally "inconclusive" at ε=0.05, clearly
equivalent at ε=0.10; the point estimate is −0.004.

Config means: Q00 3.4158 · TQ1 3.1045 · Q10 3.0079 (=Q10s B2-split 3.0079) ·
TQ4 2.9230 · FP02 3.4116 · DQ_ar 3.2290 · DQ_w4a16 2.68 · DQ_rec 1.9723 ·
DQ_first 1.4119 · DQ_both 1.2741 · Q01 1.2684 · Q11 1.1692.

## 12. Target quality alongside acceptance

WikiText-2 PPL: FP16 6.945 → W4A16 8.938 → W4A4 10.627 → W4A4KV4 10.939.
Acceptance decreases monotonically with PPL (fig6). Common-prefix (768 fixed
positions, `common_prefix_summary.json`): target top-1 flip rate 16.7%
(margin-dependent: 65% at margin<0.5, 2% at margin>4); Δentropy +0.061 (mild
flattening); Δ(top1−top2 margin) −0.844; Δlog p(draft proposal) −0.036.

## 13. Alignment vs flattening vs degradation (spec §19/§24-13)

Target W4A4 quantization DECREASED acceptance overall; the mechanism decomposes
on fixed contexts as: overlap(p,q) 0.706→0.692 (net alignment LOSS). Of 768
positions: 456 within noise; 165 overlap decreases; 147 overlap increases, of
which 62 are **distribution flattening** (entropy↑, margin↓), 53 **incidental
rank flips**, 32 alignment-like — and since PPL degrades +3.68 and margins shrink
−0.84 concurrently, the correct description of the (minority) increase positions
is **degradation-induced alignment**, not improved target quality. When both
models are quantized, overlap(p_Q,q_Q)=0.0358 vs overlap(p_FP,q_Q)=0.0320: a
correlated-error alignment gain of only **+0.004** — real but negligible; e2e it
appears as sub-additive damage (interaction +0.31: combined loss 2.25 < sum of
parts 0.41+2.15) with NO acceptance recovery (Q11 < Q01).

## 14. Which draft component is responsible

The fc projection dominates: AR head W4A4 costs only −0.18, while any projection
W4A4 costs −1.4..−2.1. **projection_first is the single most damaging
component**, by direct paired contrast: first-only vs recurrent-only
Δ=−0.560, CI [−0.629, −0.486] (meaningful). Quantizing the recurrent projection
ON TOP of the first still costs a further, meaningful −0.138
(first-only vs both: Δ=+0.138, CI [+0.086, +0.196]); adding the AR head changes
essentially nothing more (both −2.137 vs full −2.143).

## 15. Does recurrent-projection error accumulate with depth? (fig4)

YES — and the split cleanly separates the two failure modes
(`b2_depth_alphas.csv`, chain-style α_d with cycles-reaching counts n):
- recurrent-only quant: α₁=0.888 (n=651; first step intact) → α₂=0.081 (n=578)
  → α₃=0.064 (n=47) → α₄=0.0 (n=3, small-n) — depth-accumulating divergence;
- first-only quant: α₁=0.248 (n=923; initialization destroyed), deeper depths
  recover to α₂=0.393 (n=229), α₃=0.289 (n=90), α₄=0.346 (n=26), α₅=0.556 (n=9,
  small-n) — no accumulation once past the damaged first step.
Weight-only (W4A16) damage is depth-symmetric (−0.73 both), so the
first-vs-recurrent W4A4 asymmetry is driven by ACTIVATION quantization of the
target-originated `a_t` input, not by the D_γ weight fold alone.

## Answers to the five research questions

**Q1 (target-only):** acceptance DECREASES — monotone in target precision
(−0.31 W4A16, −0.41 W4A4, −0.49 W4A4KV4), all CIs outside ε. No configuration
increased acceptance.
**Q2:** the aggregate change is a statistically meaningful decrease, not
variance. Locally, 19.1% of fixed positions (147/768) show overlap increases,
decomposing as incidental rank flips 6.9% (53), distribution flattening 8.1%
(62), and alignment-like 4.2% (32) — against 21.5% decreases and 59.4% within
noise — all under a degrading target (PPL +3.68, flip rate 17%), i.e.,
degradation-phenomena rather than benign alignment.
**Q3 (draft-only):** feature-side damage concentrates in the projections;
acceptance-by-depth separates initialization damage (first) from accumulation
(recurrent); average acceptance drops to 1.27 (full draft W4A4); verification
cost per token rises ∝ 1/acceptance (2.7× more target calls at 1.27 vs 3.42).
**Q4 (both):** errors are weakly aligned (overlap gain +0.004; e2e interaction
+0.31 sub-additive) — correlated quantization does NOT rescue acceptance
(Q11 1.169 < Q01 1.268 < Q10 3.008).
**Q5:** increased-acceptance positions under target quantization reflect a less
accurate target (flattening + rank flips with PPL/margin degradation), not a
"more permissive but equally good" target. We do not use anthropomorphic
language; the operational split is in §13.

## Caveats / honesty notes

- Draft-quant results here are fake-quant (SpinQuant quantizers) with fp16
  arithmetic; the REAL packed path is validated at the kernel level (Gate B)
  but the full e2e matrix used fake quant. Real-INT4 target e2e was measured in
  the prior study (`runs/rotstudy_realint4_*` per-arm summary: real QuaRot W4A4
  target — A 4.177 / B2 4.180; the two-path split matches the explicit bridge
  under the real kernel as well).
- Under fake-quant targets, ea-vs-naive token equality degrades (dynamic
  per-token activation quantization is batch-shape sensitive between tree and
  incremental forwards); at fp16 the equality is exact in the Gate-A setting
  (8 prompts × 48 tok, 1.0 for every correct config) and 19/20 in the longer
  matrix run (one late low-margin flip at 64 tok). Acceptance length is the
  primary metric throughout.
- Greedy only; stochastic multi-seed evaluation not run in this pass.
- n=20 MT-bench prompts. All non-control effects are decided "meaningful
  decrease" at the pre-registered ε=0.05; magnitudes range from |Δ|=0.099
  (Q11 vs Q01; CI hi −0.066, flips to inconclusive at ε=0.10) and
  |Δ|=0.183 (AR-only, dz=0.83) up to |Δ|=2.25 (Q11 vs stock).
- Acceptance numbers are ea_generate deltas and include the +1 target
  (bonus) token per round — the standard EAGLE τ convention; the α_d
  estimator strips it (accepted draft tokens = delta−1). Cycle-level α_d
  pools cycles across prompts and carries no CI; cycles-reaching counts are
  reported alongside.
