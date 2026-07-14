# CONCAT-SELECTIVE ROTATION STUDY — FINAL REPORT

Date 2026-07-14 · branch `exp/eagle1-concat-selective-rotation` (from B2 HEAD
9c59d94; previous implementation preserved as the comparison baseline) · GPUs
physical 6,7 only (asserted) · Llama-2-7b-chat + EAGLE-llama2-chat-7B (v1
@4a9cf3a) · rotations: **random Hadamard (random_rotation_control — NOT learned
SpinQuant)** · greedy, MT-bench, mc_sim_7b_63 · evidence:
`artifacts/concat_selective_rotation_study/`.

Answers to the 12 required questions (§22):

## 1. Does original-basis embedding + post-PL R1 preserve the original EAGLE function?
**YES — exactly.** fp16 e2e (n=8×48): F2 (explicit) = F3 (folded preR) =
**3.7538** accepted/round with **exact-match 1.0** on every prompt, identical
to F1 (previous B2) and within fp16 noise of stock F0 (3.7208 on the stock
target; rotated-target greedy == stock 8/8). Tiny-fp64 chain through the real
vendored decoder: rel-L2 < 1e-6 at every depth 1-5, top-1 agreement 1.0
(`fp64_algebra/tiny_chain.json`).

## 2. Is R1ᵀ + gamma applied only to the hidden slice on the first forward?
**YES.** Explicit path: slice ops touch only `z[..., D:]`
(`concat_selective_explicit`); folded path: `projection_first_preR =
[W_e | W_h·D_γ·R1]` with `W_e` bit-untouched (asserted,
`test_concat_selective_first_folded.py`). Zero-hidden probe: doubling gamma
changes nothing when the hidden slice is zero → no leakage into the e-slice.

## 3. Is the embedding slice numerically unchanged before the projection?
**YES.** The draft embedding is a **bit-exact copy of the untouched checkpoint**
(rel-L2 0.0, `CONCAT_SELECTIVE_EMBEDDING_AUDIT.md`); adapter `install()` asserts
the table checksum; the primary path applies NO transform to the e-slice
(structurally impossible: the only input op is the Linear itself). Negative
control F4 (e·R1 into untouched W_e) collapses to **1.509** (drop 2.25, 20/20
gate prompts degraded... n=8 gate: 100% prompts).

## 4. Does the recurrent projection absorb hidden-side R1ᵀ via W_h·R1?
**YES.** `projection_recurrent_preR = [W_e | W_h·R1]`; algebra
`r_d·(W_h R1)ᵀ = h_d·W_hᵀ` verified to fp64 tolerance; negative control F5
(original `W_h`, no absorption) degrades to **2.317** (drop 1.44, 100%).

## 5. Is post-projection R1 applied to the complete D-dim output?
**YES — explicit `PostProjectionR1` (dense fp32 GEMM), executed on every draft
forward** (post_R1_calls == calls_first + calls_recurrent in every dispatch
summary; e.g. 585 = 117 + 468). Omitting it (N_no_output_R) collapses to
**1.003**; applying R1 to the 2D input instead (N_R_before_PL) collapses to
**1.045**.

## 6. Does the new architecture match previous B2 in FP16?
**YES — token-identical.** F1 = F2 = F3 = 3.7538 with the same per-prompt
outputs (exact-match 1.0 against the same naive reference), same dispatch
counts (117 first / 468 recurrent).

## 7. Which architecture is more robust under W4A4?
**Practically equivalent** (n=20×64, paired bootstrap):
- draft-only full W4A4: concat-selective 1.2450 vs prev-B2 1.2684 —
  Δ = −0.023, CI [−0.056, +0.005] → inconclusive/equivalent-leaning.
- both-quantized (Q11): concat-selective 1.2140 vs prev-B2 1.1692 —
  Δ = **+0.045, CI [+0.020, +0.068]** — statistically above zero but BELOW the
  pre-registered ε = 0.05 → practically equivalent (tiny edge to the new
  architecture when both models are quantized).
No architecture rescues W4A4 draft quantization.

## 8. Did rotating the embedding (previous B2) increase quantization error?
**No meaningful effect.** The two architectures quantize essentially equally
(above). Direct embedding evidence: quantizing the ORIGINAL-basis embedding
table (fake W4A16) costs nothing — Δ = +0.006, CI [−0.049, +0.066] vs the fp16
baseline — and in the previous study the rotated-embedding architecture reached
the same collapse levels. The embedding (rotated or not) is NOT the
quantization bottleneck; the fc projections are.

## 9. Are real packed W4A4 results consistent with fake quantization?
**YES — measured e2e (STOP GATE B PASS)**: on the REAL-W4A4 target
(224 QuaRot CUTLASS linears): REAL_Q10 (fp16 draft) = **3.281**;
REAL_Q11 (draft real W4A4: both pre-R projections via bias-wrapped QuaRot +
7 AR linears) = **1.0955**; FAKE draft counterpart on identical prompts =
1.1734. Real vs fake draft quantization differ by only 0.08 accepted
tokens/round — the fake-quant conclusions (projection-driven collapse) carry
over to the real packed kernels; the residual gap reflects the recipe
difference (per-channel sym absmax/7 + per-token sym int4 vs RTN+MSE-clip +
per-token asym). Dispatch proven per module (forward counters + module types;
`real_w4a4_dispatch/real_e2e_dispatch_proof.csv`); KV fp16; kernel-level
match vs fp emulation rel 1.8e-2.

## 10. Does quantized tree verification preserve the target's output?
**fp16: YES** — verifier-consistency (EAGLE KVCache): full-sequence vs
chunked-KV vs token-incremental logits agree at **top-1 = 1.000** on all
evaluated positions (rel-L2 ≈ 0.0017, certificate fraction 0.97-1.0), and every
fp16 config is output-preserving (exact 1.0). **fake-W4A4 target: execution-
shape sensitivity is real and now quantified** — full-sequence vs chunked/
incremental KV paths agree only at **top-1 = 0.906-0.911** (certificate
fraction ||Δ||∞<margin/2 = 0.43, rel-L2 0.37, mean max|Δ| ≈ 6 logits;
`verifier_consistency/target_logit_consistency.csv`). ~9% of positions flip
top-1 purely from execution shape; consequently ea-vs-naive exact-match is
0.00 at 64 tokens under the quantized target. Per the protocol this is reported
as a verifier-semantics caveat: under a quantized target, acceptance measures
alignment with the tree-execution-shaped target distribution, not with a unique
target string.

## 11. Does target quantization increase or decrease acceptance (corrected architecture)?
**DECREASE**: Q10 (target fake-W4A4, fp16 draft) 3.0079 vs Q00 3.4158 —
Δ = −0.408, CI [−0.572, −0.255], meaningful decrease (identical under either
architecture; matches the B2 study).

## 12. Which component dominates draft acceptance degradation?
**The fc projection — and `projection_first_preR` specifically**: first-only
W4A4 −2.050 [CI −2.265,−1.844] vs recurrent-only −1.446 [−1.680,−1.226];
direct contrast first-vs-recurrent Δ = **−0.604, CI [−0.639, −0.569]**
(meaningful). AR-head-only: −0.101 [−0.181,−0.019] (inconclusive at ε=0.05 —
mild). Embedding: free (+0.006). Weight-only (W4A16) damage is depth-symmetric
(first −0.801 vs recurrent −0.803; direct contrast +0.002 [−0.168,+0.187]) —
the W4A4 asymmetry is driven by ACTIVATION quantization of the first-step
input, reproducing the B2-study mechanism under the new architecture.

## Depth signatures (cs_depth_alphas.csv)
- recurrent-only W4A4: α₁ high (first step intact) then collapse by depth —
  accumulating divergence;
- first-only W4A4: α₁ destroyed, deeper depths partially recover — pure
  initialization damage. (Same two signatures as the B2 study.)

## Caveats
- Random-rotation control study (NOT learned SpinQuant rotations); learned-R
  exists only as target PPL (10.44 vs 10.63 W4A4).
- Matrix draft-quant runs use fake quantization (SpinQuant quantizers); the
  real-kernel path is validated e2e in Gate B configs (REAL_Q10/REAL_Q11) with
  its own (symmetric absmax) recipe.
- Greedy only; n=20 MT-bench prompts; ea-vs-naive exact-match under fake-quant
  targets is limited by verifier execution-shape sensitivity (§10).
- Acceptance numbers are EAGLE τ (accepted draft tokens + 1 bonus per round).
