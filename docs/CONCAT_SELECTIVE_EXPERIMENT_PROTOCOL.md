# CONCAT-SELECTIVE EXPERIMENT PROTOCOL (pre-registered)

Hardware/execution: physical GPUs 6,7 only (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=6,7`, asserted; cuda:0=phys6, cuda:1=phys7). Models:
Llama-2-7b-chat + EAGLE-llama2-chat-7B (v1 @4a9cf3a). Greedy, MT-bench turn-1,
mc_sim_7b_63 tree, fixed token budgets. Rotations: random Hadamard
(random_rotation_control — NOT learned SpinQuant; see SPINQUANT_AUDIT).

## Gates
- GATE A (fp16, n=8×48): stock==rotated-target greedy 8/8; F1(prev B2) = F2
  (explicit) = F3 (folded) all output-preserving (exact_match_rate 1.0) and
  within 0.15 acceptance of each other; negative controls (F4 embedding-rotated,
  F5 orig-PL-recurrent, no_output_R, R_before_PL, first↔recurrent swaps) must
  degrade by the paired criterion (drop>0.1 AND ≥75% prompts).
- Verifier-consistency stage (§17): full-sequence vs chunked-KV vs
  token-incremental target logits on identical sequences (EAGLE's own KVCache),
  fp16 and fake-W4A4 targets; report rel-L2 / top-1 / margin-certificate
  fraction. Acceptance under quantized targets is interpreted WITH this
  execution-shape context (fp16 must be top-1 consistent; fake-quant targets
  are expected to show shape sensitivity — reported, not hidden).
- GATE B (real kernels): target linears + BOTH pre-R projections + 7 AR linears
  dispatch through QuaRot CUTLASS INT4×INT4 inside actual ea_generate; per-
  module forward counters + module types as proof; KV fp16.

## Matrix (n=20 × 64 tok, greedy; paired bootstrap 10k, 95% CI)
- Q00 stock; Q10 target-fake-W4A4 + fp16 draft (A-explicit).
- Q01_new / Q01_prevB2: draft full fake-W4A4 under the two architectures on the
  SAME rotated-fp16 target — the architecture-robustness comparison.
- Q11_new / Q11_prevB2: both quantized.
- Ablations (rot target): first-only / recurrent-only / both-proj / AR-only
  (fake W4A4); first/recurrent W4A16; embedding-table-only fake W4A16
  (embedding is fp16 + original-basis in ALL primary configs).
- REAL e2e: REAL_Q10, REAL_Q11 (concat-selective, R1-only decoder), fake
  counterpart on identical prompts.

## Statistics (identical to the B2 study, pre-registered)
Paired prompt-level bootstrap (10k, 95% percentile CI); ε = max(0.05, 1% of
baseline); decisions increase/decrease/equivalent/inconclusive; sensitivity
0.02/0.05/0.10. Depth alphas: chain-style α_d from per-cycle accepted counts
(delta−1), cycles-reaching reported; acceptance numbers are ea_generate deltas
(include the +1 bonus token — EAGLE τ convention).

## Anti-overselling
Higher acceptance ≠ quality ≠ speed. Architecture-robustness claims must come
from the SAME quant policy on the SAME target with paired CIs. Real-vs-fake
kernel differences are reported as recipe differences (sym absmax/7 vs
RTN+MSE-clip asym), not as anomalies.
