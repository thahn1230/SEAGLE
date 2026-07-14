# B2 EXPERIMENT PROTOCOL (pre-registered before inspecting matrix results)

## Hardware / execution
- Physical GPUs 6,7 only (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=6,7`);
  every process asserts the mask and `torch.cuda.device_count()==2`; single-GPU
  runs pin cuda:0 or cuda:1; groups sharded across the two GPUs, one process each.
- Models: Llama-2-7b-chat-hf + yuhuili/EAGLE-llama2-chat-7B (EAGLE v1 @4a9cf3a),
  llama-2 chat template, mc_sim_7b_63 tree, greedy (T=0), fixed prompts.

## Decoding / datasets
- Primary: MT-bench turn-1 prompts, n=20, max_new_tokens=64 (matrix);
  n=8 × 48 for the fp16 gate. Same tokenizer/template/tree/token budget everywhere.
- Metric: mean accepted length per verification round (deltas of ea_generate),
  chain-style acceptance-by-depth alpha_d = P(accepted≥d | accepted≥d-1)
  computed from per-cycle accepted counts (delta−1; the +1 bonus token excluded).
- Target quality: WikiText-2 PPL (SpinQuant harness, RTN, random-Hadamard R):
  FP16 6.945 / W4A16 8.938 / W4A4 10.627 / W4A4KV4 10.939 (runs/spinquant_ppl).

## Quantization labels
- fake_W4A4 / fake_W4A16 / fake_W8A8: SpinQuant quantizers (weights per-channel
  sym RTN+MSE-clip; activations per-token asym), fp16 arithmetic.
- real_packed_W4A4: QuaRot CUTLASS INT4×INT4 (weights per-channel sym absmax/7,
  activations per-token sym, INT32 accumulate), KV fp16.
- real_packed_W4A16: torch tinygemm `aten._weight_int4pack_mm`, group=128 asym.
- Nothing is called W4A4KV4 unless k_bits=v_bits=4 (target group quant_kv4 only).

## Statistics (pre-registered)
- Paired prompt-level analysis; paired bootstrap over prompts, 10,000 resamples,
  95% percentile CI. Repeated greedy executions are NOT independent samples;
  uncertainty comes from prompt-level resampling only.
- Practical-equivalence margin: eps = max(0.05 accepted tokens/round,
  1% of baseline mean). Sensitivity: 0.02 / 0.05 / 0.10.
- Decision rule: CI entirely above +eps → meaningful increase; below −eps →
  meaningful decrease; inside ±eps → practically equivalent; else inconclusive.
- Negative-control criterion (fixed at Gate A, before any quantized run):
  paired mean drop > 0.1 AND ≥75% prompts degraded.

## Configurations
FP gate (n=8): FP00 stock; FP01 A-explicit; FP01b A-folded; FP02 B2-split;
FP02x explicit-Mγ (positive control); FP03 single-folded, FP04 gamma-omitted,
N1 naive, N2 elementwise-γ, N4 wrong-orientation, N5 fold-both-halves (negative).
Matrix (n=20): Q00 stock; Q10/Q10s/TQ1/TQ4 target-only; FP02 rotation control;
DQ_first_only_W4A4 / DQ_recurrent_only_W4A4 / DQ_both_proj_W4A4 / DQ_ar_only_W4A4 /
Q01 full-draft / DQ_first_only_W4A16 / DQ_recurrent_only_W4A16 draft-only;
Q11 both. Draft embed/head remain fp16 isolated copies in all draft-quant runs.

## Interpretation guardrails (anti-overselling)
Higher acceptance ≠ better model ≠ faster ≠ better quantization. Any acceptance
increase under target quantization must be cross-examined against PPL/top-1
agreement (degradation-induced alignment vs benign alignment) before being
described as an improvement. Tree acceptance is NOT interpreted as independent
chain proposals; alpha_d is a chain-style estimate along accepted paths.
