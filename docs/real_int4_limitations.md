# Real INT4 — exact limitations of what this project measures

Companion to docs/real_int4_kernel_audit.md. Everything below is a hard
constraint of the runs in `runs/rotstudy_realint4_*`; any wording stronger
than this in a paper draft is an overclaim.

## What IS real

- **W4A16 (tinygemm arm)**: target per-layer linear weights stored packed
  4-bit (`aten._convert_weight_to_int4pack`), consumed directly by
  `aten._weight_int4pack_mm`. Activations are 16-bit (bf16 inside the
  kernel, fp16 in the surrounding pipeline). This is weight-only INT4:
  call it "real W4A16" or "real W4 weight-only", NEVER "real W4A4".
- **W4A4 linears (QuaRot arm, if its rows carry int4_dispatch_proven=true)**:
  per-layer linear weights packed 4-bit AND activations per-token
  symmetric-quantized to packed 4-bit on the fly; the GEMM consumes two
  4-bit operands (CUTLASS INT4xINT4 -> INT32). This is real W4A4 FOR THE
  7 PER-LAYER LINEARS ONLY.

## What is NOT real, in every configuration

- **KV cache**: fp16 always. QuaRot's real KV4 lives in its flashinfer
  attention fork, which is not portable to EAGLE's vendored KV-cache
  attention (tree masks, custom cache layout) without rewriting that
  attention. Any KV4 number in this project is fake-quant simulation.
- **lm_head and embeddings**: 16-bit always (matches SpinQuant convention).
- **Attention math** (QK^T, softmax, PV): fp16 always.
- **The draft**: fp16 always; the interface variants (naive/A/B2) transform
  only the target->draft hand-off.

## Comparability warnings

- Real-arm quant recipes differ from the fake-quant study's SpinQuant RTN
  recipe (tinygemm: groupwise-128 asymmetric weights; QuaRot: per-channel
  symmetric weights + per-token symmetric activations). Acceptance numbers
  from real arms must not be inserted into the fake-quant tables.
- The fake-quant "relative_speedup_same_runtime" ratios and the real-arm
  tokens/s are different universes; never quote a fake ratio next to a real
  tokens/s as if they compose.
- Rotation in real arms is R1+R2 fused (+ R4 folded for the QuaRot arm);
  no R3 (KV stays fp16, so R3 has no role).

## Wall-clock interpretation limits

- Batch=1 decode on RTX 4090; no continuous batching, no CUDA graphs, no
  fused attention kernels. Absolute tokens/s is NOT a deployment number for
  any serving stack; only within-run comparisons (vanilla vs EAGLE vs A vs
  B2 on the SAME build) are meaningful.
- The QuaRot per-linear chain is quantize -> pack-GEMM -> dequant (3+ kernel
  launches vs 1 fp16 GEMM); at M=1 launch overhead can eat the INT4 gain.
  If the W4A4 arm is slower than fp16 end-to-end, that is a property of
  THIS integration (unfused, vendored model), not of QuaRot's own e2e stack.
- A-vs-B2: the unrotation GEMM removed by B2 is a 4096x4096 fp16 GEMM once
  per draft entry (tens of us); rounds are tens of ms. B2's dispatch
  overhead (Python branch + .data pointer swap) is measured and reported
  per prompt (b2_swap_overhead_ms_total). "B2 is faster" may legitimately
  come out as "no measurable difference at 7B batch-1" — that result is
  reported as-is.
