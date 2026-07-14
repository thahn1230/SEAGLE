# CONCAT-SELECTIVE SPINQUANT AUDIT

## Rotation type — this study is a RANDOM-ROTATION CONTROL

The rotations in every configuration of this study are **random**, not learned
SpinQuant rotations:

- **R1**: random Hadamard, `outputs/rotations/random_hadamard/R.bin` seed 0
  (dense-materialized [4096,4096]).
- **R2** (only where the draft AR head is quantized with `ar_r2r4=True`, and in
  the target's standard 'full' build): random-QR per-head [128,128].
- **R3**: N/A in all main configs (k_bits=16 → SpinQuant adds R3 only for KV
  quant; the earlier TQ4/W4A4KV4 target group is the only place R3 was active).
- **R4**: structured exact Hadamard folded into down_proj + online
  `matmul_hadU(_cuda)` on the MLP intermediate (target 'full' build; draft when
  `ar_r2r4=True`).

Learned-rotation reference exists ONLY as target WikiText-2 PPL
(`runs/spinquant_ppl`): W4A4 learned-R 10.436 vs random-R 10.627. No
learned-rotation EAGLE acceptance runs were performed → all conclusions here
are labeled `random_rotation_control`; a `learned_spinquant_rotation` group
remains future work and its conclusions must not be mixed with these.

## Placement table (target, standard 'full' build via SpinQuant fusion code)

| transform | placement | offline/online | affects |
|---|---|---|---|
| R1 | embeddings·R1; all block in/out weights conjugated; head W_lm·D_γ·R1 | offline fold | W+A quant basis |
| R2 | v_proj/o_proj per-head conjugation | offline fold | W+A |
| R3 | online Q/K Hadamard post-RoPE | online (fast had) | KV quant (only if k<16) |
| R4 | down_proj had-fold + online matmul_hadU on intermediate | offline+online | A |

## Draft (concat-selective architecture)

| transform | placement | offline/online |
|---|---|---|
| R1 | decoder conjugation (q/k/v in-fold, o out-fold, gate/up γ_l-fused in-fold, down out-fold); projections `W_h·D_γ·R1` / `W_h·R1`; head `W_lm·R1` | offline |
| post_projection_R1 | explicit y→y·R1 after fc | ONLINE, dense fp32 GEMM |
| R2/R4 | only in `ar_r2r4=True` quant configs (v/o conjugation + down had-fold/online) | offline+online |
| embedding | NONE (original basis, fp16) | — |

## Quantization (labels used everywhere)

- `fake_W4A4/W4A16/W8A8`: SpinQuant quantizers — weights per-channel symmetric
  RTN+MSE-clip, activations per-token asymmetric; fp16 arithmetic.
- `real_packed_W4A4`: QuaRot CUTLASS INT4×INT4 (per-channel sym absmax/7
  weights, per-token sym int4 dynamic activations, INT32 accum), KV fp16.
- `real_packed_W4A16`: torch tinygemm, group=128 asymmetric, bf16 activations.
- KV cache: fp16 in every config of THIS study (no config is called KV4).
