# Real INT4 kernel audit — what exists, what builds, what integrates

Date: 2026-07-04. Environment: 8x RTX 4090 (sm_89, Ada — INT4 tensor-core mma
IS supported on 8.x, removed on sm_90), torch 2.6.0+cu124, transformers 4.51.3,
nvcc 12.8 at /usr/local/cuda-12.8 (system default nvcc is 11.5 — every build
must export CUDA_HOME/PATH), cmake 3.22.1, fast_hadamard_transform 1.1.0
(built from source earlier in this project).

## 1. Backends found in this environment

| Backend | Present? | Class | Notes |
|---|---|---|---|
| torch native tinygemm (`aten._weight_int4pack_mm` + `aten._convert_weight_to_int4pack`) | YES (ships in torch 2.6, verified by `hasattr`) | **real W4A16** (packed int4 weights, 16-bit activations) | groupwise asymmetric int4 weights; the gpt-fast/torchao kernel; no build needed |
| QuaRot (spcl/QuaRot @ 5008669) | cloned to /data/thahn1230/quarot; **build attempted this session** (cutlass submodule 114M, sm_89 gencode added by local patch, fast-hadamard reinstall skipped) | **real W4A4** (packed int4 weights AND packed int4 per-token activations; `quarot.matmul` = CUTLASS INT4xINT4 -> INT32, `sym_quant`/`sym_dequant` CUDA kernels) | see build log /data/thahn1230/quarot/build.log; result recorded in §4 |
| SpinQuant (third_party/SpinQuant) | present | fake quant only | grep confirms zero CUDA kernels; QDQ + FP16 matmuls |
| torchao / AutoAWQ / GPTQModel / Marlin / bitsandbytes / vLLM | NOT installed | (various) | possible via pip but redundant: tinygemm already covers real W4A16 with zero build risk; not pursued |
| torch `_int_mm` | YES | INT8 GEMM | not 4-bit; not used |

## 2. Capability matrix (for the two candidates)

| Capability | tinygemm (torch native) | QuaRot kernels |
|---|---|---|
| W4A16 | YES (by design) | possible but pointless (their path targets A4) |
| W4A4 | NO (activations stay 16-bit) | YES: `Quantizer` (per-token absmax/7 fp16 scale) -> `sym_quant` pack -> INT4 GEMM -> `sym_dequant` |
| KV4 | NO | only via their flashinfer attention path (e2e fork of HF Llama); NOT portable to EAGLE's vendored KV-cache attention without rewriting it — **out of scope, KV stays 16-bit here** |
| batch=1 decode (M=1) | YES (kernel is decode-oriented) | YES (M padded internally; CUTLASS tile) |
| prefill (large M) | YES | YES |
| hidden 4096 / inter 11008 | K must divide by group (128): 4096 ok, 11008 ok (86 groups); N mult of 8: ok | K mult of 32 after packing: 4096, 11008 ok |
| activation dtype | bf16 native (fp16 support checked empirically in validation) | fp16 scales/dequant |
| integration into EAGLE tree decode | straightforward: swap `nn.Linear` -> wrapper module in the VENDORED target (attention math, KV cache, tree buffers untouched) | same surgery + a `Quantizer` (+ online Hadamard for down_proj/o_proj) in FRONT of each linear; attention/KV kept fp16 |

## 3. What counts as proof of INT4 dispatch (spec §3.2)

- tinygemm: torch profiler must show `aten::_weight_int4pack_mm` dispatching a
  kernel with `tinygemm` in its name (e.g. `tinygemm_m16n8k16_...`); weights
  stored as packed int32/uint8 tensor 8x smaller than fp16.
- QuaRot: profiler must show the CUTLASS INT4 GEMM kernel from
  `quarot/kernels/gemm.cu` plus `sym_quant`/`sym_dequant` kernels from quant.cu.
- Both recorded to `runs/rotstudy_realint4_*/profiler_kernel_names.txt` by
  `scripts/validate_real_int4_kernel.py`. A run with no such kernel names in
  its trace is NOT a real-INT4 run, whatever its config says.

## 4. Build result (filled after the attempt)

- torch tinygemm: nothing to build. AVAILABLE. Standalone validation
  (runs/rotstudy_realint4_20260704_1857/real_int4_kernel_validation.json):
  kernel-vs-QDQ-reference worst rel-L2 0.24% across all shapes/M (kernel
  correct); bf16 activations ONLY (fp16/fp32 raise RuntimeError on this
  torch build — the integration casts fp16<->bf16 at module boundaries);
  packing uses the uint8 [N, K/2] layout. Kernel name proof:
  `at::native::tinygemm_m16n8k16_chunk_kernel<..., BLayout_TC_int4<8,128>>`.
  Microbench vs fp16 F.linear: M=1 2.0-7.2x FASTER, M=26 0.40-0.76x
  (SLOWER), M>=512 ~0.11x — a decode-shape kernel; EAGLE's ~26-node tree
  forward sits in its slow regime.
- QuaRot: BUILD SUCCEEDED (patched setup.py: +sm_89 gencode, skip
  fast-hadamard editable reinstall; cutlass @ shallow submodule).
  Standalone validation: full quantize->INT4 GEMM->dequant chain vs QDQ
  reference worst rel-L2 1.7% (fp16 scale roundoff; cosine ~1); kernel name
  proof: CUTLASS `integer_subbyte<4>` MmaTensorOp GEMM +
  `sym_quantize_f16_i4_kernel` / `sym_dequantize_i32_f16_kernel`.
  Microbench (whole chain) vs fp16: 4096->11008 2.2-3.1x faster at all M;
  4096->4096 launch-bound at M=1 (0.31x) but 2-3x at M>=512.

## 5. Decision (spec Phase 5)

**Case A/B hybrid:**
- Real **W4A16** is available and integrable into the EAGLE runtime with low
  risk (tinygemm). This is the primary wall-clock arm: rotated target with
  packed-INT4 weights, A vs B2 draft interfaces. Labeled **real W4A16** —
  never "real W4A4".
- Real **W4A4** (QuaRot) is available at KERNEL level after a successful
  local build; standalone validation gates any e2e use. E2E integration into
  EAGLE's vendored target is attempted ONLY for the 7 per-layer linears
  (q/k/v/o/gate/up/down; per-token sym A4, per-channel sym W4, online
  Hadamard for down_proj input); lm_head, embeddings, KV cache, attention
  math stay 16-bit. If the e2e attempt is unstable, we report the standalone
  kernel numbers and document the blocker instead of shipping a fake.
- KV4: NOT real in any configuration here (no portable real KV4 kernel for
  the vendored attention); any KV4 numbers in this project remain fake-quant
  simulation and are labeled as such.

## 6. Known limitations of the plan

- QuaRot's activation scheme (per-token symmetric absmax/7, clip 1.0) differs
  from SpinQuant's fake-quant recipe (asymmetric per-token); real-W4A4
  acceptance is NOT directly comparable to the fake-W4A4 acceptance tables —
  it is its own arm with its own baselines.
- tinygemm W4A16 weight quant (groupwise asymmetric, g=128) also differs from
  the SpinQuant RTN recipe; same caveat.
- Batch=1 decode on RTX 4090 is bandwidth-bound; INT4 weights reduce weight
  traffic ~4x vs fp16, so vanilla decode should speed up materially, but
  EAGLE tree steps (batch ~26 nodes) shift toward compute — real speedups
  must be measured, not extrapolated from the fake-quant ratios.
- The A-vs-B2 wall-clock question is expected to be dominated by round time
  (tens of ms) vs the unrotation GEMM (tens of µs). We measure and report
  whatever the data shows, including "no measurable difference".
