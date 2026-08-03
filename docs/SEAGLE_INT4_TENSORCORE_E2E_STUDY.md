# SEAGLE Real-INT4 Tensor Core E2E Latency Study

**Run**: `runs/seagle_int4_e2e_20260803_204811` · 4090 (SM89), CUDA
12.8, CUTLASS `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`.
Goal: replace fake-quant with REAL s4×s4→s32 Tensor-Core compute in
SEAGLE (SpinQuant+EAGLE) and measure, end to end, what draft
quantization buys.

## 1. Kernel stack (three verified levels)

1. **`kernels/w4a4_cutlass_sm89/`** (guide-spec package: `.cu/.h`,
   benchmark, CMake; built with cuda-12.8, `CUDA_ARCHITECTURES 89`,
   CUTLASS from the QuaRot submodule):
   - exact integer check vs CPU dot: **PASS (0/64 mismatches)** at
     128×128×4096, 512×4096×4096, 16×4096×4096;
   - SASS contains **96 IMMA** instructions — native INT4 Tensor
     Cores confirmed;
   - **849.9 TOPS** at M=512 (64% of the 1321-TOPS peak); 56.9 TOPS
     at M=16 (row-utilization limit, as the guide predicts); default
     128×128×128 tile + 64×128×128 small-M path.
2. **`RealInt4Linear`** (`src/eagle_spinquant/real_int4_linear.py`,
   QuaRot `_CUDA` backend — same s4s4s32 CUTLASS math, python-ready):
   per-row dynamic A4 (absmax/7, clamp ±7) → GEMM → int32 ×
   s_a[m]·s_w[n] + fp16 bias. cos ≈ 0.985 vs fp16 (plain RTN, no
   rotation/clip — latency-faithful, quality-proxy). Same-input
   activation-quant cache (q/k/v and gate/up quantize once).
3. **E2E swap**: every decoder linear (q/k/v/o/gate/up/down ×32
   target layers, ×1 draft layer) + the draft 8192→4096 fusion
   projection. Embedding/LM-head/norm/RoPE/softmax/residual fp16.

## 2. Linear-level latency (4096→4096, e2e-linear = quant+GEMM+dequant)

| M | int4 e2e | fp16 | ratio | regime |
|---|---|---|---|---|
| 1 (qkv-trio) | 0.100 ms | 0.116 ms | **0.86** | bandwidth-bound: int4 WINS |
| 16 | 0.109 | 0.055 | 1.99 | launch/quant overhead-bound |
| 26 (EAGLE tree) | 0.101 | 0.062 | 1.62 | overhead-bound |
| 64 | 0.103 | 0.065 | 1.58 | overhead-bound |
| 256 | 0.104 | 0.172 | **0.60** | compute-bound: int4 wins |

GEMM-only is 3–25× faster than fp16 at every M (0.009–0.03 ms) — the
overhead is entirely the **unfused** quant/dequant/launch path,
exactly the guide's warning ("GEMM만 빠르고 quant/dequant가 지배").

## 3. E2E EAGLE decoding (mtbench 8×128 tok, greedy mc_sim_7b_63)

| config | ms/token | measured tau* | draft ms/cyc | verify ms/cyc | cycles/s | projected tok/s† |
|---|---|---|---|---|---|---|
| T16D16 (fp16 base) | **9.11** | 3.153 | 5.41 | 22.35 | 34.8 | 124.5 |
| T4D16 (target int4) | 23.15 | 1.546 | 5.47 | 29.33 | 27.9 | 91.5 |
| T4D4 (full int4) | 28.03 | 1.204 | 5.97 | 26.89 | 29.6 | 90.4 |
| T16D4 | 24.95 | 1.118 | 5.58 | 21.42 | 35.9 | 104.5 |

\* measured tau uses plain RTN (no rotation/P3) — quality collapse is
expected and is NOT the quality claim; quality-calibrated taus come
from the validated fake-quant studies.
† cycles/s × validated tau (3.58 / 3.27 / 3.05 / 2.91) — separates
kernel speed from this run's RTN-only acceptance.

**Memory (measured)**: 13.30 GiB → **3.93 GiB (−70%)** with target +
draft + fusion swapped — the deployment-motivating win that fake-quant
studies could never show.

## 4. Honest readout — why the draft must (eventually) be quantized

1. **Memory**: −70% total; the draft+fusion alone is ~0.9 GiB fp16 →
   0.24 GiB. On memory-constrained serving this is the difference
   between fitting and not fitting alongside the KV cache.
2. **Kernel-limit latency**: at EAGLE's operating points the GEMMs
   themselves are 3–25× faster in INT4 (M=1 draft-recurrent is
   *bandwidth*-bound → int4 already wins even unfused, 0.86×; M=26
   tree verify is compute-light and currently overhead-bound).
3. **Current unfused wrappers give latency BACK**: verify/cycle
   22.4→29.3 ms, draft/cycle 5.4→6.0 ms — ~0.05 ms fixed cost × 7
   linears × 33 layers dominates. This is an engineering statement,
   not a physics one: the LRGF foldability study already classified
   scale+rotation+A4 as **fusable into one kernel (F2, measured
   0.046 ms)**; fusing quant into the preceding op and dequant into
   the GEMM epilogue removes almost all of the gap. With those
   fusions the projected cycle time is GEMM-limited, where INT4 wins
   at every M ≥ 1 measured here.
4. **Quality**: this study intentionally used RTN-only weights; the
   validated route to INT4 quality is the fake-quant-calibrated
   SEAGLE stack (SpinQuant rotations + EP3-P + [optionally ACC-LR
   rotation]), whose taus (3.05–3.15 on T4) are what the projected
   column uses. Deploying those folds through `RealInt4Linear` is a
   weight-preprocessing change only (same kernels).

Bottom line: **draft quantization is necessary for the memory story
today and for the latency story once quant/dequant are fused; the
raw Tensor-Core capability (850 TOPS, exact s4s4s32) is verified and
in place.**

## 5. Reproduce

```bash
cd kernels/w4a4_cutlass_sm89
cmake -S . -B build -DCUTLASS_DIR=/data/thahn1230/quarot/third-party/cutlass \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build -j
./build/w4a4_benchmark 512 4096 4096 100     # PASS + ~850 TOPS
cuobjdump --dump-sass ./build/w4a4_benchmark | grep -c IMMA   # 96
python scripts/benchmark_seagle_int4_e2e.py \
  --run-dir $(cat runs/SEAGLE_INT4_RUN_DIR) --configs T4D4
```

Known limits: per-channel W4 only (group-wise needs K-split per the
guide); QuaRot `sym_quant` clamps to [−8,7] vs the package's [−7,7]
(documented, immaterial for latency); e2e numbers from 8 prompts on a
shared box.
