// W4A4 (s4 x s4 -> s32) Tensor Core GEMM for SM89 (RTX 4090).
// Instruction: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
// A: packed signed-int4 [M,K] row-major (two nibbles per byte,
//    element 2i in the low nibble). W: packed [N,K] row-major,
//    consumed untransposed as CUTLASS ColumnMajor B [K,N].
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace w4a4_sm89 {

enum class Status { kSuccess = 0, kErrorShape, kErrorInternal };

// bytes of workspace required by the GEMM (0 for the default config)
size_t gemm_workspace_bytes(int M, int N, int K);

// C[m,n] = sum_k A4[m,k] * W4[n,k]  (int32 accumulate)
Status gemm_s4s4s32(const uint8_t* a4_packed, const uint8_t* w4_packed,
                    int32_t* c32, int M, int N, int K,
                    void* workspace, size_t workspace_bytes,
                    cudaStream_t stream);

// dynamic per-row symmetric activation quantization:
//   scale_a[m] = max|A[m,:]| / 7;  q = clamp(round(A/s), -7, 7)
cudaError_t quantize_fp16_per_row_s4(const __half* a_fp16,
                                     uint8_t* a4_packed,
                                     float* row_scales,
                                     int M, int K,
                                     cudaStream_t stream);

// Y[m,n] = C32[m,n] * s_a[m] * s_w[n] + bias[n]
cudaError_t dequant_s32_to_fp16(const int32_t* c32,
                                const float* activation_row_scales,
                                const float* weight_col_scales,
                                const __half* bias,   // nullable
                                __half* out,
                                int M, int N,
                                cudaStream_t stream);

}  // namespace w4a4_sm89
