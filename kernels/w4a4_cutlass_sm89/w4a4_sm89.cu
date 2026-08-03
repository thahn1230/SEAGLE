// W4A4 CUTLASS GEMM + quant/dequant kernels for SM89.
// See w4a4_sm89.h for the API contract.
#include "w4a4_sm89.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

namespace w4a4_sm89 {

// s4 x s4 -> s32, A [M,K] RowMajor, B(=W^T view) [K,N] ColumnMajor.
// A PyTorch-contiguous W [N,K] RowMajor buffer IS the [K,N]
// ColumnMajor operand — no transpose needed.
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,                    // compat operator tag (Ada)
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<
        int32_t, 128 / cutlass::sizeof_bits<int32_t>::value,
        int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// small-M specialization for the EAGLE draft recurrent path
using GemmSmallM = cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<
        int32_t, 128 / cutlass::sizeof_bits<int32_t>::value,
        int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4>;

size_t gemm_workspace_bytes(int, int, int) { return 0; }

template <typename G>
static Status run_gemm(const uint8_t* a, const uint8_t* w, int32_t* c,
                       int M, int N, int K, cudaStream_t stream) {
  typename G::Arguments args(
      {M, N, K},
      {reinterpret_cast<const cutlass::int4b_t*>(a), K},   // lda in
      {reinterpret_cast<const cutlass::int4b_t*>(w), K},   // int4 elems
      {c, N},
      {c, N},
      {1, 0});
  G op;
  if (op.can_implement(args) != cutlass::Status::kSuccess)
    return Status::kErrorShape;
  if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
    return Status::kErrorInternal;
  return op(stream) == cutlass::Status::kSuccess
             ? Status::kSuccess
             : Status::kErrorInternal;
}

Status gemm_s4s4s32(const uint8_t* a, const uint8_t* w, int32_t* c,
                    int M, int N, int K, void*, size_t,
                    cudaStream_t stream) {
  if (K % 64 != 0) return Status::kErrorShape;
  if (M <= 64) return run_gemm<GemmSmallM>(a, w, c, M, N, K, stream);
  return run_gemm<Gemm>(a, w, c, M, N, K, stream);
}

// ---------------- quant / dequant --------------------------------
__global__ void quant_kernel(const __half* __restrict__ a,
                             uint8_t* __restrict__ q,
                             float* __restrict__ scales,
                             int M, int K) {
  int m = blockIdx.x;
  if (m >= M) return;
  extern __shared__ float red[];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    mx = fmaxf(mx, fabsf(__half2float(a[(size_t)m * K + k])));
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  float scale = fmaxf(red[0], 1e-8f) / 7.f;
  if (threadIdx.x == 0) scales[m] = scale;
  float inv = 1.f / scale;
  for (int k2 = threadIdx.x * 2; k2 < K; k2 += blockDim.x * 2) {
    int q0 = __float2int_rn(__half2float(a[(size_t)m * K + k2]) * inv);
    int q1 = __float2int_rn(
        __half2float(a[(size_t)m * K + k2 + 1]) * inv);
    q0 = max(-7, min(7, q0));
    q1 = max(-7, min(7, q1));
    q[((size_t)m * K + k2) / 2] =
        (uint8_t)((q0 & 0x0f) | ((q1 & 0x0f) << 4));
  }
}

cudaError_t quantize_fp16_per_row_s4(const __half* a, uint8_t* q,
                                     float* scales, int M, int K,
                                     cudaStream_t stream) {
  int threads = 256;
  quant_kernel<<<M, threads, threads * sizeof(float), stream>>>(
      a, q, scales, M, K);
  return cudaGetLastError();
}

__global__ void dequant_kernel(const int32_t* __restrict__ c,
                               const float* __restrict__ sa,
                               const float* __restrict__ sw,
                               const __half* __restrict__ bias,
                               __half* __restrict__ y,
                               int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx % N;
  float v = (float)c[idx] * sa[m] * sw[n];
  if (bias) v += __half2float(bias[n]);
  y[idx] = __float2half(v);
}

cudaError_t dequant_s32_to_fp16(const int32_t* c, const float* sa,
                                const float* sw, const __half* bias,
                                __half* y, int M, int N,
                                cudaStream_t stream) {
  int total = M * N, threads = 256;
  dequant_kernel<<<(total + threads - 1) / threads, threads, 0,
                   stream>>>(c, sa, sw, bias, y, M, N);
  return cudaGetLastError();
}

}  // namespace w4a4_sm89
