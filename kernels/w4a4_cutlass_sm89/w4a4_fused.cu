// Fused s4s4s32 GEMM + (row-scale x col-scale + bias) fp16 epilogue.
// Mainloop identical to w4a4_sm89.cu (m16n8k64 IMMA); the visitor
// replaces the separate dequant kernel: one CUDA kernel, no [M,N]
// int32 intermediate in global memory.
#include "w4a4_sm89.h"
#include "epilogue_visitor_row_col_scale.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/gemm_universal.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_memory.h>
#include "gemm_with_epilogue_visitor.h"   // example-35 kernel wrapper

namespace w4a4_sm89 {

template <typename TbShape, typename WarpShape, int Stages>
struct FusedGemmConfig {
  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator, ElementCompute>;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA, cutlass::layout::RowMajor, 32,
      ElementB, cutlass::layout::ColumnMajor, 32,
      ElementC, cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TbShape, WarpShape, InstructionShape,
      EpilogueFunctorOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages, true,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
          ElementA, ElementB, ElementC,
          ElementAccumulator>::Operator,
      cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

  using Visitor = EpilogueVisitorRowColScale<
      TbShape, DefaultGemmKernel::kThreadCount,
      typename DefaultGemmKernel::Epilogue::OutputTileIterator,
      ElementAccumulator, ElementCompute, ElementC>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          Visitor, typename DefaultGemmKernel::Epilogue>::Epilogue;

  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      typename DefaultGemmKernel::Mma, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
};

using FusedDefault =
    FusedGemmConfig<cutlass::gemm::GemmShape<128, 128, 128>,
                    cutlass::gemm::GemmShape<64, 64, 128>, 3>;
using FusedSmallM =
    FusedGemmConfig<cutlass::gemm::GemmShape<64, 128, 128>,
                    cutlass::gemm::GemmShape<32, 64, 128>, 4>;

template <typename Config>
static Status launch_fused(const uint8_t* a, const uint8_t* w,
                           const float* sa, const float* sw,
                           const __half* bias, __half* out,
                           int M, int N, int K,
                           cudaStream_t stream) {
  using GemmKernel = typename Config::GemmKernel;
  using Visitor = typename Config::Visitor;

  typename Visitor::Arguments visitor_args(
      sa, sw, reinterpret_cast<cutlass::half_t const*>(bias),
      reinterpret_cast<cutlass::half_t*>(out), N);

  typename GemmKernel::TensorRefA ref_A(
      const_cast<cutlass::int4b_t*>(
          reinterpret_cast<cutlass::int4b_t const*>(a)),
      typename GemmKernel::LayoutA(K));
  typename GemmKernel::TensorRefB ref_B(
      const_cast<cutlass::int4b_t*>(
          reinterpret_cast<cutlass::int4b_t const*>(w)),
      typename GemmKernel::LayoutB(K));
  typename GemmKernel::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord(M, N, K),
      1, ref_A, ref_B,
      /*batch_stride_A=*/0, /*batch_stride_B=*/0,
      visitor_args);

  typename GemmKernel::Params params(args);
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> swizzle;
  dim3 grid = swizzle.get_grid_shape(params.grid_tiled_shape);
  dim3 block(GemmKernel::kThreadCount, 1, 1);
  int smem = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem >= (48 << 10)) {
    cudaError_t r = cudaFuncSetAttribute(
        cutlass::Kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (r != cudaSuccess) return Status::kErrorInternal;
  }
  cutlass::Kernel<GemmKernel><<<grid, block, smem, stream>>>(params);
  return cudaGetLastError() == cudaSuccess ? Status::kSuccess
                                           : Status::kErrorInternal;
}

Status gemm_s4s4_fp16_fused(const uint8_t* a4_packed,
                            const uint8_t* w4_packed,
                            const float* activation_scales,
                            const float* weight_scales,
                            const __half* bias, __half* output,
                            int M, int N, int K, void*, size_t,
                            cudaStream_t stream) {
  if (K % 64 != 0 || N % 8 != 0) return Status::kErrorShape;
  if (M <= 64)
    return launch_fused<FusedSmallM>(a4_packed, w4_packed,
                                     activation_scales, weight_scales,
                                     bias, output, M, N, K, stream);
  return launch_fused<FusedDefault>(a4_packed, w4_packed,
                                    activation_scales, weight_scales,
                                    bias, output, M, N, K, stream);
}

}  // namespace w4a4_sm89
