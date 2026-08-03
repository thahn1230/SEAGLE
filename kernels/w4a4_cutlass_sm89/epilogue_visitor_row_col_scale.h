// Epilogue visitor: D[m,n] = fp16( acc_i32 * s_a[m] * s_w[n] + b[n] )
// Modeled on cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h
// (CUTLASS 3.4.1, 2.x visitor API): int32 accumulator -> fp32 scale
// math -> fp16 store, all inside the GEMM kernel's epilogue.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/memory.h"

namespace w4a4_sm89 {

template <typename ThreadblockShape_, int ThreadCount,
          typename OutputTileIterator_, typename ElementAccumulator_,
          typename ElementCompute_, typename ElementOutput_>
class EpilogueVisitorRowColScale {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;
  using OutputTileIterator = OutputTileIterator_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementOutput = ElementOutput_;

  static int const kElementsPerAccess =
      OutputTileIterator::kElementsPerAccess;
  static int const kIterations = OutputTileIterator::kIterations;
  using AccumulatorFragment =
      cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment =
      cutlass::Array<ElementCompute, kElementsPerAccess>;
  using OutputVector =
      cutlass::Array<ElementOutput, kElementsPerAccess>;

  struct SharedStorage {};

  struct Arguments {
    float const* scale_row;        // [M]
    float const* scale_col;        // [N]
    cutlass::half_t const* bias;   // [N] or nullptr
    ElementOutput* ptr_D;
    int64_t ldd;

    Arguments() : scale_row(nullptr), scale_col(nullptr),
                  bias(nullptr), ptr_D(nullptr), ldd(0) {}
    Arguments(float const* sr, float const* sc,
              cutlass::half_t const* b, ElementOutput* d, int64_t ld)
        : scale_row(sr), scale_col(sc), bias(b), ptr_D(d), ldd(ld) {}
  };

  struct Params {
    typename OutputTileIterator::Params params_D;
    float const* scale_row;
    float const* scale_col;
    cutlass::half_t const* bias;
    ElementOutput* ptr_D;

    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : params_D(args.ldd),
          scale_row(args.scale_row),
          scale_col(args.scale_col),
          bias(args.bias),
          ptr_D(args.ptr_D) {}
  };

 private:
  Params const& params_;
  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_D_;
  cutlass::MatrixCoord extent_;
  cutlass::MatrixCoord thread_offset_;

 public:
  CUTLASS_DEVICE
  EpilogueVisitorRowColScale(Params const& params,
                             SharedStorage& shared_storage,
                             cutlass::MatrixCoord const& problem_size,
                             int thread_idx, int warp_idx,
                             int lane_idx,
                             cutlass::MatrixCoord const&
                                 threadblock_offset =
                                     cutlass::MatrixCoord(0, 0))
      : params_(params),
        iterator_D_(params.params_D, params.ptr_D, problem_size,
                    thread_idx, threadblock_offset),
        extent_(problem_size) {}

  CUTLASS_DEVICE void set_k_partition(int, int) {}

  CUTLASS_DEVICE
  void set_batch_index(int) {}

  CUTLASS_DEVICE
  void begin_epilogue() {}

  CUTLASS_DEVICE
  void begin_step(int) { fragment_D_.clear(); }

  CUTLASS_DEVICE
  void begin_row(int) {}

  CUTLASS_DEVICE
  void visit(int iter_idx, int row_idx, int column_idx, int frag_idx,
             AccumulatorFragment const& accum) {
    thread_offset_ =
        iterator_D_.thread_start() +
        OutputTileIterator::ThreadMap::iteration_offset(frag_idx);
    int row = thread_offset_.row();
    int col = thread_offset_.column();
    bool row_guard = row < extent_.row();
    float sr = row_guard ? params_.scale_row[row] : 0.f;

    cutlass::NumericArrayConverter<ElementCompute,
                                   ElementAccumulator,
                                   kElementsPerAccess>
        acc_conv;
    ComputeFragment v = acc_conv(accum);
    OutputVector out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      int c = col + i;
      bool guard = row_guard && (c < extent_.column());
      float y = 0.f;
      if (guard) {
        y = v[i] * sr * params_.scale_col[c];
        if (params_.bias) {
          y += static_cast<float>(params_.bias[c]);
        }
      }
      out[i] = static_cast<ElementOutput>(y);
    }
    reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx] = out;
  }

  CUTLASS_DEVICE
  void end_row(int) {}

  CUTLASS_DEVICE
  void end_step(int) {
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  CUTLASS_DEVICE
  void end_epilogue() {}
};

}  // namespace w4a4_sm89
