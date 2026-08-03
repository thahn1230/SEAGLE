// PyTorch extension: seagle_int4_ext
// quantize (per-row s4 + fp32 scales), unfused (gemm+dequant), fused.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "w4a4_sm89.h"

#define CHK(x, msg) TORCH_CHECK(x, msg)

std::vector<torch::Tensor> quantize(torch::Tensor a) {
  CHK(a.is_cuda() && a.dtype() == torch::kFloat16 && a.dim() == 2 &&
      a.is_contiguous(), "a: fp16 2D contiguous cuda");
  int M = a.size(0), K = a.size(1);
  CHK(K % 64 == 0, "K%64");
  auto q = torch::empty({M, K / 2},
                        a.options().dtype(torch::kUInt8));
  auto s = torch::empty({M}, a.options().dtype(torch::kFloat32));
  auto st = w4a4_sm89::quantize_fp16_per_row_s4(
      reinterpret_cast<const __half*>(a.data_ptr()),
      q.data_ptr<uint8_t>(), s.data_ptr<float>(), M, K,
      at::cuda::getCurrentCUDAStream());
  CHK(st == cudaSuccess, "quant launch");
  return {q, s};
}

torch::Tensor gemm_unfused(torch::Tensor qa, torch::Tensor qw,
                           torch::Tensor sa, torch::Tensor sw,
                           c10::optional<torch::Tensor> bias) {
  int M = qa.size(0), N = qw.size(0), K = qa.size(1) * 2;
  auto c32 = torch::empty({M, N},
                          qa.options().dtype(torch::kInt32));
  auto y = torch::empty({M, N},
                        qa.options().dtype(torch::kFloat16));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto st = w4a4_sm89::gemm_s4s4s32(
      qa.data_ptr<uint8_t>(), qw.data_ptr<uint8_t>(),
      c32.data_ptr<int32_t>(), M, N, K, nullptr, 0, stream);
  CHK(st == w4a4_sm89::Status::kSuccess, "gemm");
  auto err = w4a4_sm89::dequant_s32_to_fp16(
      c32.data_ptr<int32_t>(), sa.data_ptr<float>(),
      sw.data_ptr<float>(),
      bias ? reinterpret_cast<const __half*>(bias->data_ptr())
           : nullptr,
      reinterpret_cast<__half*>(y.data_ptr()), M, N, stream);
  CHK(err == cudaSuccess, "dequant");
  return y;
}

torch::Tensor gemm_fused(torch::Tensor qa, torch::Tensor qw,
                         torch::Tensor sa, torch::Tensor sw,
                         c10::optional<torch::Tensor> bias) {
  int M = qa.size(0), N = qw.size(0), K = qa.size(1) * 2;
  auto y = torch::empty({M, N},
                        qa.options().dtype(torch::kFloat16));
  auto st = w4a4_sm89::gemm_s4s4_fp16_fused(
      qa.data_ptr<uint8_t>(), qw.data_ptr<uint8_t>(),
      sa.data_ptr<float>(), sw.data_ptr<float>(),
      bias ? reinterpret_cast<const __half*>(bias->data_ptr())
           : nullptr,
      reinterpret_cast<__half*>(y.data_ptr()), M, N, K, nullptr, 0,
      at::cuda::getCurrentCUDAStream());
  CHK(st == w4a4_sm89::Status::kSuccess, "fused gemm");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
  m.def("gemm_unfused", &gemm_unfused);
  m.def("gemm_fused", &gemm_fused);
}
