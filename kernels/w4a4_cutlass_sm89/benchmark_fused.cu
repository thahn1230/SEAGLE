// fused-vs-unfused correctness + latency (GPU-visible device 0 = the
// CUDA_VISIBLE_DEVICES-selected physical GPU).
#include "w4a4_sm89.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 26;
  int N = argc > 2 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  int iters = argc > 4 ? atoi(argv[4]) : 500;
  int use_bias = argc > 5 ? atoi(argv[5]) : 1;
  srand(11);
  std::vector<uint8_t> ha((size_t)M * K / 2), hw((size_t)N * K / 2);
  for (auto& b : ha) b = rand() & 0xff;
  for (auto& b : hw) b = rand() & 0xff;
  std::vector<float> hsa(M), hsw(N);
  std::vector<__half> hb(N);
  for (auto& v : hsa) v = 0.001f + 0.01f * (rand() % 100) / 100.f;
  for (auto& v : hsw) v = 0.001f + 0.01f * (rand() % 100) / 100.f;
  for (auto& v : hb) v = __float2half(0.1f * ((rand() % 200) - 100) / 100.f);
  uint8_t *da, *dw; int32_t* dc; float *dsa, *dsw;
  __half *dbias, *dy_ref, *dy_fused;
  cudaMalloc(&da, ha.size()); cudaMalloc(&dw, hw.size());
  cudaMalloc(&dc, (size_t)M * N * 4);
  cudaMalloc(&dsa, M * 4); cudaMalloc(&dsw, N * 4);
  cudaMalloc(&dbias, N * 2);
  cudaMalloc(&dy_ref, (size_t)M * N * 2);
  cudaMalloc(&dy_fused, (size_t)M * N * 2);
  cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsa, hsa.data(), M * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsw, hsw.data(), N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, hb.data(), N * 2, cudaMemcpyHostToDevice);
  const __half* bias = use_bias ? dbias : nullptr;

  // reference: unfused
  w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0);
  w4a4_sm89::dequant_s32_to_fp16(dc, dsa, dsw, bias, dy_ref, M, N, 0);
  auto st = w4a4_sm89::gemm_s4s4_fp16_fused(
      da, dw, dsa, dsw, bias, dy_fused, M, N, K, nullptr, 0, 0);
  cudaDeviceSynchronize();
  if (st != w4a4_sm89::Status::kSuccess) { printf("fused launch fail\n"); return 1; }
  std::vector<__half> yr((size_t)M * N), yf((size_t)M * N);
  cudaMemcpy(yr.data(), dy_ref, yr.size() * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(yf.data(), dy_fused, yf.size() * 2, cudaMemcpyDeviceToHost);
  double maxe = 0, sume = 0, l2n = 0, l2d = 0, dot = 0, na = 0, nb = 0;
  long uneq = 0;
  for (size_t i = 0; i < yr.size(); i++) {
    double a = __half2float(yr[i]), b = __half2float(yf[i]);
    double d = fabs(a - b);
    maxe = fmax(maxe, d); sume += d;
    l2n += d * d; l2d += a * a;
    dot += a * b; na += a * a; nb += b * b;
    if (yr[i] != yf[i]) uneq++;
  }
  printf("CORRECTNESS M=%d N=%d K=%d bias=%d: max_abs=%.3e "
         "mean_abs=%.3e relL2=%.3e cos=%.9f unequal_fp16=%ld/%zu\n",
         M, N, K, use_bias, maxe, sume / yr.size(),
         sqrt(l2n / fmax(l2d, 1e-30)), dot / sqrt(na * nb), uneq,
         yr.size());
  cudaEvent_t e0, e1; float ms;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
#define TIME(body, label) \
  for (int i = 0; i < 100; i++) { body; } \
  cudaEventRecord(e0); \
  for (int i = 0; i < iters; i++) { body; } \
  cudaEventRecord(e1); cudaEventSynchronize(e1); \
  cudaEventElapsedTime(&ms, e0, e1); \
  printf("%s: %.5f ms\n", label, ms / iters);
  TIME(w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0),
       "GEMM_only");
  TIME(w4a4_sm89::dequant_s32_to_fp16(dc, dsa, dsw, bias, dy_ref, M, N, 0),
       "dequant_only");
  TIME({ w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0);
         w4a4_sm89::dequant_s32_to_fp16(dc, dsa, dsw, bias, dy_ref, M, N, 0); },
       "unfused_total");
  TIME(w4a4_sm89::gemm_s4s4_fp16_fused(da, dw, dsa, dsw, bias, dy_fused,
                                       M, N, K, nullptr, 0, 0),
       "fused_total");
  return 0;
}
