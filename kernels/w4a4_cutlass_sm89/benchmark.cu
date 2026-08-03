// correctness (exact integer check vs CPU dot) + TOPS measurement
#include "w4a4_sm89.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static int8_t unpack(const std::vector<uint8_t>& p, size_t i) {
  uint8_t b = p[i / 2];
  int8_t v = (i % 2 == 0) ? (b & 0x0f) : ((b >> 4) & 0x0f);
  return v >= 8 ? v - 16 : v;
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 128;
  int N = argc > 2 ? atoi(argv[2]) : 128;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  int iters = argc > 4 ? atoi(argv[4]) : 100;
  if (K % 64) { printf("K must be /64\n"); return 1; }
  srand(7);
  std::vector<uint8_t> ha(M * K / 2), hw(N * K / 2);
  for (auto& b : ha) b = rand() & 0xff;
  for (auto& b : hw) b = rand() & 0xff;
  uint8_t *da, *dw; int32_t* dc;
  cudaMalloc(&da, ha.size());
  cudaMalloc(&dw, hw.size());
  cudaMalloc(&dc, (size_t)M * N * 4);
  cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice);
  auto st = w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0);
  cudaDeviceSynchronize();
  if (st != w4a4_sm89::Status::kSuccess) { printf("gemm fail\n"); return 1; }
  std::vector<int32_t> hc((size_t)M * N);
  cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < 64; t++) {
    int m = rand() % M, n = rand() % N;
    long ref = 0;
    for (int k = 0; k < K; k++)
      ref += (long)unpack(ha, (size_t)m * K + k)
           * (long)unpack(hw, (size_t)n * K + k);
    if (ref != hc[(size_t)m * N + n]) bad++;
  }
  printf("exact-check: %s (%d/64 mismatches)\n",
         bad ? "FAIL" : "PASS", bad);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 10; i++)
    w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++)
    w4a4_sm89::gemm_s4s4s32(da, dw, dc, M, N, K, nullptr, 0, 0);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double tops = 2.0 * M * N * K * iters / (ms / 1e3) / 1e12;
  printf("M=%d N=%d K=%d  GEMM-only %.4f ms  %.1f TOPS\n",
         M, N, K, ms / iters, tops);
  return 0;
}
