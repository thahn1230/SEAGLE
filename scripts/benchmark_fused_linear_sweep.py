#!/usr/bin/env python
"""Linear-level A-E segment sweep (CUDA events, 100 warmup/500 iters)
+ INT32-intermediate memory proof. GPU: CUDA_VISIBLE_DEVICES-selected."""
import argparse, csv, json, os, sys
import torch
sys.path.insert(0, "src")
sys.path.insert(0, "kernels/w4a4_cutlass_sm89")
from build_torch_ext import ext

def bench(fn, iters=500, warmup=100):
    for _ in range(warmup): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(5):
        s.record()
        for _ in range(iters // 5): fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / (iters // 5))
    import statistics as st
    return dict(median=round(st.median(ts), 5),
                mean=round(st.mean(ts), 5),
                p10=round(sorted(ts)[0], 5), p90=round(sorted(ts)[-1], 5),
                std=round(st.pstdev(ts), 6))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    for K in (4096, 8192):
        N = 4096
        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        qw, sw = ext.quantize(W)
        lin = torch.nn.Linear(K, N, bias=True).half().cuda()
        Ms = (1, 16, 26, 64, 256, 512) if K == 4096 else (1, 16, 26, 64)
        for M in Ms:
            X = torch.randn(M, K, device="cuda", dtype=torch.float16)
            qa, sa = ext.quantize(X)
            c32 = torch.empty(M, N, dtype=torch.int32, device="cuda")
            r = dict(M=M, N=N, K=K)
            r["quant_ms"] = bench(lambda: ext.quantize(X))["median"]
            r["gemm_only_ms"] = bench(lambda: ext.gemm_unfused(qa, qw, sa, sw, None))["median"]  # includes dequant; isolate below
            ded = bench(lambda: ext.gemm_fused(qa, qw, sa, sw, b))
            r["fused_gemm_epilogue_ms"] = ded["median"]
            r["fused_stats"] = ded
            r["unfused_total_ms"] = bench(lambda: (ext.quantize(X), ext.gemm_unfused(*ext.quantize(X), sw, None)) and None)["median"] if False else None
            r["full_unfused_ms"] = bench(lambda: ext.gemm_unfused(*ext.quantize(X), qw and qw, sw, b) if False else (lambda q=ext.quantize(X): ext.gemm_unfused(q[0], qw, q[1], sw, b))())["median"]
            r["full_fused_ms"] = bench(lambda: (lambda q=ext.quantize(X): ext.gemm_fused(q[0], qw, q[1], sw, b))())["median"]
            r["fp16_ms"] = bench(lambda: lin(X))["median"]
            r["fused_over_unfused"] = round(r["full_fused_ms"] / r["full_unfused_ms"], 3)
            r["fused_over_fp16"] = round(r["full_fused_ms"] / r["fp16_ms"], 3)
            rows.append(r)
            print(f"M={M:4d} K={K}: fused {r['full_fused_ms']:.4f} unfused {r['full_unfused_ms']:.4f} fp16 {r['fp16_ms']:.4f}", flush=True)
    # memory proof: no int32 intermediate in fused path
    X = torch.randn(512, 4096, device="cuda", dtype=torch.float16)
    qa, sa = ext.quantize(X)
    W = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    qw, sw = ext.quantize(W)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = ext.gemm_fused(qa, qw, sa, sw, None)
    fused_peak = torch.cuda.max_memory_allocated() - base
    del y; torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = ext.gemm_unfused(qa, qw, sa, sw, None)
    unfused_peak = torch.cuda.max_memory_allocated() - base
    mem = dict(fused_peak_extra_bytes=int(fused_peak),
               unfused_peak_extra_bytes=int(unfused_peak),
               int32_intermediate_bytes=512 * 4096 * 4,
               fused_allocates_c32=bool(fused_peak >= 512 * 4096 * 4 + 512 * 4096 * 2))
    print("memory:", mem)
    rd = a.run_dir
    with open(os.path.join(rd, "linear_benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "fused_stats"])
        w.writeheader()
        for r in rows: w.writerow({k: v for k, v in r.items() if k != "fused_stats"})
    json.dump(dict(rows=rows, memory=mem), open(os.path.join(rd, "linear_benchmark.json"), "w"), indent=1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
