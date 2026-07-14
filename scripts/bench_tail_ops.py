#!/usr/bin/env python
"""Tail-only microbench: isolate the cost of R1.T, RMSNorm, lm_head, and the
combined explicit (T2/T3) vs fused (T1) tails across representative M.

CUDA events; 30 warmup + 200 measured iters; median/p10/p90/mean/std (us).
Output runs/<run>/tail_microbench.csv.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/bench_tail_ops.py \
      --out-dir runs/tail_unfused_<ts> --vocab 32000
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
from eagle_spinquant import logging_utils, study, tail_unfused as tu  # noqa: E402

DEV = "cuda:0"
MS = [1, 8, 16, 26, 32, 64, 128, 256]


def bench(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)     # ms -> us
    t = torch.tensor(times)
    return {"median_us": t.median().item(),
            "p10_us": t.quantile(0.10).item(), "p90_us": t.quantile(0.90).item(),
            "mean_us": t.mean().item(), "std_us": t.std().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--hidden", type=int, default=4096)
    args = ap.parse_args()
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    D, V, dt, eps = args.hidden, args.vocab, torch.float16, 1e-5
    torch.manual_seed(0)
    R1t = torch.randn(D, D, device=DEV, dtype=dt) * (D ** -0.5)
    gamma = torch.randn(D, device=DEV, dtype=dt).abs() + 0.5
    W_lm = torch.randn(V, D, device=DEV, dtype=dt) * 0.02
    W_fused = torch.randn(V, D, device=DEV, dtype=dt) * 0.02

    rows = []
    for M in MS:
        x = torch.randn(M, D, device=DEV, dtype=dt)
        ops = {
            "r1t_gemm": lambda: (x @ R1t),
            "rmsnorm": lambda: tu.rmsnorm0(x, eps),
            "lm_head": lambda: (x @ W_lm.t()),
            "explicit_tail_T2": lambda: tu.tail_T2(x, R1t, gamma, W_lm, eps)[1],
            "explicit_tail_T3": lambda: tu.tail_T3(x, R1t, gamma, W_lm, eps)[1],
            "fused_tail_T1": lambda: tu.tail_T1(x, W_fused, eps)[1],
        }
        for name, fn in ops.items():
            st = bench(fn)
            rows.append({"variant": name, "M": M, "hidden_dim": D,
                         "vocab_size": V, "dtype": "fp16", "op_name": name,
                         **st, "notes": ""})
        # marginal overhead of explicit tail vs fused tail
        t2 = next(r for r in rows if r["M"] == M and r["op_name"] == "explicit_tail_T2")
        t1 = next(r for r in rows if r["M"] == M and r["op_name"] == "fused_tail_T1")
        print(f"M={M:4d}  T1_fused={t1['median_us']:7.1f}us  "
              f"T2_explicit={t2['median_us']:7.1f}us  "
              f"overhead=+{t2['median_us']-t1['median_us']:6.1f}us "
              f"({100*(t2['median_us']-t1['median_us'])/t1['median_us']:+.1f}%)",
              flush=True)
    logging_utils.write_csv(os.path.join(args.out_dir, "tail_microbench.csv"), rows)
    print(f"-> {args.out_dir}/tail_microbench.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
