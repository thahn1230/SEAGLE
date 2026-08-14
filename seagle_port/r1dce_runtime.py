"""R1DCE §25 — runtime/memory cost of the H_t context rotation.

Measures the dense Ht @ R matmul (the CLASS-A runtime op, identical shape
for R1_D / R1_T / any dense R_C) at S in {1,8,32,128,512}, fp32 and bf16,
plus the structured fast-Hadamard transform if importable.  CUDA-event
timing, 200 iters after 50 warmup.  Writes
tables/runtime_context_rotation.csv + tables/memory_cost.csv.
"""
import argparse
import csv
import os

import torch

D = 4096
SS = (1, 8, 32, 128, 512)


def time_op(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3          # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    R32 = torch.randn(D, D, device=dev)
    Q, _ = torch.linalg.qr(R32)
    R32 = Q.contiguous()
    R16 = R32.to(torch.bfloat16)
    rows = []
    mmac_tok = D * D / 1e6
    for S in SS:
        x32 = torch.randn(S, D, device=dev)
        x16 = x32.to(torch.bfloat16)
        for dt, x, R in (("fp32", x32, R32), ("bf16", x16, R16)):
            us = time_op(lambda: x @ R)
            rows.append({"op": f"Ht@R_dense_{dt}", "S": S,
                         "us_per_forward": us, "us_per_token": us / S,
                         "MMAC_per_token": mmac_tok,
                         "mem_read_MB": (x.numel() * x.element_size()
                                         + R.numel() * R.element_size())
                         / 1e6,
                         "mem_write_MB": x.numel() * x.element_size() / 1e6})
        try:
            from fast_hadamard_transform import hadamard_transform
            scale = 1.0 / (D ** 0.5)
            us = time_op(lambda: hadamard_transform(x16, scale))
            rows.append({"op": "Ht_fast_hadamard_bf16", "S": S,
                         "us_per_forward": us, "us_per_token": us / S,
                         "MMAC_per_token": 0.0,   # O(d log d) adds ~0.05
                         "mem_read_MB": x16.numel() * 2 / 1e6,
                         "mem_write_MB": x16.numel() * 2 / 1e6})
        except Exception as ex:                       # noqa: BLE001
            if S == SS[0]:
                print(f"[runtime] fast_hadamard_transform unavailable: {ex}")
    for r in rows:
        print(f"  {r['op']:24s} S={r['S']:4d} {r['us_per_forward']:8.1f} "
              f"us/fwd  {r['us_per_token']:8.2f} us/tok")
    os.makedirs(f"{args.run_dir}/tables", exist_ok=True)
    with open(f"{args.run_dir}/tables/runtime_context_rotation.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mem = [
        {"item": "separate R_C matrix (current M5/M6)",
         "bytes_fp32": D * D * 4, "bytes_bf16": D * D * 2,
         "note": "extra checkpoint + resident buffer"},
        {"item": "R1_D reused at H_t (C1)",
         "bytes_fp32": 0, "bytes_bf16": 0,
         "note": "R1_frozen [4096^2] fp32 already resident for the draft "
                 "path in freeze_for_eval; rc_matrix_buf can alias the SAME "
                 "tensor -> zero ADDITIONAL storage, zero new checkpoint. "
                 "Ht@R1_D REMAINS a runtime op (not zero-runtime)."},
        {"item": "ctx K/V weight views (any rotation, incl. none)",
         "bytes_fp32": None,
         "bytes_bf16": 5 * 2 * 1024 * D * 2,
         "note": "mandatory in M3 already (gamma contract); rotation folded "
                 "into the same tensors offline"},
    ]
    with open(f"{args.run_dir}/tables/memory_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mem[0].keys()))
        w.writeheader()
        w.writerows(mem)
    print("wrote tables/runtime_context_rotation.csv + memory_cost.csv")


if __name__ == "__main__":
    main()
