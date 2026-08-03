#!/usr/bin/env python
"""Runtime/memory cost of the projection-local rotation (spec §20).

Measures, with CUDA events (Python-level fake-quant pipeline — stated
as such; not packed-INT4 kernel numbers):
  - offline weight transform time per config
  - activation rotation latency vs token count and block size
  - projection latency (rotation + linear) ratio vs EP3-P
  - end-to-end: reuses gen_seconds from the MT-Bench shards
  - peak memory delta
Also reports FWHT arithmetic-op/memory-traffic estimates per token.
Writes tables/rep3p_overhead.json.
"""
import argparse, csv, json, math, os, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eagle_spinquant.projection_rotation import StructuredRotation

D = 4096


def cuda_ms(fn, iters=50, warmup=10):
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
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    dev = "cuda:0"
    W = torch.randn(D, 2 * D, device=dev, dtype=torch.float16)
    lin = torch.nn.Linear(2 * D, D, bias=True).half().to(dev)
    out = dict(note="Python/fake-quant-level timing; packed-INT4 "
                    "kernel numbers NOT available in this repo "
                    "(fake-quant only). FWHT op counts are exact.")
    configs = dict(
        blockdiag=dict(family="dual", block=256, seed=14),
        cross_b16=dict(family="cross", block=16, seed=5,
                       interleave_chunk=1),
        cross_b8192=dict(family="cross", block=8192, seed=12,
                         interleave_chunk=1),
        full_b8192=dict(family="full", block=8192, seed=7))
    rows = []
    for name, spec in configs.items():
        rot = StructuredRotation(spec, device=dev)
        rot.sign = rot.sign.to(dev).half()
        if rot.perm_in is not None:
            rot.perm_in = rot.perm_in.to(dev)
        t0 = time.time()
        _ = rot.apply(W.float()).half()
        torch.cuda.synchronize()
        offline_s = time.time() - t0
        for ntok in (1, 16, 64, 256):
            x = torch.randn(ntok, 2 * D, device=dev,
                            dtype=torch.float16)
            rot_ms = cuda_ms(lambda: rot.apply(x))
            base_ms = cuda_ms(lambda: lin(x))
            both_ms = cuda_ms(lambda: lin(rot.apply(x)))
            b = spec["block"]
            rows.append(dict(
                config=name, block=b, n_tokens=ntok,
                rotation_ms=round(rot_ms, 4),
                linear_ms=round(base_ms, 4),
                rot_plus_linear_ms=round(both_ms, 4),
                projection_ratio=round(both_ms / base_ms, 3),
                offline_weight_transform_s=round(offline_s, 3),
                fwht_adds_per_token=int(2 * D * math.log2(b)),
                fwht_bytes_per_token=int(2 * D * 2 * 2
                                         * math.log2(b)),
                kernel_launches="~log2(block) butterfly steps + "
                                "gather (unfused torch)"))
    out["micro"] = rows
    # peak memory: transformed weight adds one fp16 [D, 2D] copy per
    # path when rotation is enabled (same as EP3-P view)
    out["extra_weight_mib"] = round(D * 2 * D * 2 / 2 ** 20, 1)
    # end-to-end from shards when present
    e2e = {}
    for tag in ("MTX_EP3P", "MTX_R0", "MTX_BLOCKDIAG", "MTX_FULL"):
        p = os.path.join(rd, "shards",
                         f"al__{tag}__int4__mtbench.csv")
        if os.path.exists(p):
            secs = toks = 0.0
            for r in csv.DictReader(open(p)):
                secs += float(r["gen_seconds"])
                toks += sum(json.loads(r["acceptance_list"]))
            e2e[tag] = round(1000 * secs / max(toks, 1), 2)
    # include best R tag dynamically
    import glob
    for p in glob.glob(os.path.join(rd, "shards",
                                    "al__MTX_R?__int4__mtbench.csv")):
        tag = os.path.basename(p).split("__")[0][4:]
        secs = toks = 0.0
        for r in csv.DictReader(open(p)):
            secs += float(r["gen_seconds"])
            toks += sum(json.loads(r["acceptance_list"]))
        e2e[tag] = round(1000 * secs / max(toks, 1), 2)
    out["e2e_ms_per_token"] = e2e
    json.dump(out, open(os.path.join(
        rd, "tables", "rep3p_overhead.json"), "w"), indent=1)
    for r in rows:
        if r["n_tokens"] == 64:
            print(f"[ovh] {r['config']:12s} rot {r['rotation_ms']:.3f}"
                  f"ms proj-ratio {r['projection_ratio']}")
    print(f"[ovh] e2e {e2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
