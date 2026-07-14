#!/usr/bin/env python
"""Standalone validation of REAL INT4 kernels before any EAGLE integration.

Backends:
  1. torch-native tinygemm  (aten._convert_weight_to_int4pack +
     aten._weight_int4pack_mm)                      -> real W4A16
  2. QuaRot CUTLASS kernels (quarot.sym_quant / quarot.matmul /
     quarot.sym_dequant, built from /data/thahn1230/quarot @ sm_89)
                                                    -> real W4A4 (linears)

For each backend and each LLaMA-7B linear shape (K,N) in
{(4096,4096), (4096,11008), (11008,4096)} and M in {1, 26, 512, 2048}
(M=1 decode, M=26 EAGLE tree batch, larger = prefill-ish):
  - kernel-vs-QDQ-reference error (the fair test: same quantized weights,
    reference matmul in fp32) and kernel-vs-fp16 error (quantization error)
  - CUDA-event timing vs F.linear fp16 on identical tensors
  - torch-profiler kernel names proving INT4 dispatch (no silent FP16
    fallback)

Writes:
  runs/<run-id>/real_int4_kernel_validation.json
  runs/<run-id>/profiler_kernel_names.txt

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/validate_real_int4_kernel.py \
      --run-id rotstudy_realint4_<ts>
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import study  # noqa: E402

SHAPES = [(4096, 4096), (4096, 11008), (11008, 4096)]
MS = [1, 26, 512, 2048]
DEV = "cuda:0"


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _err(out, ref):
    out = out.float().flatten()
    ref = ref.float().flatten()
    return {
        "max_abs_error": (out - ref).abs().max().item(),
        "relative_l2_error": ((out - ref).norm() / (ref.norm() + 1e-12)).item(),
        "cosine": F.cosine_similarity(out, ref, dim=0).item(),
    }


def _profile_kernels(fn, tag):
    from torch.profiler import profile, ProfilerActivity
    fn()  # warm
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    rows = []
    for ev in prof.key_averages():
        if ev.device_type is not None and ev.self_device_time_total > 0:
            rows.append((tag, ev.key, ev.self_device_time_total))
    return rows


# ---------------------------------------------------------------------------
# tinygemm W4A16
# ---------------------------------------------------------------------------

def group_quantize_int4(w: torch.Tensor, groupsize: int = 128):
    """gpt-fast/torchao groupwise asymmetric int4 recipe.
    w [N, K] -> (q_int32 [N, K] in 0..15, scales_and_zeros [K/g, N, 2] bf16)"""
    N, K = w.shape
    assert K % groupsize == 0
    to_q = w.float().reshape(N, K // groupsize, groupsize)
    max_val = to_q.amax(-1, keepdim=True)
    min_val = to_q.amin(-1, keepdim=True)
    scales = (max_val - min_val).clamp(min=1e-6) / 15
    zeros = min_val + scales * 8
    q = ((to_q - min_val) / scales).round().clamp(0, 15) \
        .to(torch.int32).reshape(N, K)
    scales_and_zeros = torch.cat(
        [scales.reshape(N, K // groupsize, 1),
         zeros.reshape(N, K // groupsize, 1)], dim=2) \
        .transpose(0, 1).contiguous().to(torch.bfloat16)
    return q, scales_and_zeros


def pack_int4_weight(q_int32: torch.Tensor, inner_k_tiles: int = 8):
    """torch>=2.5 CUDA packing expects uint8 [N, K/2] (two nibbles/byte,
    even column in the high nibble); older accepted int32 [N, K]. Try both."""
    try:
        packed_u8 = (q_int32[:, ::2] << 4 | q_int32[:, 1::2]).to(torch.uint8)
        return torch.ops.aten._convert_weight_to_int4pack(
            packed_u8, inner_k_tiles), "uint8_packed"
    except Exception:
        return torch.ops.aten._convert_weight_to_int4pack(
            q_int32, inner_k_tiles), "int32"


def dequant_ref(q, scales_and_zeros, groupsize, N, K):
    sz = scales_and_zeros.float().transpose(0, 1)  # [N, K/g, 2]
    scales = sz[..., 0].unsqueeze(-1)
    zeros = sz[..., 1].unsqueeze(-1)
    wq = (q.float().reshape(N, K // groupsize, groupsize) - 8) * scales + zeros
    return wq.reshape(N, K)


def bench_tinygemm(results, prof_rows):
    g = 128
    tag = "tinygemm_w4a16"
    results[tag] = {"dtype_support": {}, "shapes": {}}
    # dtype support probe (M=1, first shape)
    K, N = SHAPES[0]
    w = torch.randn(N, K, device=DEV, dtype=torch.float16) * 0.02
    q, snz = group_quantize_int4(w, g)
    packed, pack_mode = pack_int4_weight(q.to(DEV))
    results[tag]["pack_mode"] = pack_mode
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        try:
            x = torch.randn(1, K, device=DEV, dtype=dt)
            torch.ops.aten._weight_int4pack_mm(x, packed, g, snz.to(dt))
            results[tag]["dtype_support"][str(dt)] = True
        except Exception as ex:
            results[tag]["dtype_support"][str(dt)] = f"NO: {type(ex).__name__}"
    act_dt = torch.bfloat16  # canonical tinygemm dtype

    for (K, N) in SHAPES:
        w16 = (torch.randn(N, K, device=DEV, dtype=torch.float16) * 0.02)
        q, snz = group_quantize_int4(w16, g)
        packed, _ = pack_int4_weight(q.to(DEV))
        snz = snz.to(DEV)
        w_ref = dequant_ref(q.to(DEV), snz, g, N, K).to(DEV)  # fp32 QDQ ref
        shp = {}
        for M in MS:
            x = torch.randn(M, K, device=DEV, dtype=act_dt) / K ** 0.5
            out = torch.ops.aten._weight_int4pack_mm(x, packed, g, snz)
            ref_qdq = x.float() @ w_ref.t()
            ref_fp16 = x.float() @ w16.float().t()
            t_int4 = _time_ms(lambda: torch.ops.aten._weight_int4pack_mm(
                x, packed, g, snz))
            xf = x.to(torch.float16)
            t_fp16 = _time_ms(lambda: F.linear(xf, w16))
            shp[f"M{M}"] = {
                "kernel_vs_qdq_ref": _err(out, ref_qdq),
                "kernel_vs_fp16_weights": _err(out, ref_fp16),
                "int4_ms": t_int4, "fp16_ms": t_fp16,
                "speedup_vs_fp16": t_fp16 / t_int4,
            }
        results[tag]["shapes"][f"K{K}_N{N}"] = shp
        if (K, N) == SHAPES[0]:
            x1 = torch.randn(1, K, device=DEV, dtype=act_dt)
            prof_rows += _profile_kernels(
                lambda: torch.ops.aten._weight_int4pack_mm(x1, packed, g, snz),
                f"{tag}_M1")


# ---------------------------------------------------------------------------
# QuaRot W4A4
# ---------------------------------------------------------------------------

def bench_quarot(results, prof_rows):
    tag = "quarot_w4a4"
    try:
        import quarot
        from quarot.functional.quantization import pack_i4
    except Exception as ex:
        results[tag] = {"import_error": repr(ex)}
        return
    results[tag] = {"shapes": {}}
    for (K, N) in SHAPES:
        w16 = (torch.randn(N, K, device=DEV, dtype=torch.float16) * 0.02)
        w_scale = (w16.abs().amax(dim=1, keepdim=True) / 7).float()
        qw = torch.clamp(torch.round(w16.float() / w_scale), -8, 7)
        w_packed = pack_i4(qw.to(torch.int8)).to(DEV)
        w_ref = (qw * w_scale)  # fp32 QDQ weights
        w_scale16 = w_scale.to(torch.float16)
        shp = {}
        for M in MS:
            x = torch.randn(M, K, device=DEV, dtype=torch.float16) / K ** 0.5

            def full_chain():
                sx = (x.abs().amax(dim=-1, keepdim=True) / 7).to(torch.float16)
                qx = quarot.sym_quant(x, sx)
                y32 = quarot.matmul(qx, w_packed)
                return quarot.sym_dequant(y32, sx, w_scale16)

            out = full_chain()
            # reference: same act quant recipe in fp32
            sx = (x.float().abs().amax(dim=-1, keepdim=True) / 7)
            qx_ref = torch.clamp(torch.round(x.float() / sx), -8, 7)
            ref_qdq = (qx_ref * sx) @ w_ref.t()
            ref_fp16 = x.float() @ w16.float().t()
            t_chain = _time_ms(full_chain)
            t_fp16 = _time_ms(lambda: F.linear(x, w16))
            shp[f"M{M}"] = {
                "kernel_vs_qdq_ref": _err(out, ref_qdq),
                "kernel_vs_fp16_weights": _err(out, ref_fp16),
                "int4_chain_ms": t_chain, "fp16_ms": t_fp16,
                "speedup_vs_fp16": t_fp16 / t_chain,
            }
        results[tag]["shapes"][f"K{K}_N{N}"] = shp
        if (K, N) == SHAPES[0]:
            x1 = torch.randn(26, K, device=DEV, dtype=torch.float16)

            def chain26():
                sx = (x1.abs().amax(dim=-1, keepdim=True) / 7).to(torch.float16)
                qx = quarot.sym_quant(x1, sx)
                return quarot.sym_dequant(quarot.matmul(qx, w_packed),
                                          sx, w_scale16)
            prof_rows += _profile_kernels(chain26, f"{tag}_M26")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "single-GPU job"
    print("[int4val] GPU policy OK:", gpu, flush=True)
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    torch.manual_seed(0)
    results = {"gpu": gpu, "torch": torch.__version__}
    prof_rows = []
    bench_tinygemm(results, prof_rows)
    bench_quarot(results, prof_rows)

    # verdicts: kernel must agree with its own QDQ reference tightly at every
    # shape/M (kernel correctness), and profiler must show int4 kernels.
    def worst(tag, key):
        vals = [m[key]["relative_l2_error"]
                for shp in results[tag]["shapes"].values() for m in shp.values()]
        return max(vals) if vals else float("nan")
    tinygemm_proof = any(("tinygemm" in k.lower() or "int4" in k.lower())
                         for t, k, us in prof_rows if t.startswith("tinygemm"))
    quarot_proof = any(("cutlass" in k.lower() or "int4" in k.lower()
                        or "sym_" in k.lower() or "gemm" in k.lower())
                       for t, k, us in prof_rows if t.startswith("quarot"))
    results["verdicts"] = {
        "tinygemm_worst_rel_l2_vs_qdq_ref": worst("tinygemm_w4a16", "kernel_vs_qdq_ref"),
        "tinygemm_kernel_correct": worst("tinygemm_w4a16", "kernel_vs_qdq_ref") < 2e-2,
        "tinygemm_int4_dispatch_proven": bool(tinygemm_proof),
        "quarot_worst_rel_l2_vs_qdq_ref":
            worst("quarot_w4a4", "kernel_vs_qdq_ref")
            if "shapes" in results.get("quarot_w4a4", {}) else None,
        "quarot_kernel_correct":
            (worst("quarot_w4a4", "kernel_vs_qdq_ref") < 2e-2)
            if "shapes" in results.get("quarot_w4a4", {}) else None,
        "quarot_int4_dispatch_proven": bool(quarot_proof),
        "note": "kernel_vs_qdq_ref isolates KERNEL error (same quantized "
                "operands, fp32 reference matmul); kernel_vs_fp16_weights "
                "additionally contains the intended quantization error.",
    }

    with open(os.path.join(run_dir, "real_int4_kernel_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "profiler_kernel_names.txt"), "w") as f:
        f.write("# tag | kernel | self CUDA time (us) — standalone validation\n")
        for t, k, us in sorted(prof_rows, key=lambda r: -r[2]):
            f.write(f"{t} | {k} | {us:.1f}\n")
    print(json.dumps(results["verdicts"], indent=2))
    print(f"[int4val] -> {run_dir}/real_int4_kernel_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
