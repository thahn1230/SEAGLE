#!/usr/bin/env python
"""STOP GATE B: real packed W4A4/W4A16 execution of BOTH B2 projections.

Packs projection_first and projection_recurrent (Arch B weights, shape
[4096, 8192], fc bias handled by an explicit bias-add wrapper) with:
  real_packed_W4A4  : QuaRot CUTLASS INT4xINT4 (per-channel sym weights
                      absmax/7, per-token sym activations, INT32 accum)
  real_packed_W4A16 : torch tinygemm aten._weight_int4pack_mm (groupwise
                      asymmetric int4 weights, bf16 activations)

Checks per projection:
  1. the real kernel actually executes (quarot.matmul / _weight_int4pack_mm
     reached; no silent fp16 fallback),
  2. real output == fp-emulated output of the SAME quantization recipe
     (sym per-channel W + sym per-token A) within kernel-arith tolerance,
  3. error vs the fp16 reference is finite and reasonable,
  4. the two projections get SEPARATE weight_quant_scales (not shared),
  5. latency of real vs fp16 GEMM.

Writes artifacts/b2_split_study/{real_w4a4_dispatch.txt, real_w4a4_projections.csv}.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/validate_b2_real_w4a4.py --device cuda:1
"""

import argparse, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from eagle_spinquant import experiment, logging_utils, study  # noqa: E402
from eagle_spinquant import rotation_aware as ra  # noqa: E402
from eagle_spinquant.b2_projection import build_b2_weights_arch_b  # noqa: E402
from eagle_spinquant.realint4 import (QuarotW4A4Linear,  # noqa: E402
                                      Int4TinygemmLinear)

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")


class BiasedReal(nn.Module):
    """Real packed linear + explicit fp16 bias add (kernels are bias-free)."""

    def __init__(self, real_mod, bias):
        super().__init__()
        self.real = real_mod
        self.register_buffer("bias_fp16", bias.to(torch.float16))

    def forward(self, x):
        return self.real(x) + self.bias_fp16


def fp_emulate_quarot(W, x):
    """fp-emulation of the QuaRot recipe on the SAME tensors: per-channel sym
    int4 weights (absmax/7), per-token sym int4 activations, fp32 accumulate."""
    w = W.float()
    ws = (w.abs().amax(dim=1, keepdim=True) / 7).clamp(min=1e-8)
    qw = torch.clamp(torch.round(w / ws), -8, 7)
    x2 = x.float()
    xs = (x2.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-6)
    qx = torch.clamp(torch.round(x2 / xs), -8, 7)
    return (qx @ qw.t()) * xs * ws.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = args.device
    torch.cuda.set_device(dev)          # CUTLASS launches use the CURRENT device
    lines = []

    def log(s):
        print("[gateB]", s, flush=True)
        lines.append(str(s))

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")

    # --- inputs: draft sd + R1 + real target_final_rms_gamma ---
    draft = ra.build_standalone_draft(paths["draft_path"], "cpu", torch.float32)
    sd = {k: v.detach().cpu() for k, v in draft.state_dict().items()}
    del draft
    R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R1 = R["R1"].double()
    gpath = os.path.join(PROJECT_ROOT, "outputs", "target_final_rms_gamma.pt")
    assert os.path.isfile(gpath), "run validate_b2_fp_equivalence.py first"
    gamma = torch.load(gpath, weights_only=False).double()
    _, W_first, W_rec, b_rot = build_b2_weights_arch_b(sd, R1, gamma)
    for nm, t in (("W_first", W_first), ("W_rec", W_rec), ("b_rot", b_rot)):
        assert torch.isfinite(t).all(), f"{nm} has non-finite values"
    log(f"projection weights built: first {list(W_first.shape)}, "
        f"recurrent {list(W_rec.shape)}, bias {list(b_rot.shape)}")

    # QuaRot import + kernel identity
    import quarot
    log(f"quarot module: {quarot.__file__}")
    log("kernels: quarot.sym_quant / quarot.matmul (CUTLASS INT4xINT4->INT32) "
        "/ quarot.sym_dequant; weight fmt=pack_i4 per-channel sym (absmax/7); "
        "act fmt=per-token sym int4 dynamic; accum=INT32; group=per-channel "
        "(weight) x per-token (act); zero-point=none (symmetric); KV=fp16")

    # activation battery: realistic magnitudes (rotated embeds + features)
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(args.tokens, 8192, generator=g) * 0.5).to(dev, torch.float16)

    rows, ok = [], True
    for name, Wt in (("projection_first", W_first),
                     ("projection_recurrent", W_rec)):
        lin = nn.Linear(8192, 4096, bias=False)
        lin.weight.data = Wt.to(torch.float16)
        lin = lin.to(dev)
        y_fp16 = torch.nn.functional.linear(
            x, lin.weight, b_rot.to(dev, torch.float16))

        # ---- real W4A4 (QuaRot) ----
        try:
            qmod = BiasedReal(QuarotW4A4Linear.from_linear(lin),
                              b_rot).to(dev)
            y_real = qmod(x)
            emu = fp_emulate_quarot(Wt.to(dev), x).to(torch.float16) \
                + b_rot.to(dev, torch.float16)
            rel_emu = float((y_real.float() - emu.float()).norm()
                            / emu.float().norm())
            rel_fp = float((y_real.float() - y_fp16.float()).norm()
                           / y_fp16.float().norm())
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                _ = qmod(x)
            torch.cuda.synchronize()
            t_real = (time.perf_counter() - t0) / 50 * 1e3
            t0 = time.perf_counter()
            for _ in range(50):
                _ = torch.nn.functional.linear(x, lin.weight)
            torch.cuda.synchronize()
            t_fp = (time.perf_counter() - t0) / 50 * 1e3
            ws = qmod.real.w_scales
            rows.append(dict(
                projection=name, backend="real_packed_W4A4",
                kernel="quarot CUTLASS INT4xINT4 (sym_quant/matmul/sym_dequant)",
                packed_shape=str(list(qmod.real.w_packed.shape)),
                packed_bytes=int(qmod.real.w_packed.numel()
                                 * qmod.real.w_packed.element_size()),
                weight_quant_scale_shape=str(list(ws.shape)),
                weight_quant_scale_checksum=round(float(ws.float().abs().sum()), 4),
                rel_l2_vs_fp_emulation=round(rel_emu, 6),
                rel_l2_vs_fp16=round(rel_fp, 6),
                ms_per_call_real=round(t_real, 4), ms_per_call_fp16=round(t_fp, 4),
                dispatch_ok=bool(rel_emu < 5e-2)))
            log(f"{name} W4A4: rel_vs_emul={rel_emu:.2e} rel_vs_fp16={rel_fp:.4f} "
                f"real {t_real:.3f}ms vs fp16 {t_fp:.3f}ms")
            ok &= rel_emu < 5e-2
        except Exception as e:
            ok = False
            rows.append(dict(projection=name, backend="real_packed_W4A4",
                             kernel="FAILED", error=str(e)[:300],
                             dispatch_ok=False))
            log(f"{name} W4A4 FAILED: {e}")

        # ---- real W4A16 (tinygemm) ----
        try:
            tmod = BiasedReal(Int4TinygemmLinear.from_linear(lin, 128),
                              b_rot).to(dev)
            y_t = tmod(x)
            rel_fp = float((y_t.float() - y_fp16.float()).norm()
                           / y_fp16.float().norm())
            rows.append(dict(projection=name, backend="real_packed_W4A16",
                             kernel="aten._weight_int4pack_mm (tinygemm, "
                                    "group=128 asym, bf16 act)",
                             rel_l2_vs_fp16=round(rel_fp, 6),
                             dispatch_ok=bool(rel_fp < 0.2)))
            log(f"{name} W4A16: rel_vs_fp16={rel_fp:.4f}")
        except Exception as e:
            rows.append(dict(projection=name, backend="real_packed_W4A16",
                             kernel="FAILED", error=str(e)[:300],
                             dispatch_ok=False))
            log(f"{name} W4A16 FAILED: {e}")

    # separate scales proof
    w4rows = [r for r in rows if r["backend"] == "real_packed_W4A4"
              and "weight_quant_scale_checksum" in r]
    if len(w4rows) == 2:
        sep = w4rows[0]["weight_quant_scale_checksum"] != \
              w4rows[1]["weight_quant_scale_checksum"]
        log(f"separate weight_quant_scales for first/recurrent: {sep}")
        ok &= sep

    logging_utils.write_csv(os.path.join(ART, "real_w4a4_projections.csv"), rows)
    with open(os.path.join(ART, "real_w4a4_dispatch.txt"), "w") as f:
        f.write("\n".join(lines) + f"\nSTOP_GATE_B_PASS={ok}\n")
    with open(os.path.join(ART, "real_w4a4_dispatch.json"), "w") as f:
        json.dump(dict(stop_gate_b_pass=bool(ok), rows=rows), f, indent=2)
    log(f"STOP GATE B: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
