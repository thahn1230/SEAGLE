"""REAL packed-INT4 execution for the EAGLE rotation study target.

Two backends, both proven standalone by scripts/validate_real_int4_kernel.py
before any use here:

  tinygemm_w4a16 : torch-native aten._weight_int4pack_mm (packed int4 weights,
                   16-bit activations). REAL W4A16 — never call it W4A4.
  quarot_w4a4    : QuaRot CUTLASS kernels (packed int4 weights AND per-token
                   packed int4 activations; INT4xINT4 GEMM -> INT32 ->
                   dequant). REAL W4A4 for the 7 per-layer linears only;
                   lm_head/embeddings/KV cache/attention math stay fp16.

Only the TARGET's per-layer linears (q/k/v/o/gate/up/down) are swapped.
The draft, lm_head and embeddings are untouched, so the draft-interface
variants (naive/A/B2) mean exactly what they meant in the fake-quant study.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from . import spinquant_bridge as sb
from .study import TwoPathAdapter

LINEAR_NAMES = [
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
]


# ---------------------------------------------------------------------------
# tinygemm W4A16
# ---------------------------------------------------------------------------

def group_quantize_int4(w: torch.Tensor, groupsize: int = 128):
    """gpt-fast/torchao groupwise asymmetric int4 (see validate script)."""
    N, K = w.shape
    assert K % groupsize == 0, (N, K, groupsize)
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
    # .contiguous() is load-bearing: rotated v/o_proj weights are
    # non-contiguous after SpinQuant's R2 per-head transforms, and that
    # layout propagates through the quant arithmetic into the packed bytes.
    q_int32 = q_int32.contiguous()
    try:
        packed_u8 = (q_int32[:, ::2] << 4
                     | q_int32[:, 1::2]).to(torch.uint8).contiguous()
        return torch.ops.aten._convert_weight_to_int4pack(packed_u8,
                                                          inner_k_tiles)
    except Exception:
        return torch.ops.aten._convert_weight_to_int4pack(q_int32,
                                                          inner_k_tiles)


class Int4TinygemmLinear(nn.Module):
    """REAL W4A16: packed int4 weights, tinygemm kernel, bf16 activations
    internally (cast from/to the pipeline dtype at the module boundary)."""

    def __init__(self, packed, scales_and_zeros, in_features, out_features,
                 groupsize, out_dtype):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales_and_zeros", scales_and_zeros)
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.out_dtype = out_dtype

    @classmethod
    def from_linear(cls, lin: nn.Linear, groupsize: int = 128):
        assert lin.bias is None, "llama target linears carry no bias"
        w = lin.weight.data
        q, snz = group_quantize_int4(w, groupsize)
        packed = pack_int4_weight(q.to(w.device))
        return cls(packed, snz.to(w.device), lin.in_features,
                   lin.out_features, groupsize, w.dtype)

    @property
    def weight(self):
        # EAGLE's ea_model reads `<linear>.weight.device` at generate time
        # (ea_model.py:178/278/382); expose the packed buffer for that.
        return self.packed

    def forward(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        y = torch.ops.aten._weight_int4pack_mm(
            x2.to(torch.bfloat16), self.packed, self.groupsize,
            self.scales_and_zeros)
        return y.to(self.out_dtype).reshape(*shp, self.out_features)


# ---------------------------------------------------------------------------
# QuaRot W4A4 (linears only)
# ---------------------------------------------------------------------------

class QuarotW4A4Linear(nn.Module):
    """REAL W4A4 linear: per-channel symmetric int4 weights (absmax/7),
    per-token symmetric int4 activations quantized on the fly (QuaRot
    Quantizer recipe, clip 1.0), INT4xINT4 CUTLASS GEMM, fused dequant.
    online_had=True applies the SpinQuant R4 online Hadamard to the input
    (down_proj only; its inverse must already be folded into the weight)."""

    def __init__(self, w_packed, w_scales16, in_features, out_features,
                 online_had=False, had_K=None, K=None):
        super().__init__()
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scales", w_scales16)
        self.in_features = in_features
        self.out_features = out_features
        self.online_had = online_had
        if online_had:
            self.register_buffer("had_K", had_K)
            self.K = K

    @classmethod
    def from_linear(cls, lin: nn.Linear, online_had=False, had_K=None, K=None):
        import quarot
        from quarot.functional.quantization import pack_i4
        assert lin.bias is None
        w = lin.weight.data.float().contiguous()
        w_scale = (w.abs().amax(dim=1, keepdim=True) / 7).clamp(min=1e-8)
        qw = torch.clamp(torch.round(w / w_scale), -8, 7)
        w_packed = pack_i4(qw.to(torch.int8).cpu().contiguous()) \
            .contiguous().to(lin.weight.device)
        return cls(w_packed, w_scale.to(torch.float16).to(lin.weight.device),
                   lin.in_features, lin.out_features,
                   online_had=online_had, had_K=had_K, K=K)

    @property
    def weight(self):
        # device-discovery shim for EAGLE's ea_model (see Int4TinygemmLinear)
        return self.w_packed

    def forward(self, x):
        import quarot
        shp = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.online_had:
            sb.add_spinquant_to_syspath()
            from utils import hadamard_utils
            x2 = hadamard_utils.matmul_hadU_cuda(x2, self.had_K, self.K)
        x2 = x2.to(torch.float16).contiguous()
        sx = (x2.abs().amax(dim=-1, keepdim=True) / 7).to(torch.float16) \
            .clamp(min=1e-6)
        qx = quarot.sym_quant(x2, sx)
        y32 = quarot.matmul(qx, self.w_packed)
        y = quarot.sym_dequant(y32, sx, self.w_scales)
        return y.reshape(*shp, self.out_features)


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------

def fold_r4_into_down_proj(base_model):
    """Fold the exact Hadamard into W_down offline (SpinQuant R4); returns
    (had_K, K) for the matching online transform on the input side."""
    sb.add_spinquant_to_syspath()
    from utils import hadamard_utils
    had_K, K = hadamard_utils.get_hadK(base_model.config.intermediate_size)
    for layer in base_model.model.layers:
        hadamard_utils.apply_exact_had_to_linear(
            layer.mlp.down_proj, had_dim=-1, output=False)
    return had_K, K


@torch.inference_mode()
def swap_target_linears(base_model, backend: str, device: str,
                        groupsize: int = 128) -> dict:
    """Replace the 7 per-layer linears of the (already rotated) target with
    real-INT4 modules. Frees the fp16 weights as it goes."""
    assert backend in ("tinygemm_w4a16", "quarot_w4a4")
    had_K = K = None
    if backend == "quarot_w4a4":
        had_K, K = fold_r4_into_down_proj(base_model)
        if had_K is not None:
            had_K = had_K.to(device)
    n = 0
    for layer in base_model.model.layers:
        for parent_name, name in LINEAR_NAMES:
            parent = getattr(layer, parent_name)
            lin = getattr(parent, name)
            lin = lin.to(device)
            if backend == "tinygemm_w4a16":
                mod = Int4TinygemmLinear.from_linear(lin, groupsize)
            else:
                online = (name == "down_proj")
                mod = QuarotW4A4Linear.from_linear(
                    lin, online_had=online, had_K=had_K if online else None,
                    K=K if online else None)
            setattr(parent, name, mod)
            del lin
            n += 1
        torch.cuda.empty_cache()
    return {"replaced_linears": n, "backend": backend,
            "weight_bits": 4,
            "act_bits": 16 if backend == "tinygemm_w4a16" else 4,
            "kv_bits": 16,
            "post_swap_mem_alloc_gib":
                torch.cuda.memory_allocated() / 2 ** 30}


# ---------------------------------------------------------------------------
# B2 with swap-overhead accounting
# ---------------------------------------------------------------------------

class TimedTwoPathAdapter(TwoPathAdapter):
    """B2 + CPU-side accounting of the two-path dispatch overhead (pointer
    swap + branch). The overhead is CPU-side by construction (no kernel is
    launched), so perf_counter is the right clock."""
    name = "B2"

    def install(self):
        out = super().install()
        self.swap_seconds = 0.0
        self.swap_calls = 0
        fc = self.ea_layer.fc
        adapter = self
        inner = self.ea_layer.forward  # = patched_forward from super()

        def timed_forward(hidden_states, *a, **k):
            t0 = time.perf_counter()
            if adapter._pending_external:
                fc.weight.data = adapter.W_folded
                adapter._pending_external = False
            else:
                fc.weight.data = adapter.W_orig
            adapter.swap_seconds += time.perf_counter() - t0
            adapter.swap_calls += 1
            return adapter._orig_forward(hidden_states, *a, **k)
        self.ea_layer.forward = timed_forward
        self._inner_patched = inner
        return out

    def swap_stats(self):
        return {"b2_swap_overhead_ms_total": self.swap_seconds * 1e3,
                "b2_swap_calls": self.swap_calls}

    def reset_swap(self):
        self.swap_seconds = 0.0
        self.swap_calls = 0
