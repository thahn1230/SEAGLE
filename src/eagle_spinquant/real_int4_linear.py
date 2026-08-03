"""Real INT4 (s4 x s4 -> s32 Tensor Core) linear for SEAGLE E2E.

Backends (SEAGLE_INT4_EPILOGUE env, default "fused"):
  fused   : quant -> single CUTLASS kernel (GEMM + s_a[m]*s_w[n]
            + bias + fp16 store via epilogue visitor). No [M,N] int32
            intermediate, no separate dequant launch. BITWISE equal
            to unfused (verified all shapes, bias/no-bias).
  unfused : quant -> s4s4s32 GEMM -> separate dequant kernel
            (A/B reference path).
Both use the self-contained seagle_int4_ext (w4a4_cutlass_sm89),
IMMA-verified; per-row dynamic A4 (absmax/7, clamp +-7, fp32 scales),
per-output-channel W4.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "kernels", "w4a4_cutlass_sm89"))
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
from build_torch_ext import ext as _EXT  # noqa: E402

_BACKEND = os.environ.get("SEAGLE_INT4_EPILOGUE", "fused")
assert _BACKEND in ("fused", "unfused"), _BACKEND


@torch.no_grad()
def pack_weight_int4(weight: torch.Tensor):
    """fp16/fp32 [N, K] -> (packed uint8 [N, K/2], fp32 scales [N])
    per-output-channel symmetric (absmax/7, clamp +-7)."""
    w = weight.detach().half().cuda().contiguous()
    return _EXT.quantize(w)


# same-input activation-quant cache: q/k/v (and gate/up) consume the
# SAME tensor within one forward — quantize once per live tensor.
_QCACHE = {"key": None, "qa": None, "sa": None}


def _act_quant_cached(x2):
    key = (id(x2), x2.data_ptr(), x2.shape)
    if _QCACHE["key"] == key:
        return _QCACHE["qa"], _QCACHE["sa"]
    qa, sa = _EXT.quantize(x2)
    _QCACHE.update(key=key, qa=qa, sa=sa)
    return qa, sa


class RealInt4Linear(torch.nn.Module):
    """Drop-in nn.Linear replacement with real INT4 compute."""

    backend = _BACKEND

    def __init__(self, lin: torch.nn.Linear | None = None,
                 weight: torch.Tensor = None,
                 bias: torch.Tensor = None, name: str = ""):
        super().__init__()
        w = lin.weight if lin is not None else weight
        b = (lin.bias if lin is not None else bias)
        assert w.ndim == 2 and w.shape[1] % 64 == 0, w.shape
        self.in_features = int(w.shape[1])
        self.out_features = int(w.shape[0])
        packed, scale = pack_weight_int4(w)
        self.register_buffer("w4", packed)
        self.register_buffer("sw", scale)
        self.register_buffer(
            "bias_fp16",
            b.detach().half().cuda().contiguous()
            if b is not None else None)
        if lin is not None:               # free fp16 source LAST
            lin.weight.data = torch.empty(0)
        self.name = name

    @property
    def weight(self):        # device/dtype discovery shim (EAGLE code)
        return self.sw

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.float16 or not x2.is_contiguous():
            x2 = x2.half().contiguous()
        qa, sa = _act_quant_cached(x2)
        if self.backend == "fused":
            y = _EXT.gemm_fused(qa, self.w4, sa, self.sw,
                                self.bias_fp16)
        else:
            y = _EXT.gemm_unfused(qa, self.w4, sa, self.sw,
                                  self.bias_fp16)
        return y.reshape(*shp[:-1], self.out_features)


@torch.no_grad()
def swap_llama_linears_int4(model, layer_attr_paths=None,
                            verbose=False):
    """Replace q/k/v/o/gate/up/down of every decoder layer with
    RealInt4Linear (embed/norm/head stay fp16)."""
    n = 0
    names = ("q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj")
    for mod_name, mod in model.named_modules():
        for attr in names:
            child = getattr(mod, attr, None)
            if isinstance(child, torch.nn.Linear) \
                    and child.weight.ndim == 2 \
                    and child.weight.numel() > 0:
                setattr(mod, attr, RealInt4Linear(
                    child, name=f"{mod_name}.{attr}"))
                n += 1
                if verbose:
                    print(f"[int4] {mod_name}.{attr}")
    return n
