"""Real INT4 (s4 x s4 -> s32 Tensor Core) linear for SEAGLE E2E
latency. Backend: the QuaRot CUDA extension (CUTLASS m16n8k64 IMMA,
verified exact against CPU integer dot by the standalone
kernels/w4a4_cutlass_sm89 package: PASS + 850 TOPS @M=512 on 4090).

Pipeline per forward (guide contract):
  fp16 x [M,K] -> per-row sym A4 (scale = absmax/7) -> packed s4
  -> s4s4s32 GEMM vs offline-packed per-out-channel W4
  -> int32 * s_a[m] * s_w[n] (+bias fp16) -> fp16

Constraints: K % 64 == 0 (all SEAGLE shapes qualify: 4096/8192/11008).
Rows are processed as M = prod(batch dims); no padding needed (kernel
handles arbitrary M; Tensor-Core row utilization drops below M=16 —
that is part of what this study measures).
"""
import sys

import torch

sys.path.insert(0, "/data/thahn1230/quarot")
from quarot import _CUDA  # noqa: E402


@torch.no_grad()
def pack_weight_int4(weight: torch.Tensor):
    """weight fp16/fp32 [N, K] -> (packed uint8 [N, K/2], scales
    fp16 [N,1]) — per-output-channel symmetric RTN (guide contract)."""
    w = weight.detach().float().cuda()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
    packed = _CUDA.sym_quant(w.half(), scale.half())
    return packed, scale.half()


# same-input activation-quant cache: q_proj/k_proj/v_proj (and
# gate/up) consume the SAME tensor within one forward — quantize once.
_QCACHE = {"key": None, "qa": None, "sa": None}


def _act_quant_cached(x2):
    # inference-mode tensors have no version counter; the
    # (python-id, data_ptr, shape) triple identifies the same
    # live tensor within one forward (qkv / gate-up reuse)
    key = (id(x2), x2.data_ptr(), x2.shape)
    if _QCACHE["key"] == key:
        return _QCACHE["qa"], _QCACHE["sa"]
    sa = (x2.abs().amax(dim=1, keepdim=True)
          .clamp_min(1e-6).half() / 7.0)
    qa = _CUDA.sym_quant(x2, sa)
    _QCACHE.update(key=key, qa=qa, sa=sa)
    return qa, sa


class RealInt4Linear(torch.nn.Module):
    """Drop-in replacement for nn.Linear with real INT4 compute."""

    def __init__(self, lin: torch.nn.Linear | None = None,
                 weight: torch.Tensor = None,
                 bias: torch.Tensor = None, name: str = ""):
        super().__init__()
        w = lin.weight if lin is not None else weight
        b = (lin.bias if lin is not None else bias)
        assert w.shape[1] % 64 == 0, w.shape
        self.in_features = int(w.shape[1])
        self.out_features = int(w.shape[0])
        packed, scale = pack_weight_int4(w)
        self.register_buffer("w4", packed)
        self.register_buffer("sw", scale)
        self.register_buffer(
            "bias_fp16",
            b.detach().half().cuda() if b is not None else None)
        # free the fp16 source LAST (prevents transient 2x memory
        # during whole-model swap; shape/bias captured above)
        if lin is not None:
            lin.weight.data = torch.empty(0)
        self.name = name

    @property
    def weight(self):        # device/dtype discovery shim (EAGLE code)
        return self.sw

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).half().contiguous()
        qa, sa = _act_quant_cached(x2)
        c32 = _CUDA.matmul(qa, self.w4)
        y = _CUDA.sym_dequant(c32, sa, self.sw, 32)
        if self.bias_fp16 is not None:
            y = y + self.bias_fp16
        return y.reshape(*shp[:-1], self.out_features)


@torch.no_grad()
def swap_llama_linears_int4(model, layer_attr_paths=None,
                            verbose=False):
    """Replace q/k/v/o/gate/up/down of every decoder layer with
    RealInt4Linear (embed/norm/head stay fp16 per the study
    contract). Works on the vendored EAGLE KV llama and the draft
    cnets layer alike."""
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
