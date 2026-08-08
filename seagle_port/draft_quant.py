"""Fake-quantized DFlash draft (SEAGLE policy).

Quantized components ("full" scope): fc (W_c) + every draft layer
q/k/v/o/gate/up/down. Kept bf16: target-shared embed/head, all RMSNorms
(q_norm/k_norm/hidden_norm/input/post/final), RoPE, softmax, residual,
draft KV cache.

Weights: per-channel symmetric RTN + MSE clip. Acts: per-token asymmetric.
P2 option on fc: independent per-source-branch activation scales.
Gate F: `components` restricts quantization to named submodules.
Gate E: QLinear exposes .codes_changed for a quantizer-effect check.
"""
import copy

import torch
from torch import nn

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils

COMPONENTS = ("fc", "q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")


def _wquant(w, bits):
    q = quant_utils.WeightQuantizer()
    q.configure(bits, perchannel=True, sym=True, mse=True)
    q.find_params(w)
    return q.quantize(w)


class QLinear(nn.Module):
    """Weight fake-quantized Linear with per-token asym input act quant.
    branch_dims: optional list of slice widths for independent act scales
    (P2 on fc: [4096]*5)."""

    def __init__(self, lin: nn.Linear, w_bits=4, a_bits=4, branch_dims=None,
                 name=""):
        super().__init__()
        self.name = name
        self.out_features, self.in_features = lin.weight.shape
        w = lin.weight.data.float().cuda()
        if w_bits < 16:
            wq = _wquant(w, w_bits)
            self.codes_changed = bool(not torch.equal(wq, w))
            w = wq
        else:
            self.codes_changed = False
        self.weight = nn.Parameter(
            w.to(dtype=lin.weight.dtype, device=lin.weight.device),
            requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(
            lin.bias.data.clone(), requires_grad=False)
        self.a_bits = a_bits
        self.branch_dims = branch_dims

    def _aq(self, x):
        if self.a_bits >= 16:
            return x
        q = quant_utils.ActQuantizer()
        q.configure(bits=self.a_bits, groupsize=-1, sym=False, clip_ratio=1.0)
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        q.find_params(x2)
        out = q(x2).reshape(shp)
        q.free()
        return out

    def forward(self, x):
        if self.a_bits < 16 and self.branch_dims:
            xs, o = [], 0
            for d in self.branch_dims:
                xs.append(self._aq(x[..., o:o + d]))
                o += d
            x = torch.cat(xs, dim=-1)
        else:
            x = self._aq(x)
        return nn.functional.linear(x, self.weight, self.bias)


def quantize_draft(draft, w_bits=4, a_bits=4, components=None,
                   fc_branch_dims=None):
    """Deep-copied draft with selected components replaced by QLinear.
    components: subset of COMPONENTS (None = all)."""
    comps = set(components or COMPONENTS)
    bad = comps - set(COMPONENTS)
    assert not bad, f"unknown components {bad}"
    d2 = copy.deepcopy(draft)
    n = 0
    if "fc" in comps:
        d2.fc = QLinear(d2.fc, w_bits, a_bits, branch_dims=fc_branch_dims,
                        name="fc")
        n += 1
    for li, layer in enumerate(d2.layers):
        for attr_path in ("self_attn.q_proj", "self_attn.k_proj",
                          "self_attn.v_proj", "self_attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            leaf = attr_path.split(".")[-1]
            if leaf not in comps:
                continue
            parent = layer
            for p in attr_path.split(".")[:-1]:
                parent = getattr(parent, p)
            setattr(parent, leaf,
                    QLinear(getattr(parent, leaf), w_bits, a_bits,
                            name=f"layers.{li}.{attr_path}"))
            n += 1
    d2._quantized_components = sorted(comps)
    d2._n_qlinear = n
    return d2


def audit_quantized(draft):
    """Gate E/F: list QLinear names + codes_changed flags."""
    rows = []
    for name, mod in draft.named_modules():
        if isinstance(mod, QLinear):
            rows.append({"name": name, "w_changed": mod.codes_changed,
                         "a_bits": mod.a_bits})
    return rows
