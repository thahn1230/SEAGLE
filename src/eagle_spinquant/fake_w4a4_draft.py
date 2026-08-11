"""Fake W4A4 for the pure-R1 SpinQuant EAGLE draft, IN the generation path.

Matches the target's SpinQuant W4A4 policy by reusing SpinQuant's own quantizers:
  weight     : WeightQuantizer(bits=4, perchannel=True, sym=True, mse=True)  (RTN + clip search)
  activation : ActQuantizer(bits=4, groupsize=-1, sym=False)  (per-token asymmetric)
KV is 16-bit (w4a4, not w4a4kv4), so R3 (Q/K Hadamard) is N/A here — exactly as
the target does (r3 auto-added only iff k_bits<16). The draft gets R1 (conjugation),
R2 (V/O), R4 (MLP-down Hadamard), matching the target's {r1,r2,r4} for w4a4.

Every FakeW4A4Linear counts its forward calls and its weight/act quant so a
generation run PROVES the draft is fake-W4A4 (not fp16, not unit-test-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import pure_r1_eagle as pr, rotation_aware as ra, spinquant_bridge as sb
from . import spinquant_draft as spd
from .spinquant_draft import SpinquantDraftPureR1Adapter

D = 4096
NH, HD, INTER = 32, 128, 11008


def _act_quantizer(bits=4):
    sb.add_spinquant_to_syspath()
    from utils import quant_utils
    q = quant_utils.ActQuantizer()
    q.configure(bits=bits, groupsize=-1, sym=False, clip_ratio=1.0)  # per-token asym
    return q


def _weight_fake_quant(w, bits=4):
    """RTN per-channel symmetric fake quant with MSE clip search (target policy)."""
    sb.add_spinquant_to_syspath()
    from utils import quant_utils
    q = quant_utils.WeightQuantizer()
    q.configure(bits=bits, perchannel=True, sym=True, mse=True)
    q.find_params(w)
    deq, _q, _s = q.fake_quantize(w)          # (dequant, int, scale)
    return deq.to(w.dtype)


# quantization policy strings (shared W4A4 / W8A8)
WEIGHT_SCHEME = "per_channel_symmetric_RTN_MSEclip"
ACT_SCHEME = "per_token_asymmetric"


class FakeW4A4Linear(nn.Module):
    """weight fake-quant (static RTN) + per-forward activation fake-quant, then
    fp16 matmul of the dequantized operands. online_had applies the R4 Hadamard
    to the input first (down_proj only)."""

    def __init__(self, weight, bias, name, online_had=False, had_K=None, K=None,
                 quant_weight=True, quant_act=True, w_bits=4, a_bits=4):
        super().__init__()
        self.name = name
        self.quant_weight = quant_weight
        self.quant_act = quant_act
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.register_buffer("w_fake", _weight_fake_quant(weight.data, w_bits)
                             if quant_weight else weight.data.clone())
        self.register_buffer("bias", bias.data.clone() if bias is not None else None)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.online_had = online_had
        if online_had:
            self.register_buffer("had_K", had_K)
            self.K = K
        self.aq = _act_quantizer(a_bits) if quant_act else None
        # proof / trace state
        self.n_forward = 0
        self.n_weight_quant = 1 if quant_weight else 0
        self.n_act_quant = 0
        self.last = {}
        self.ctx_fn = None                    # () -> (prompt_id, cycle, depth)
        self.trace = []                       # bounded per-forward hook trace

    @property
    def weight(self):                          # device-discovery shim for EAGLE
        return self.w_fake

    def forward(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.online_had:
            sb.add_spinquant_to_syspath()
            from utils import hadamard_utils
            x2 = hadamard_utils.matmul_hadU_cuda(x2, self.had_K, self.K)
        in_absmax = float(x2.detach().abs().max())
        if self.aq is not None:
            self.aq.find_params(x2)            # dynamic per-token
            xq = self.aq(x2)                   # fake-quant activation
            self.n_act_quant += 1
        else:
            xq = x2
        y = torch.nn.functional.linear(xq, self.w_fake, self.bias)
        self.n_forward += 1
        self.last = dict(input_shape=list(x2.shape), output_shape=list(y.shape),
                         input_dtype=str(x2.dtype), output_dtype=str(y.dtype),
                         input_absmax=round(in_absmax, 4),
                         output_absmax=round(float(y.detach().abs().max()), 4))
        if len(self.trace) < 60:
            pid, cyc, dep = self.ctx_fn() if self.ctx_fn else (-1, -1, -1)
            self.trace.append(dict(
                prompt_id=pid, verification_cycle_id=cyc, tree_depth=dep,
                module_name=self.name,
                fake_weight_quant_called=bool(self.quant_weight),
                fake_activation_quant_called=bool(self.quant_act),
                weight_bits=self.w_bits, activation_bits=self.a_bits, **self.last,
                notes="R4 online had" if self.online_had else ""))
        return y.reshape(*shp, self.out_features)


def build_spinquant_w4a4_draft_state(sd, R1, gamma, r2_seed=0,
                                     r2_override=None):
    """R1-conjugate (pure-R1) + R2 (V/O per-head) + R4 (MLP-down Hadamard fold).
    Returns (state_dict, meta) where meta has r2/r4 flags + had_K/K for R4.
    r2_override: composed [HD,HD] rotation (e.g. learned draft-aware R6 =
    R2_base @ C(B)) replacing the seeded baseline R2; folded identically."""
    conv = pr.build_pure_r1_draft_state(sd, R1, gamma)          # R1 conjugation
    p = "layers.0."
    # R2: conjugate v_proj/o_proj (weights) by the shared [128,128] R2
    if r2_override is not None:
        R2 = r2_override.double()
    else:
        g = torch.Generator().manual_seed(r2_seed)
        R2 = torch.linalg.qr(torch.randn(HD, HD, generator=g,
                                         dtype=torch.float64))[0]
    v = conv[p + "self_attn.v_proj.weight"].double()
    o = conv[p + "self_attn.o_proj.weight"].double()
    v2, o2 = spd.conjugate_v_o(v, o, R2)
    conv[p + "self_attn.v_proj.weight"] = v2.to(conv[p + "self_attn.v_proj.weight"].dtype)
    conv[p + "self_attn.o_proj.weight"] = o2.to(conv[p + "self_attn.o_proj.weight"].dtype)
    # R4: fold the down_proj Hadamard into the weight; online Hadamard on input
    sb.add_spinquant_to_syspath()
    from utils import hadamard_utils
    lin = nn.Linear(INTER, D, bias=False)
    lin.weight.data = conv[p + "mlp.down_proj.weight"].clone()
    hadamard_utils.apply_exact_had_to_linear(lin, had_dim=-1, output=False)
    conv[p + "mlp.down_proj.weight"] = lin.weight.data
    had_K, K = hadamard_utils.get_hadK(INTER)
    return conv, dict(r2_applied=True, r4_applied=True, had_K=had_K, K=K)


REQUIRED = ["fc", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj",
            "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj"]


class FakeW4A4DraftAdapter(SpinquantDraftPureR1Adapter):
    """Pure-R1 draft with R1/R2/R4 in the generation path AND fake W4A4 on all 8
    draft linears (fc, q/k/v/o, gate/up/down). Head W_lm@R1 stays fp16. Runtime
    h@R1 external-only. Coverage + hook trace collected from real generation."""
    name = "fake_w4a4_pure_r1"
    r1_only = False
    quant_weight = True
    quant_act = True
    w_bits = 4
    a_bits = 4

    def configure_fq(self, r1_only=False, quant_weight=True, quant_act=True,
                     w_bits=4, a_bits=4):
        self.r1_only = r1_only
        self.quant_weight = quant_weight
        self.quant_act = quant_act
        self.w_bits = w_bits
        self.a_bits = a_bits
        return self

    def install(self):
        ea = self.ea_layer
        dev = self.device
        self._sd_backup = {k: v.detach().cpu().clone() for k, v in ea.state_dict().items()}
        if self.r1_only:
            conv = pr.build_pure_r1_draft_state(
                {k: v.detach().cpu() for k, v in ea.state_dict().items()},
                self.R1.cpu(), self.gamma.cpu())
            meta = dict(r2_applied=False, r4_applied=False, had_K=None, K=None)
        else:
            conv, meta = build_spinquant_w4a4_draft_state(
                {k: v.detach().cpu() for k, v in ea.state_dict().items()},
                self.R1.cpu(), self.gamma.cpu())
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype) for k, v in conv.items()},
                           strict=True)
        ea.to(dev)
        self.r2_applied = meta["r2_applied"]
        self.r4_applied = meta["r4_applied"]
        had_K = meta["had_K"].to(dev) if meta["had_K"] is not None else None

        # replace the 8 required linears with FakeW4A4Linear
        self.fq_modules = {}
        self._replaced = []                    # (parent, attr, original_module)
        def replace(parent, attr, name, online=False):
            lin = getattr(parent, attr).to(dev)
            self._replaced.append((parent, attr, lin))
            fq = FakeW4A4Linear(lin.weight, getattr(lin, "bias", None), name,
                                online_had=online, had_K=had_K if online else None,
                                K=meta["K"] if online else None,
                                quant_weight=self.quant_weight,
                                quant_act=self.quant_act,
                                w_bits=self.w_bits, a_bits=self.a_bits).to(dev)
            setattr(parent, attr, fq)
            self.fq_modules[name] = fq
        replace(ea, "fc", "fc")
        attn = ea.layers[0].self_attn
        for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
            replace(attn, pn, f"layers.0.self_attn.{pn}")
        mlp = ea.layers[0].mlp
        for pn in ("gate_proj", "up_proj"):
            replace(mlp, pn, f"layers.0.mlp.{pn}")
        replace(mlp, "down_proj", "layers.0.mlp.down_proj",
                online=bool(self.r4_applied))       # R4 online only if R4 folded
        # give every fake linear the generation context for the hook trace
        _self = self
        for m in self.fq_modules.values():
            m.ctx_fn = lambda s=_self: (s._prompt_id, s._cycle, s._fc_idx)

        # runtime h@R1 + head substitution + fc-depth trace (parent logic, but
        # our fc is now FakeW4A4Linear — patch its .forward for the depth trace)
        adapter = self
        self._orig_fc_forward = ea.fc.forward
        fc_mod = ea.fc

        def fc_forward(z, *a, **k):
            if adapter.rotate_recurrent and adapter._fc_idx > 0:
                z = torch.cat([z[..., :D],
                               (z[..., D:].to(torch.float32) @ adapter.R1).to(z.dtype)], -1)
            if adapter.trace and adapter._prompt_id >= 0 and len(adapter.forward_trace) < 400:
                adapter.forward_trace.append(dict(
                    prompt_id=adapter._prompt_id, verification_cycle_id=adapter._cycle,
                    tree_depth=adapter._fc_idx,
                    is_external_target_forward=bool(adapter._fc_idx == 0),
                    is_recurrent_draft_forward=bool(adapter._fc_idx > 0),
                    input_hidden_source=("target_h" if adapter._fc_idx == 0 else "draft_f_R"),
                    runtime_transform=("@R1" if adapter._fc_idx == 0 and adapter.rotate_external else "none"),
                    expected_basis=("h_R" if adapter._fc_idx == 0 else "f_R"),
                    hidden_norm=round(float(z[..., D:].detach().float().norm()), 4)))
            adapter._fc_idx += 1
            return adapter._orig_fc_forward(z, *a, **k)
        fc_mod.forward = fc_forward

        assert self._orig_topk is None
        self._orig_topk = ea.topK_genrate

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            adapter._fc_idx = 0
            do_rot = adapter.rotate_external and (
                not adapter.once_per_prompt or not adapter._did_rotate_once)
            hs = hidden_states
            if do_rot:
                hs = (hidden_states.to(torch.float32) @ adapter.R1).to(hidden_states.dtype)
                adapter._did_rotate_once = True
            adapter._cycle += 1
            return adapter._orig_topk(hs, input_ids, adapter.head, logits_processor, *a, **k)
        ea.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None
        if hasattr(self, "_orig_fc_forward"):
            # fc module is about to be replaced; drop the forward patch
            del self._orig_fc_forward
        # restore the ORIGINAL nn.Linear modules (FakeW4A4Linear changed the
        # module structure, so a bare load_state_dict would fail), then load the
        # pristine draft weights.
        for parent, attr, orig in getattr(self, "_replaced", []):
            setattr(parent, attr, orig)
        self.ea_layer.load_state_dict(self._sd_backup, strict=True)
        self.ea_layer.to(self.device)
        del self._sd_backup

    def coverage(self):
        rows = []
        for name in REQUIRED:
            m = self.fq_modules.get(name)
            called = bool(m and m.n_forward > 0)
            rows.append(dict(
                module_name=f"ea_layer.{name}", module_type="FakeW4A4Linear",
                weight_shape=str(list(m.w_fake.shape)) if m else "",
                required_for_w8a8=True, required_for_w4a4=True,
                fake_weight_quant_applied=bool(m and m.n_weight_quant > 0),
                fake_activation_quant_applied=bool(m and m.n_act_quant > 0),
                fake_quant_called_during_generation=called,
                weight_bits=(m.w_bits if m else None),
                activation_bits=(m.a_bits if m else None),
                weight_quant_scheme=WEIGHT_SCHEME, activation_quant_scheme=ACT_SCHEME,
                r1_applied=True,
                r2_applied=bool(self.r2_applied and "self_attn.v_proj" in name or
                                self.r2_applied and "self_attn.o_proj" in name),
                r3_applied=False,   # N/A (k_bits=16), consistent with target
                r4_applied=bool(self.r4_applied and "down_proj" in name),
                dtype_before="fp16", dtype_after="fp16(fake-quant)",
                notes=("R4 online Hadamard" if m and m.online_had else "")))
        return rows

    def all_required_covered(self):
        return all(r["fake_weight_quant_applied"] and r["fake_activation_quant_applied"]
                   and r["fake_quant_called_during_generation"] for r in self.coverage())


class FakeW4A4OrigDraftAdapter(SpinquantDraftPureR1Adapter):
    """CONTROL: fake-W4A4 the ORIGINAL (un-conjugated, no-R1) draft on an fp16
    target. No runtime h@R1, original head. Isolates whether 4-bit ALONE kills
    the tiny draft (vs the R1 conjugation)."""
    name = "fake_w4a4_orig_draft"
    quant_weight = True
    quant_act = True
    w_bits = 4
    a_bits = 4

    def configure_fq(self, quant_weight=True, quant_act=True, w_bits=4, a_bits=4, **_):
        self.quant_weight = quant_weight
        self.quant_act = quant_act
        self.w_bits = w_bits
        self.a_bits = a_bits
        return self

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16, **k):
        super().__init__(ea_model, stash, device, dtype)
        from .rotation_interface import build_original_head
        self.head = build_original_head(stash["lm_head_weight"], device, dtype)

    def install(self):
        ea = self.ea_layer
        dev = self.device
        self.r2_applied = False
        self.r4_applied = False
        self._sd_backup = {k: v.detach().cpu().clone() for k, v in ea.state_dict().items()}
        self.fq_modules, self._replaced = {}, []
        def replace(parent, attr, name):
            lin = getattr(parent, attr).to(dev)
            self._replaced.append((parent, attr, lin))
            m = FakeW4A4Linear(lin.weight, getattr(lin, "bias", None), name,
                               quant_weight=self.quant_weight, quant_act=self.quant_act,
                               w_bits=self.w_bits, a_bits=self.a_bits).to(dev)
            setattr(parent, attr, m); self.fq_modules[name] = m
        replace(ea, "fc", "fc")
        for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
            replace(ea.layers[0].self_attn, pn, f"layers.0.self_attn.{pn}")
        for pn in ("gate_proj", "up_proj", "down_proj"):
            replace(ea.layers[0].mlp, pn, f"layers.0.mlp.{pn}")
        adapter = self
        self._orig_topk = ea.topK_genrate
        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            return adapter._orig_topk(hidden_states, input_ids, adapter.head,
                                      logits_processor, *a, **k)   # NO h@R1
        ea.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None
        for parent, attr, orig in self._replaced:
            setattr(parent, attr, orig)
        self.ea_layer.load_state_dict(self._sd_backup, strict=True)
        self.ea_layer.to(self.device); del self._sd_backup

    def coverage(self):
        rows = []
        for name in REQUIRED:
            m = self.fq_modules.get(name)
            rows.append(dict(
                module_name=f"ea_layer.{name}", module_type="FakeW4A4Linear",
                weight_shape=str(list(m.w_fake.shape)) if m else "",
                required_for_w8a8=True, required_for_w4a4=True,
                fake_weight_quant_applied=bool(m and m.n_weight_quant > 0),
                fake_activation_quant_applied=bool(m and m.n_act_quant > 0),
                fake_quant_called_during_generation=bool(m and m.n_forward > 0),
                weight_bits=(m.w_bits if m else None),
                activation_bits=(m.a_bits if m else None),
                weight_quant_scheme=WEIGHT_SCHEME, activation_quant_scheme=ACT_SCHEME,
                r1_applied=False, r2_applied=False, r3_applied=False, r4_applied=False,
                dtype_before="fp16", dtype_after="fp16(fake-quant)",
                notes="CONTROL: original un-conjugated draft, no rotations"))
        return rows

    def all_required_covered(self):
        return all(r["fake_quant_called_during_generation"] for r in self.coverage())
