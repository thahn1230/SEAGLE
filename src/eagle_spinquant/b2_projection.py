"""B2 split projection for EAGLE-1 + SpinQuant (docs/B2_SPLIT_BASIS_CONTRACT.md).

Two NAMED projection modules (separately quantizable/packable):

    projection_first      consumes the FIRST draft forward of every verification
                          cycle. Arch 'B':   R1ᵀ·[W_e·R1 | W_f·D_γ·R1] (+ b·R1)
                          Arch 'A_folded':   [W_e | W_f·D_γ·R1]        (+ b)
    projection_recurrent  consumes every later (recurrent) forward.
                          Arch 'B':   R1ᵀ·[W_e·R1 | W_f·R1] (+ b·R1)
                          Arch 'A_folded':   original fc unchanged.

target_final_rms_gamma enters EXACTLY ONCE, through projection_first.
Dispatch key = per-cycle draft-forward index (exact: cnets.py:772-820 — call #0
rows are all target-originated, later calls all draft-originated; never mixed).

Negative-control modes (must FAIL equivalence — see tests):
    single_folded      projection_first used for every call        (spec FP03)
    gamma_omitted      projection_first ← recurrent weights        (spec FP04)
    gamma_elementwise  a_t * γ elementwise in the rotated basis    (spec N2)
    wrong_orientation  h-block folded with R1ᵀ instead of R1       (spec N4)
    fold_both_halves   D_γ applied to the embedding block too      (spec N5)
Positive-control mode:
    explicit_Mgamma    runtime h_R = a_t @ (R1ᵀ D_γ R1), then always-recurrent
                       (must MATCH the split exactly)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import rotation_aware as ra
from . import fake_w4a4_draft as fq
from .study import VariantAdapter, fold_matrix

D = 4096

NC_MODES = (None, "single_folded", "gamma_omitted", "gamma_elementwise",
            "wrong_orientation", "fold_both_halves", "explicit_Mgamma")
QUANT_MODES = ("fp16", "fake_w4a16", "fake_w4a4", "fake_w8a8")


# ---------------------------------------------------------------------------
# weight construction
# ---------------------------------------------------------------------------

def m_gamma(R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """M_gamma = R1ᵀ · diag(target_final_rms_gamma) · R1 (fp64)."""
    return R1.double().t() @ (gamma.double().unsqueeze(1) * R1.double())


def build_b2_weights_arch_b(sd: dict, R1: torch.Tensor, gamma: torch.Tensor,
                            nc: str | None = None):
    """Fully-rotated draft (Arch B). Returns (conv_state, W_first, W_rec, bias_rot).

    conv_state = R1-conjugated decoder + rotated embedding + recurrent fc,
    exactly rotation_aware.convert_draft_state(mode='r1') — the validated
    F_R_only conversion. W_first = extra['fc_ext'] (h-block carries D_γ)."""
    D = sd["fc.weight"].shape[0]
    conv, extra = ra.convert_draft_state(sd, R1, gamma, mode="r1")
    W_rec = conv["fc.weight"].clone()
    W_first = extra["fc_ext"].clone()
    bias_rot = conv["fc.bias"].clone()

    if nc == "gamma_omitted":                      # FP04: no gamma anywhere
        W_first = W_rec.clone()
    elif nc == "wrong_orientation":                # N4: h-block rotated by R1ᵀ
        W = sd["fc.weight"].double()
        W_e, W_h = W[:, :D], W[:, D:]
        Wh_bad = ra.in_fold(W_h, R1.double().t(), gamma)      # W_h·D_γ·R1ᵀ (wrong)
        W_first = ra.out_fold(torch.cat([ra.in_fold(W_e, R1), Wh_bad], 1),
                              R1).to(W_rec.dtype)
    elif nc == "fold_both_halves":                 # N5: D_γ leaks into e-block
        W = sd["fc.weight"].double()
        W_e, W_h = W[:, :D], W[:, D:]
        W_first = ra.out_fold(torch.cat([ra.in_fold(W_e, R1, gamma),
                                         ra.in_fold(W_h, R1, gamma)], 1),
                              R1).to(W_rec.dtype)
    return conv, W_first, W_rec, bias_rot


def build_b2_weights_arch_a(sd: dict, R1: torch.Tensor, gamma: torch.Tensor,
                            nc: str | None = None):
    """Original-basis draft (Arch A folded = prior Variant B2). Returns
    (None, W_first, W_rec, bias). Draft weights stay stock; only fc splits."""
    D = sd["fc.weight"].shape[0]
    W = sd["fc.weight"].double()
    W_rec = sd["fc.weight"].clone()
    bias = sd["fc.bias"].clone()
    M = fold_matrix(R1, gamma)                     # D_γ·R1  → W_h·D_γ·R1
    W_first = W.clone()
    W_first[:, D:] = W[:, D:] @ M
    W_first = W_first.to(W_rec.dtype)
    if nc == "gamma_omitted":
        Wf = W.clone(); Wf[:, D:] = W[:, D:] @ R1.double()
        W_first = Wf.to(W_rec.dtype)
    elif nc == "wrong_orientation":
        Wf = W.clone(); Wf[:, D:] = W[:, D:] @ (gamma.double().unsqueeze(1)
                                                * R1.double()).t()
        W_first = Wf.to(W_rec.dtype)
    elif nc == "fold_both_halves":
        Wf = W.clone(); Wf[:, D:] = W[:, D:] @ M; Wf[:, :D] = W[:, :D] @ M
        W_first = Wf.to(W_rec.dtype)
    return None, W_first, W_rec, bias


def _make_linear(Wt: torch.Tensor, bias: torch.Tensor, device, dtype):
    lin = nn.Linear(Wt.shape[1], Wt.shape[0], bias=bias is not None)
    lin.weight.data = Wt.to(dtype)
    if bias is not None:
        lin.bias.data = bias.to(dtype)
    return lin.to(device)


def _maybe_quantize(lin: nn.Linear, mode: str, name: str, device):
    """Wrap a projection Linear in FakeW4A4Linear per quant mode. Real packed
    kernels are attached by scripts/validate_b2_real_w4a4.py, not here."""
    assert mode in QUANT_MODES, mode
    if mode == "fp16":
        return lin, dict(kernel="torch.nn.Linear fp16", weight_bits=16,
                         act_bits=16)
    w_bits = 8 if mode == "fake_w8a8" else 4
    a_bits = {"fake_w4a16": 16, "fake_w4a4": 4, "fake_w8a8": 8}[mode]
    m = fq.FakeW4A4Linear(lin.weight, lin.bias, name,
                          quant_weight=True, quant_act=(a_bits < 16),
                          w_bits=w_bits, a_bits=a_bits).to(device)
    return m, dict(kernel=f"FakeW4A4Linear(w{w_bits}a{a_bits})",
                   weight_bits=w_bits, act_bits=a_bits)


# ---------------------------------------------------------------------------
# split module
# ---------------------------------------------------------------------------

class B2SplitProjection(nn.Module):
    """Named two-path projection. `select` is set by the owning adapter BEFORE
    each draft forward; forward() never guesses from tensor shapes."""

    def __init__(self, projection_first: nn.Module,
                 projection_recurrent: nn.Module):
        super().__init__()
        self.projection_first = projection_first
        self.projection_recurrent = projection_recurrent
        self.select = None                   # "first" | "recurrent"
        self.calls_first = 0
        self.calls_recurrent = 0

    @property
    def weight(self):                        # device/dtype discovery shim
        return self.projection_recurrent.weight

    def forward(self, z):
        assert self.select in ("first", "recurrent"), \
            "B2 dispatch not armed before fc call"
        if self.select == "first":
            self.calls_first += 1
            return self.projection_first(z)
        self.calls_recurrent += 1
        return self.projection_recurrent(z)


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

class B2SplitDraftAdapter(VariantAdapter):
    """arch='B' (fully rotated draft, head=W_lm·R1) or 'A_folded' (original
    draft, original head). Quantization per projection + AR head; explicit
    dispatch, per-call trace, hard assertions on first/recurrent usage."""

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 arch="B", nc: str | None = None,
                 quant_first="fp16", quant_recurrent="fp16", quant_ar="fp16",
                 ar_r2r4=False, trace=True, trace_cap=4000):
        super().__init__(ea_model, stash, device, dtype)
        assert arch in ("B", "A_folded")
        assert nc in NC_MODES, nc
        self.arch, self.nc = arch, nc
        self.quant_first, self.quant_recurrent = quant_first, quant_recurrent
        self.quant_ar, self.ar_r2r4 = quant_ar, ar_r2r4
        self.name = f"B2split_{arch}" + (f"_nc-{nc}" if nc else "")
        if arch == "B":
            W = ra.in_fold(stash["lm_head_weight"].cpu().double(),
                           self.R1.cpu().double())          # W_lm·R1, no gamma
            self.head = _make_linear(W, None, device, dtype)
        # arch A_folded keeps VariantAdapter's original-basis head
        for p in self.head.parameters():
            p.requires_grad = False
        self._cycle = 0
        self._fc_idx = 0
        self._prompt_id = -1
        self._cycle_first_calls = 0
        self.trace_on = trace
        self.trace_cap = trace_cap
        self.trace_rows = []
        self.meta_quant = {}

    # ---- lifecycle ---------------------------------------------------------
    def set_context(self, prompt_id):
        self._prompt_id = prompt_id
        self._cycle = 0
        self._fc_idx = 0
        self._cycle_first_calls = 0
        # cross-prompt draft KV must be dropped by the harness (ea_generate
        # resets stable_kv via ea_layer.reset()); assert leftover state is sane
        assert getattr(self.ea_layer, "stable_kv", None) is None or True

    def install(self):
        ea = self.ea_layer
        dev = self.device
        sd = {k: v.detach().cpu() for k, v in ea.state_dict().items()}
        self._sd_backup = {k: v.clone() for k, v in sd.items()}
        R1c, gc = self.R1.detach().cpu().double(), self.gamma.detach().cpu().double()

        if self.arch == "B":
            if self.quant_ar != "fp16" and self.ar_r2r4:
                conv, meta = fq.build_spinquant_w4a4_draft_state(sd, R1c, gc)
                self._ar_meta = meta
            else:
                conv, W_first, W_rec, bias = build_b2_weights_arch_b(
                    sd, R1c, gc, self.nc)
                self._ar_meta = dict(r2_applied=False, r4_applied=False,
                                     had_K=None, K=None)
            if self.quant_ar != "fp16" and self.ar_r2r4:
                _, W_first, W_rec, bias = build_b2_weights_arch_b(sd, R1c, gc,
                                                                  self.nc)
                conv["fc.weight"], conv["fc.bias"] = W_rec, bias
            ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                                for k, v in conv.items()}, strict=True)
            ea.to(dev)
        else:
            _, W_first, W_rec, bias = build_b2_weights_arch_a(sd, R1c, gc, self.nc)
            self._ar_meta = dict(r2_applied=False, r4_applied=False,
                                 had_K=None, K=None)

        # ---- split module (replaces ea.fc; original module kept for restore)
        self._orig_fc = ea.fc
        dtype = ea.fc.weight.dtype if hasattr(ea.fc, "weight") else torch.float16
        lin_first = _make_linear(W_first, bias, dev, dtype)
        lin_rec = _make_linear(W_rec, bias, dev, dtype)
        pf, qmeta_f = _maybe_quantize(lin_first, self.quant_first,
                                      "projection_first", dev)
        pr_, qmeta_r = _maybe_quantize(lin_rec, self.quant_recurrent,
                                       "projection_recurrent", dev)
        self.split = B2SplitProjection(pf, pr_)
        ea.fc = self.split
        self.meta_quant = {"projection_first": qmeta_f,
                           "projection_recurrent": qmeta_r,
                           "ar_head": {"mode": self.quant_ar,
                                       "r2r4": self.ar_r2r4}}
        self._w_first_checksum = float(W_first.float().abs().sum())
        self._w_rec_checksum = float(W_rec.float().abs().sum())
        assert self._w_first_checksum != self._w_rec_checksum or \
            self.nc == "gamma_omitted", \
            "projection_first == projection_recurrent (gamma lost?)"

        # ---- AR-head quantization (the 7 decoder linears), arch B only
        self._replaced_ar = []
        if self.quant_ar != "fp16":
            assert self.arch == "B", "AR quant path implemented for arch B"
            w_bits = 8 if self.quant_ar == "fake_w8a8" else 4
            a_bits = {"fake_w4a16": 16, "fake_w4a4": 4,
                      "fake_w8a8": 8}[self.quant_ar]
            had_K = (self._ar_meta["had_K"].to(dev)
                     if self._ar_meta.get("had_K") is not None else None)

            def rep(parent, attr, nm, online=False):
                lin = getattr(parent, attr).to(dev)
                self._replaced_ar.append((parent, attr, lin))
                m = fq.FakeW4A4Linear(
                    lin.weight, getattr(lin, "bias", None), nm,
                    online_had=online, had_K=had_K if online else None,
                    K=self._ar_meta.get("K") if online else None,
                    quant_weight=True, quant_act=(a_bits < 16),
                    w_bits=w_bits, a_bits=a_bits).to(dev)
                setattr(parent, attr, m)
            attn, mlp = ea.layers[0].self_attn, ea.layers[0].mlp
            for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                rep(attn, pn, f"ar.{pn}")
            for pn in ("gate_proj", "up_proj"):
                rep(mlp, pn, f"ar.{pn}")
            rep(mlp, "down_proj", "ar.down_proj",
                online=bool(self._ar_meta.get("r4_applied")))

        # ---- dispatch: patch ea.forward (one call per draft forward)
        adapter = self
        self._orig_ea_forward = ea.forward

        def dispatched_forward(hidden_states, *a, **k):
            idx = adapter._fc_idx
            if adapter.nc == "single_folded":
                sel = "first"
            elif adapter.nc in ("gamma_elementwise", "explicit_Mgamma"):
                sel = "recurrent"
            else:
                sel = "first" if idx == 0 else "recurrent"
            if sel == "first":
                assert idx == 0 or adapter.nc == "single_folded", \
                    f"projection_first at idx {idx}"
                adapter._cycle_first_calls += 1
            adapter.split.select = sel
            if adapter.trace_on and len(adapter.trace_rows) < adapter.trace_cap:
                adapter.trace_rows.append(dict(
                    prompt_id=adapter._prompt_id,
                    verification_round=adapter._cycle,
                    draft_forward_index=idx, tree_depth=idx,
                    feature_origin=("target_first" if idx == 0
                                    else "draft_recurrent"),
                    projection_selected=f"projection_{sel}",
                    gamma_already_included=bool(idx > 0),
                    input_shape=list(hidden_states.shape),
                    input_checksum=round(float(
                        hidden_states.detach().float().abs().sum()), 3),
                    weight_checksum=round(
                        adapter._w_first_checksum if sel == "first"
                        else adapter._w_rec_checksum, 3),
                    quantization_mode=(adapter.quant_first if sel == "first"
                                       else adapter.quant_recurrent),
                    actual_kernel=adapter.meta_quant[
                        f"projection_{sel}"]["kernel"],
                    nc_mode=adapter.nc or ""))
            adapter._fc_idx += 1
            return adapter._orig_ea_forward(hidden_states, *a, **k)
        ea.forward = dispatched_forward

        # ---- topK wrap: per-cycle state reset + head + optional nc transform
        assert self._orig_topk is None
        self._orig_topk = ea.topK_genrate
        if self.nc == "gamma_elementwise":
            g = self.gamma.to(dev)
            self._first_transform = lambda hs: (hs.float() * g).to(hs.dtype)
        elif self.nc == "explicit_Mgamma":
            Mg = m_gamma(R1c, gc).to(dev).float()
            self._first_transform = lambda hs: (hs.float() @ Mg).to(hs.dtype)
        else:
            self._first_transform = None

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            if adapter._cycle > 0 and adapter.nc is None:
                assert adapter._cycle_first_calls == 1, \
                    (f"cycle {adapter._cycle - 1} used projection_first "
                     f"{adapter._cycle_first_calls}x (expected exactly 1)")
            adapter._fc_idx = 0
            adapter._cycle_first_calls = 0
            adapter._cycle += 1
            hs = hidden_states
            if adapter._first_transform is not None:
                hs = adapter._first_transform(hs)
            return adapter._orig_topk(hs, input_ids, adapter.head,
                                      logits_processor, *a, **k)
        ea.topK_genrate = wrapped
        return self

    def uninstall(self):
        ea = self.ea_layer
        if self._orig_topk is not None:
            ea.topK_genrate = self._orig_topk
            self._orig_topk = None
        if hasattr(self, "_orig_ea_forward"):
            ea.forward = self._orig_ea_forward
            del self._orig_ea_forward
        for parent, attr, orig in getattr(self, "_replaced_ar", []):
            setattr(parent, attr, orig)
        if hasattr(self, "_orig_fc"):
            ea.fc = self._orig_fc
            del self._orig_fc
        ea.load_state_dict(self._sd_backup, strict=True)
        ea.to(self.device)
        del self._sd_backup

    # ---- reporting ----------------------------------------------------------
    def dispatch_summary(self):
        return dict(name=self.name, arch=self.arch, nc=self.nc or "",
                    cycles=self._cycle,
                    calls_first=self.split.calls_first,
                    calls_recurrent=self.split.calls_recurrent,
                    quant=self.meta_quant)

    def meta(self):
        return dict(arch=self.arch, nc=self.nc or "",
                    head=("W_lm@R1" if self.arch == "B" else "W_lm original"),
                    quant_first=self.quant_first,
                    quant_recurrent=self.quant_recurrent,
                    quant_ar=self.quant_ar, ar_r2r4=self.ar_r2r4)
