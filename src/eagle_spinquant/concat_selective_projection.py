"""Concat-selective rotation architecture (docs/CONCAT_SELECTIVE_ARCHITECTURE.md).

Defining properties (per the supplied diagram):
  1. draft embedding stays ORIGINAL (unrotated) — e, never e·R1
  2. concat(e, hidden) as in stock EAGLE ([embed | feature], cnets.py:593)
  3. FIRST forward: R1ᵀ then gamma applied ONLY to the hidden slice
  4. the ORIGINAL semantic projection is applied
  5. R1 applied to the COMPLETE projection output (post 2D→D)
  6. recurrent forwards: gamma NOT applied again
  7. recurrent hidden arrives already R1-rotated (r_d = h_d·R1) — absorbed as W_h·R1
  8. the embedding slice never receives R1ᵀ / gamma / any hidden transform

Folded pre-R weights (row-vector, W [out,in]; bias stays ORIGINAL b):

    projection_first_preR     = [W_e | W_h·D_γ·R1]
    projection_recurrent_preR = [W_e | W_h·R1]
    runtime: y_R = projection_*_preR(concat(e, ·)) @ R1     (explicit output R1)

Exact-arithmetic equivalence to stock EAGLE:
    first:  e·W_eᵀ + a·(W_h D_γ R1)ᵀ + b = e·W_eᵀ + n·D_γ·W_hᵀ + b = y  → y·R1
    recur:  r_d·(W_h R1)ᵀ = h_d·W_hᵀ                                → y·R1

Negative-control modes (must fail; never mixed into the primary result):
    embedding_rotated    e→e·R1 fed to the untouched W_e block
    gamma_on_embedding   gamma applied to the e-slice too
    Rt_on_whole_concat   R1ᵀ applied to BOTH slices (explicit path)
    orig_PL_recurrent    recurrent hidden block left as W_h (no R1 absorption)
    first_for_recurrent / recurrent_for_first   projection swap
    no_output_R          post-projection R1 omitted
    R_before_PL          R1 applied to the (2D) input instead of the (D) output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rotation_aware as ra
from . import fake_w4a4_draft as fq
from .study import VariantAdapter, fold_matrix

NC_MODES = (None, "embedding_rotated", "gamma_on_embedding",
            "Rt_on_whole_concat", "orig_PL_recurrent", "first_for_recurrent",
            "recurrent_for_first", "no_output_R", "R_before_PL")
QUANT_MODES = ("fp16", "fake_w4a16", "fake_w4a4", "fake_w8a8")


# ---------------------------------------------------------------------------
# weight construction (pre-R folds; embedding block NEVER transformed)
# ---------------------------------------------------------------------------

def build_concat_selective_weights(sd: dict, R1: torch.Tensor,
                                   gamma: torch.Tensor,
                                   nc: str | None = None):
    """Returns (W_first_preR, W_rec_preR, bias_original). fp64 math."""
    D = sd["fc.weight"].shape[0]
    W = sd["fc.weight"].double()
    W_e, W_h = W[:, :D].clone(), W[:, D:].clone()
    R1 = R1.double()
    M = fold_matrix(R1, gamma.double())            # D_γ·R1
    W_first = torch.cat([W_e, W_h @ M], dim=1)     # [W_e | W_h D_γ R1]
    W_rec = torch.cat([W_e, W_h @ R1], dim=1)      # [W_e | W_h R1]
    if nc == "orig_PL_recurrent":                  # F5: no hidden-side absorption
        W_rec = torch.cat([W_e, W_h], dim=1)
    b = sd["fc.bias"].clone()                      # ORIGINAL bias (pre-R)
    dt = sd["fc.weight"].dtype
    return W_first.to(dt), W_rec.to(dt), b


# ---------------------------------------------------------------------------
# explicit reference paths (numerical ground truth; used by tests + F2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def concat_selective_explicit(z, fc_orig_weight, fc_orig_bias, R1, gamma,
                              mode, nc=None):
    """z = concat(e, hidden) with hidden = a (first) or r_d (recurrent).
    Applies R1ᵀ (+gamma iff first) ONLY to the hidden slice, then the ORIGINAL
    projection, then R1 on the complete output. fp32 internal."""
    D = fc_orig_weight.shape[0]
    e, hid = z[..., :D], z[..., D:]
    R1f = R1.to(z.device, torch.float32)
    h = hid.float() @ R1f.t()
    if mode == "first":
        h = h * gamma.to(z.device, torch.float32)
    if nc == "gamma_on_embedding":
        e = (e.float() * gamma.to(z.device, torch.float32)).to(e.dtype)
    if nc == "Rt_on_whole_concat":
        e = (e.float() @ R1f.t()).to(e.dtype)
    if nc == "embedding_rotated":
        e = (e.float() @ R1f).to(e.dtype)
    y = F.linear(torch.cat([e.float(), h], -1), fc_orig_weight.float(),
                 fc_orig_bias.float())
    if nc == "no_output_R":
        return y.to(z.dtype)
    return (y @ R1f).to(z.dtype)


class PostProjectionR1(nn.Module):
    """Explicit post-projection rotation y → y·R1 on the COMPLETE D-dim output.
    Dense fp32 GEMM (R1 is a materialized random-Hadamard matrix; recorded as
    'dense GEMM' — a fast-Hadamard variant would require the factored form)."""

    def __init__(self, R1):
        super().__init__()
        self.register_buffer("R1_f32", R1.detach().to(torch.float32).clone())
        self.n_calls = 0

    def forward(self, y):
        self.n_calls += 1
        return (y.to(self.R1_f32.dtype) @ self.R1_f32).to(y.dtype)


class ConcatSelectiveProjection(nn.Module):
    """Two pre-R projections + shared explicit post-projection R1.
    `select` is armed by the adapter before every draft forward."""

    def __init__(self, projection_first_preR, projection_recurrent_preR,
                 post_projection_R1, nc=None):
        super().__init__()
        self.projection_first_preR = projection_first_preR
        self.projection_recurrent_preR = projection_recurrent_preR
        self.post_projection_R1 = post_projection_R1
        self.nc = nc
        self.select = None
        self.calls_first = 0
        self.calls_recurrent = 0

    @property
    def weight(self):                              # device/dtype shim
        return self.projection_recurrent_preR.weight

    def forward(self, z):
        if self.select == "first":
            self.calls_first += 1
            if self.nc == "R_before_PL":           # NC: rotate the 2D input halves
                D = z.shape[-1] // 2
                R = self.post_projection_R1.R1_f32
                z = torch.cat([(z[..., :D].float() @ R).to(z.dtype),
                               (z[..., D:].float() @ R).to(z.dtype)], -1)
            y = self.projection_first_preR(z)
        elif self.select == "recurrent":
            self.calls_recurrent += 1
            y = self.projection_recurrent_preR(z)
        else:
            raise RuntimeError("concat-selective dispatch not armed")
        if self.nc == "no_output_R":
            return y
        return self.post_projection_R1(y)


def _make_linear(Wt, bias, device, dtype):
    lin = nn.Linear(Wt.shape[1], Wt.shape[0], bias=bias is not None)
    lin.weight.data = Wt.to(dtype)
    if bias is not None:
        lin.bias.data = bias.to(dtype)
    return lin.to(device)


def _maybe_quantize(lin, mode, name, device):
    assert mode in QUANT_MODES, mode
    if mode == "fp16":
        return lin, dict(kernel="torch.nn.Linear fp16", weight_bits=16, act_bits=16)
    w_bits = 8 if mode == "fake_w8a8" else 4
    a_bits = {"fake_w4a16": 16, "fake_w4a4": 4, "fake_w8a8": 8}[mode]
    m = fq.FakeW4A4Linear(lin.weight, lin.bias, name, quant_weight=True,
                          quant_act=(a_bits < 16), w_bits=w_bits,
                          a_bits=a_bits).to(device)
    return m, dict(kernel=f"FakeW4A4Linear(w{w_bits}a{a_bits})",
                   weight_bits=w_bits, act_bits=a_bits)


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

class ConcatSelectiveDraftAdapter(VariantAdapter):
    """Original-basis draft embedding + R1-conjugated decoder + concat-selective
    split projection with explicit post-projection R1. Head = W_lm·R1.

    variant='folded' (primary, diagram path) or 'explicit' (runtime-slice
    reference — must match folded exactly)."""

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 variant="folded", nc: str | None = None,
                 quant_first="fp16", quant_recurrent="fp16", quant_ar="fp16",
                 quant_embed="fp16", ar_r2r4=False, trace=True, trace_cap=4000):
        super().__init__(ea_model, stash, device, dtype)
        assert variant in ("folded", "explicit")
        assert nc in NC_MODES, nc
        if nc in ("Rt_on_whole_concat", "gamma_on_embedding"):
            assert variant == "explicit", f"nc={nc} is an explicit-path control"
        self.variant, self.nc = variant, nc
        self.quant_first, self.quant_recurrent = quant_first, quant_recurrent
        self.quant_ar, self.quant_embed = quant_ar, quant_embed
        self.ar_r2r4 = ar_r2r4
        self.name = f"concat_selective_{variant}" + (f"_nc-{nc}" if nc else "")
        W = ra.in_fold(stash["lm_head_weight"].cpu().double(),
                       self.R1.cpu().double())               # W_lm·R1
        head = nn.Linear(W.shape[1], W.shape[0], bias=False)
        head.weight.data = W.to(dtype)
        self.head = head.to(device=device, dtype=dtype)
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

    def set_context(self, prompt_id):
        self._prompt_id = prompt_id
        self._cycle = 0
        self._fc_idx = 0
        self._cycle_first_calls = 0

    def install(self):
        ea = self.ea_layer
        dev = self.device
        sd = {k: v.detach().cpu() for k, v in ea.state_dict().items()}
        self._sd_backup = {k: v.clone() for k, v in sd.items()}
        R1c = self.R1.detach().cpu().double()
        gc = self.gamma.detach().cpu().double()

        # ---- decoder: R1-conjugated (reuse validated conversion); embedding
        # and fc stay ORIGINAL in the loaded state (fc module replaced below)
        if self.quant_ar != "fp16" and self.ar_r2r4:
            conv, meta = fq.build_spinquant_w4a4_draft_state(sd, R1c, gc)
            self._ar_meta = meta
        else:
            conv, _extra = ra.convert_draft_state(sd, R1c, gc, mode="r1")
            self._ar_meta = dict(r2_applied=False, r4_applied=False,
                                 had_K=None, K=None)
        new_sd = {k: v.clone() for k, v in sd.items()}       # original everything
        for k, v in conv.items():
            if k.startswith("layers.0."):
                new_sd[k] = v                                 # rotated decoder only
        if self.nc == "embedding_rotated":                    # F4 control
            new_sd["embed_tokens.weight"] = ra.in_basis_rows(
                sd["embed_tokens.weight"], R1c).to(sd["embed_tokens.weight"].dtype)
        if self.quant_embed != "fp16":                        # embedding ablation
            w_bits = 8 if self.quant_embed == "fake_w8a8" else 4
            new_sd["embed_tokens.weight"] = fq._weight_fake_quant(
                new_sd["embed_tokens.weight"], w_bits)
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype) for k, v in new_sd.items()},
                           strict=True)
        ea.to(dev)
        emb_checksum_before = float(
            ea.embed_tokens.weight.detach().float().abs().sum())

        # ---- projection module
        W_first, W_rec, bias = build_concat_selective_weights(sd, R1c, gc, self.nc)
        self._orig_fc = ea.fc
        dtype = self._orig_fc.weight.dtype
        self._fc_orig_weight = sd["fc.weight"].to(dev, dtype)
        self._fc_orig_bias = sd["fc.bias"].to(dev, dtype)
        post_r1 = PostProjectionR1(self.R1).to(dev)
        if self.variant == "explicit":
            adapter = self

            class _Explicit(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.post_projection_R1 = post_r1
                    self.select = None
                    self.calls_first = 0
                    self.calls_recurrent = 0

                @property
                def weight(self):
                    return adapter._fc_orig_weight

                def forward(self, z):
                    assert self.select in ("first", "recurrent")
                    if self.select == "first":
                        self.calls_first += 1
                    else:
                        self.calls_recurrent += 1
                    self.post_projection_R1.n_calls += 1
                    return concat_selective_explicit(
                        z, adapter._fc_orig_weight, adapter._fc_orig_bias,
                        adapter.R1, adapter.gamma, self.select, adapter.nc)
            self.split = _Explicit()
            qmeta_f = qmeta_r = dict(kernel="explicit fp32 slice ops",
                                     weight_bits=16, act_bits=16)
        else:
            lin_f = _make_linear(W_first, bias, dev, dtype)
            lin_r = _make_linear(W_rec, bias, dev, dtype)
            pf, qmeta_f = _maybe_quantize(lin_f, self.quant_first,
                                          "projection_first_preR", dev)
            pr_, qmeta_r = _maybe_quantize(lin_r, self.quant_recurrent,
                                           "projection_recurrent_preR", dev)
            self.split = ConcatSelectiveProjection(pf, pr_, post_r1, self.nc)
        ea.fc = self.split
        self.meta_quant = {"projection_first_preR": qmeta_f,
                           "projection_recurrent_preR": qmeta_r,
                           "post_projection_R1": {"kernel": "dense fp32 GEMM"},
                           "ar_head": {"mode": self.quant_ar, "r2r4": self.ar_r2r4},
                           "embedding": {"mode": self.quant_embed,
                                         "basis": ("rotated(NC)" if
                                                   self.nc == "embedding_rotated"
                                                   else "original")}}

        # embedding-unchanged assertion (primary architecture only)
        if self.nc != "embedding_rotated" and self.quant_embed == "fp16":
            orig_sum = float(sd["embed_tokens.weight"].float().abs().sum())
            assert abs(emb_checksum_before - orig_sum) / orig_sum < 1e-3, \
                "embedding slice was modified — forbidden in primary architecture"

        # ---- AR-head quantization
        self._replaced_ar = []
        if self.quant_ar != "fp16":
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

        # ---- dispatch (same proven pattern as B2)
        adapter = self
        self._orig_ea_forward = ea.forward

        def dispatched_forward(hidden_states, *a, **k):
            idx = adapter._fc_idx
            if adapter.nc == "first_for_recurrent":
                sel = "first"
            elif adapter.nc == "recurrent_for_first":
                sel = "recurrent"
            else:
                sel = "first" if idx == 0 else "recurrent"
            if sel == "first" and adapter.nc != "first_for_recurrent":
                assert idx == 0, f"projection_first at idx {idx}"
                adapter._cycle_first_calls += 1
            adapter.split.select = sel
            if adapter.trace_on and len(adapter.trace_rows) < adapter.trace_cap:
                adapter.trace_rows.append(dict(
                    prompt_id=adapter._prompt_id,
                    verification_round=adapter._cycle,
                    draft_forward_index=idx, draft_depth=idx,
                    feature_origin=("target_first" if idx == 0
                                    else "draft_recurrent"),
                    selected_projection=f"projection_{sel}_preR",
                    embedding_basis="original",
                    hidden_basis=("a=nR1" if idx == 0 else "r_d=h_dR1"),
                    output_basis="rotated(post_projection_R1)",
                    gamma_applied=bool(sel == "first" and
                                       adapter.nc != "gamma_omitted"),
                    post_projection_R1_executed=bool(
                        adapter.nc != "no_output_R"),
                    quantization_mode=(adapter.quant_first if sel == "first"
                                       else adapter.quant_recurrent),
                    nc_mode=adapter.nc or ""))
            adapter._fc_idx += 1
            return adapter._orig_ea_forward(hidden_states, *a, **k)
        ea.forward = dispatched_forward

        assert self._orig_topk is None
        self._orig_topk = ea.topK_genrate

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            if adapter._cycle > 0 and adapter.nc is None:
                assert adapter._cycle_first_calls == 1, \
                    f"cycle used projection_first {adapter._cycle_first_calls}x"
            adapter._fc_idx = 0
            adapter._cycle_first_calls = 0
            adapter._cycle += 1
            return adapter._orig_topk(hidden_states, input_ids, adapter.head,
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

    def dispatch_summary(self):
        return dict(name=self.name, variant=self.variant, nc=self.nc or "",
                    cycles=self._cycle,
                    calls_first=self.split.calls_first,
                    calls_recurrent=self.split.calls_recurrent,
                    post_R1_calls=self.split.post_projection_R1.n_calls
                    if hasattr(self.split, "post_projection_R1") else -1,
                    quant=self.meta_quant)

    def meta(self):
        return dict(variant=self.variant, nc=self.nc or "",
                    embedding_basis="original", head="W_lm@R1",
                    quant_first=self.quant_first,
                    quant_recurrent=self.quant_recurrent,
                    quant_ar=self.quant_ar, quant_embed=self.quant_embed)
