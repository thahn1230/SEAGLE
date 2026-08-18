"""Strict SEAGLE-RT Stage-B: RAW no-fold draft quantization.

Quantizes the strict natively-trained draft's own modules IN PLACE with
the canonical SpinQuant-style quantizers (per-out-channel symmetric RTN
+ MSE clip weights; dynamic per-token asymmetric activations) while
preserving the pure native runtime:

  - NO basis change of any kind: no W_h·D_gamma·R1 / W_h·R1 folds, no
    embed_scale_alpha, no R2/R4, no draft rotation (Gate L).
  - The stock cnets forward and topK_genrate are untouched; the draft
    keeps consuming a_t directly and is scored by the deployed fused
    head.

Sites (8): fc, self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj of
layers[0]. Embedding table and scoring head stay fp16 by default
(canonical draft policy); optional embed-table weight quant for the
component audit. Site masks + per-site bitwidths support the spec §20
component-sensitivity audit (w16aX / wXa16 decomposition included via
the QUANT_BITS mode strings).
"""
import torch

from .concat_selective_projection import QUANT_BITS
from .fake_w4a4_draft import FakeW4A4Linear, _weight_fake_quant

SITES = ("fc", "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj")
D = 4096


def hadamard_4096(device, dtype=torch.float64):
    """Normalized 4096 Sylvester Hadamard (exact orthogonal)."""
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (D ** 0.5)


class RotatedFcInput(torch.nn.Module):
    """SRT_W4A4_HAD / _SQ interface-boundary rotation: applies R to the
    HIDDEN half of the fc input at runtime while the wrapped quantized
    fc holds W_h@R folded weights — function-preserving in exact
    arithmetic (x@R @ (W_h R)^T == x W_h^T); the quantizers therefore
    see the ROTATED activation and weight (energy mixing), with zero
    change to the draft's external interface (no restore semantics).
    Covers BOTH the first path (a_t) and the recurrent path (recycled
    f) because cnets routes both through this single fc."""

    def __init__(self, fq_fc, R):
        super().__init__()
        self.fq = fq_fc
        self.register_buffer("R", R.half())

    @property
    def weight(self):                       # device-discovery shim
        return self.fq.w_fake

    def forward(self, z):
        e, h = z[..., :D], z[..., D:]
        h = (h.half() @ self.R).to(z.dtype)
        return self.fq(torch.cat([e, h], dim=-1))


def apply_interface_rotation(ea_model, mode, R=None, r4_downproj=True):
    """Stage-B fixed/learned rotation arm on the RAW-quantized draft.

    Call INSTEAD of quantizing fc via apply_native_raw_quant: quantizes
    all 8 sites, but fc gets W_h@R folded (fp64 fold, then fp16
    RTN-quantized) + runtime input rotation, and down_proj gets the
    canonical exact R4 Hadamard fold + online Hadamard.
    R=None -> normalized Hadamard-4096 (SRT_W4A4_HAD). Pass a learned
    orthogonal R (fp64) for SRT_W4A4_SQ.
    """
    ea = ea_model.ea_layer if hasattr(ea_model, "ea_layer") else ea_model
    w_bits, a_bits = QUANT_BITS[mode]
    dev = ea.fc.weight.device
    if R is None:
        R = hadamard_4096(dev)
    R = R.to(dev, torch.float64)
    err = (R @ R.T - torch.eye(D, dtype=torch.float64,
                               device=dev)).abs().max()
    assert float(err) < 1e-9, f"R not orthogonal ({float(err)})"

    # fc: fold W_h @ R in fp64, then quantize; wrap with runtime R
    lin = ea.fc
    assert isinstance(lin, torch.nn.Linear), "fc already replaced"
    W = lin.weight.data.double()
    Wf = torch.cat([W[:, :D], W[:, D:] @ R], dim=1).half()
    fq = FakeW4A4Linear(Wf, lin.bias, "native_had.fc",
                        online_had=False, quant_weight=w_bits < 16,
                        quant_act=a_bits < 16,
                        w_bits=w_bits, a_bits=a_bits).to(dev)
    ea.fc = RotatedFcInput(fq, R)

    # AR sites: raw RTN; down_proj additionally gets canonical R4
    from . import spinquant_bridge as sb
    sb.add_spinquant_to_syspath()
    from utils import hadamard_utils
    man = apply_native_raw_quant(
        ea, mode, site_mask=[s for s in SITES
                             if s not in ("fc", "down_proj")])
    layer = ea.layers[0]
    dp = layer.mlp.down_proj
    if r4_downproj:
        import copy
        dp2 = copy.deepcopy(dp)
        hadamard_utils.apply_exact_had_to_linear(
            dp2, had_dim=-1, output=False)
        had_K, K = hadamard_utils.get_hadK(dp.in_features)
        fqd = FakeW4A4Linear(dp2.weight.data.half(), dp2.bias,
                             "native_had.down_proj", online_had=True,
                             had_K=(had_K.to(dev) if had_K is not None
                                    else None), K=K,
                             quant_weight=w_bits < 16,
                             quant_act=a_bits < 16,
                             w_bits=w_bits, a_bits=a_bits).to(dev)
    else:
        fqd = FakeW4A4Linear(dp.weight.data.half(), dp.bias,
                             "native_had.down_proj", online_had=False,
                             quant_weight=w_bits < 16,
                             quant_act=a_bits < 16,
                             w_bits=w_bits, a_bits=a_bits).to(dev)
    layer.mlp.down_proj = fqd
    man["fc"] = dict(mode=mode, rotated="W_h@R + runtime input R")
    man["down_proj"] = dict(mode=mode,
                            r4="exact fold + online" if r4_downproj
                            else "none")
    man["_contract"] = ("interface-boundary rotation: quantizers see "
                        "rotated act/weight; function preserved exactly "
                        "in fp64; no restore, no alpha, no R2")
    return man


def _get(ea, site):
    if site == "fc":
        return ea, "fc", ea.fc
    layer = ea.layers[0]
    if site.endswith("_proj") and hasattr(layer.self_attn, site):
        return layer.self_attn, site, getattr(layer.self_attn, site)
    return layer.mlp, site, getattr(layer.mlp, site)


def apply_embed_alpha(ea_model, alpha):
    """SEAGLE P3/GS-style branch balancing for the RAW native draft:
    E' = alpha*E and W_e' = W_e/alpha — EXACT reparameterization (the
    draft embedding output feeds only the fc e-half), changing only
    what the fc quantizers see. Apply BEFORE quantization."""
    ea = ea_model.ea_layer if hasattr(ea_model, "ea_layer") else ea_model
    assert isinstance(ea.fc, torch.nn.Linear), \
        "apply_embed_alpha must run before fc quantization"
    ea.embed_tokens.weight.data = ea.embed_tokens.weight.data * alpha
    ea.fc.weight.data[:, :D] = ea.fc.weight.data[:, :D] / alpha
    return dict(alpha=alpha,
                note="exact reparam E*=a, W_e/=a (pre-quant)")


def apply_native_raw_quant(ea_model, mode, site_mask=None,
                           quant_embed="fp16", site_modes=None):
    """In-place raw quantization of the strict draft.

    mode: QUANT_BITS key (fake_w4a4 / fake_w8a8 / fake_w4a16 / ...)
    site_mask: iterable of SITES to quantize (None = all 8)
    quant_embed: QUANT_BITS key for the embedding TABLE weight
                 ('fp16' = untouched)
    site_modes: optional {site: mode} overrides (component audit)
    Returns a manifest dict (site -> {w_bits, a_bits, sha16}).
    """
    ea = ea_model.ea_layer if hasattr(ea_model, "ea_layer") else ea_model
    mask = set(site_mask) if site_mask is not None else set(SITES)
    unknown = mask - set(SITES)
    assert not unknown, f"unknown sites {unknown}"
    man = {}
    for site in SITES:
        if site not in mask:
            continue
        m = (site_modes or {}).get(site, mode)
        w_bits, a_bits = QUANT_BITS[m]
        parent, name, lin = _get(ea, site)
        assert isinstance(lin, torch.nn.Linear), \
            f"{site} already replaced ({type(lin).__name__})"
        fq = FakeW4A4Linear(
            lin.weight.data.half(), lin.bias, f"native_raw.{site}",
            online_had=False,
            quant_weight=w_bits < 16, quant_act=a_bits < 16,
            w_bits=w_bits, a_bits=a_bits).to(lin.weight.device)
        setattr(parent, name, fq)
        import hashlib
        man[site] = dict(mode=m, w_bits=w_bits, a_bits=a_bits,
                         sha16=hashlib.sha256(
                             fq.w_fake.cpu().numpy().tobytes())
                         .hexdigest()[:16])
    if quant_embed != "fp16":
        w_bits, _ = QUANT_BITS[quant_embed]
        emb = ea.embed_tokens
        emb.weight.data = _weight_fake_quant(
            emb.weight.data.half(), w_bits,
            name="native_raw.embed").to(emb.weight.dtype)
        man["embed_tokens"] = dict(mode=quant_embed, w_bits=w_bits)
    # Gate-L self-check: forbidden knobs cannot be active here by
    # construction — record the invariant for the shard manifest.
    man["_contract"] = ("raw no-fold: no R1/R2/R4, no gamma, no alpha, "
                        "no rotation; stock forward")
    return man
