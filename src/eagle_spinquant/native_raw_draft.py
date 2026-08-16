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


def _get(ea, site):
    if site == "fc":
        return ea, "fc", ea.fc
    layer = ea.layers[0]
    if site.endswith("_proj") and hasattr(layer.self_attn, site):
        return layer.self_attn, site, getattr(layer.self_attn, site)
    return layer.mlp, site, getattr(layer.mlp, site)


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
