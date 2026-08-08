"""Context-only rotation R_C for DFlash (§18) — shared train/deploy forward.

Placement (see ROTATION_BOUNDARY_ANALYSIS.md §4):
    H_t = hidden_norm(fc(H_cat))          # upstream, possibly quantized fc
    H_t' = H_t @ R_C                      # this module
    ctx K/V use folded views  W_k^ctx = quant(W_k @ R_C),  same for V
    noise branch / embed / head untouched (no RD3 obstruction).

FP invariance at R_C = I; with quantization the ctx grids change — that is
the entire effect. RCDraft wraps a (possibly QLinear-quantized) draft; the
SAME forward is used by the trainer and by eval (Gate H by construction).
"""
import copy
import types

import torch
from torch import nn

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils


def rtn_sym_perchannel(w, bits):
    """Trainer-matched weight quantizer (no MSE clip search), STE gradient."""
    if bits >= 16:
        return w
    maxq = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq
    q = torch.clamp(torch.round(w / scale), -maxq - 1, maxq)
    return w + (q * scale - w).detach()


def act_fake_ste(x, bits):
    if bits >= 16:
        return x
    maxq = 2 ** bits - 1
    xmin = x.amin(dim=-1, keepdim=True).clamp(max=0)
    xmax = x.amax(dim=-1, keepdim=True).clamp(min=0)
    scale = ((xmax - xmin).clamp(min=1e-8)) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    dq = (q - zero) * scale
    return x + (dq - x).detach()


class RCContextKV(nn.Module):
    """Per-layer ctx-specific K/V path with R_C-folded quantized views.
    MUST be given ORIGINAL fp weights (not already-fake-quantized ones):
    deployment quantizes quant(W_fp @ R_C), never a double quantization."""

    def __init__(self, wk_fp, wv_fp, w_bits, a_bits):
        super().__init__()
        self.register_buffer("wk", wk_fp.detach().clone())
        self.register_buffer("wv", wv_fp.detach().clone())
        self.w_bits, self.a_bits = w_bits, a_bits

    def forward(self, H_t, R_C):
        wk = rtn_sym_perchannel((self.wk.float() @ R_C).to(self.wk.dtype)
                                .float(), self.w_bits)
        wv = rtn_sym_perchannel((self.wv.float() @ R_C).to(self.wv.dtype)
                                .float(), self.w_bits)
        x = act_fake_ste((H_t.float() @ R_C), self.a_bits)
        return (x @ wk.t()).to(H_t.dtype), (x @ wv.t()).to(H_t.dtype)


class RCDraft(nn.Module):
    """DFlash draft with R_C context rotation. `base` may already carry
    QLinear-quantized fc/QKVO/MLP (draft_quant). R_C is a full orthogonal
    matrix (buffer at eval; parametrized Linear during training)."""

    def __init__(self, base, orig, w_bits=4, a_bits=4):
        """base: draft used for noise/fc/MLP paths (may be QLinear-quantized);
        orig: unquantized draft supplying fp K/V weights for the ctx views."""
        super().__init__()
        self.base = base
        self.block_size = base.block_size
        self.mask_token_id = base.mask_token_id
        self.target_layer_ids = base.target_layer_ids
        self.ctx_kv = nn.ModuleList([
            RCContextKV(orig.layers[i].self_attn.k_proj.weight,
                        orig.layers[i].self_attn.v_proj.weight,
                        w_bits, a_bits) for i in range(len(base.layers))])
        self.register_buffer("R_C", torch.eye(4096))
        self._rc_param = None      # set by trainer

    def rc_matrix(self):
        if self._rc_param is not None:
            return self._rc_param.weight.float()
        return self.R_C.float()

    def forward(self, position_ids, attention_mask=None, noise_embedding=None,
                target_hidden=None, past_key_values=None, use_cache=False,
                **kw):
        b = self.base
        R = self.rc_matrix().to(target_hidden.device)
        H_t = b.hidden_norm(b.fc(target_hidden))
        hidden_states = noise_embedding
        pos_emb = b.rotary_emb(hidden_states, position_ids)
        for li, layer in enumerate(b.layers):
            k_ctx, v_ctx = self.ctx_kv[li](H_t, R)
            hidden_states = _layer_forward_ctxkv(
                layer, hidden_states, k_ctx, v_ctx, position_ids,
                past_key_values, use_cache, pos_emb, kw)
        return b.norm(hidden_states)


def _layer_forward_ctxkv(layer, hidden_states, k_ctx, v_ctx, position_ids,
                         past_key_value, use_cache, position_embeddings, kw):
    """Qwen3DFlashDecoderLayer.forward with precomputed ctx K/V injected."""
    from dflash.model import apply_rotary_pos_emb, eager_attention_forward
    from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
    residual = hidden_states
    hs = layer.input_layernorm(hidden_states)
    att = layer.self_attn
    bsz, q_len = hs.shape[:-1]
    ctx_len = k_ctx.shape[1]
    q = att.q_proj(hs).view(bsz, q_len, -1, att.head_dim)
    q = att.q_norm(q).transpose(1, 2)
    k_noise = att.k_proj(hs)
    v_noise = att.v_proj(hs)
    k = torch.cat([k_ctx, k_noise], dim=1).view(
        bsz, ctx_len + q_len, -1, att.head_dim)
    v = torch.cat([v_ctx, v_noise], dim=1).view(
        bsz, ctx_len + q_len, -1, att.head_dim)
    k = att.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_value is not None:
        k, v = past_key_value.update(
            k, v, att.layer_idx, {"sin": sin, "cos": cos,
                                  "cache_position": None})
    attn_fn = eager_attention_forward
    if att.config._attn_implementation != "eager":
        attn_fn = ALL_ATTENTION_FUNCTIONS[att.config._attn_implementation]
    out, _ = attn_fn(att, q, k, v, None, dropout=0.0, scaling=att.scaling,
                     sliding_window=att.sliding_window, is_causal=False)
    out = out.reshape(bsz, q_len, -1)
    hs = residual + att.o_proj(out)
    residual = hs
    hs = residual + layer.mlp(layer.post_attention_layernorm(hs))
    return hs


def load_rc_draft(base, orig, rc_ckpt=None, w_bits=4, a_bits=4):
    m = RCDraft(base, orig, w_bits=w_bits, a_bits=a_bits)
    if rc_ckpt:
        R = torch.load(rc_ckpt, map_location="cpu", weights_only=False)
        R = R["R_C"] if isinstance(R, dict) else R
        m.R_C = R.float()
        err = (R.float() @ R.float().t() - torch.eye(R.shape[0])).abs().max()
        assert err < 1e-3, f"R_C not orthogonal: {err}"  # Gate G
    return m.eval()
