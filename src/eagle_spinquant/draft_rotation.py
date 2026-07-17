"""Phase G: draft-specific residual rotation R_D (spec §6, §15, §20).

Basis contract (row-vector convention, verified against the repository's
concat-selective folds):

  target exposes  a_t = n · R_T           (pre-gamma normalized, rotated)
  draft internal  gauge R_D               (features live as y · R_D)

  first projection  (input [e | a_t]):
      W_first' = [ W_e | W_h · D_γ · R_T ] , output → · R_D
      (the T→D bridge R_Tᵀ·R_D never materializes: R_T is folded into the
       hidden-side columns, R_D into the post-projection rotation)
  recurrent projection (input [e | h_d · R_D]):
      W_rec'  = [ W_e | W_h · R_D ]        , output → · R_D
  AR decoder: R_D-conjugated (same construction as the validated R1
      conjugation), with draft R2/R4 for W4A4.
  LM head: W_lm · R_D.
  Embedding: ORIGINAL basis always (P3 alpha optional, folded).

R_D == R_T  reproduces the validated shared-rotation adapter exactly.

Training: R_D is the only trainable tensor (plus optional foldable alpha).
Fake quantizers use straight-through estimators. Orthogonality is kept by
projection after each optimizer step (QR retraction) and tracked as
||R_DᵀR_D − I||_F (Gate G).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- STE fake quantizers (training-time) ----------------

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def w4_fake_quant_ste(w, bits=4):
    """Per-output-channel symmetric RTN with fixed clip=1.0 (training proxy
    for the MSE-clip inference quantizer; the eval path uses the official
    clip search)."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return _RoundSTE.apply(w / scale).clamp(-qmax - 1, qmax) * scale


def a4_fake_quant_ste(x, bits=4):
    """Per-token asymmetric over the last dim."""
    qmax = 2 ** bits - 1
    mn = x.amin(-1, keepdim=True)
    mx = x.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-6) / qmax
    q = _RoundSTE.apply((x - mn) / scale).clamp(0, qmax)
    return q * scale + mn


# ---------------- trainable rotated draft ----------------

class RotatedDraftTrainer(nn.Module):
    """Differentiable quantized EAGLE draft under a trainable R_D.

    Frozen inputs: draft state dict tensors (fc, decoder, head, embed) and
    the target-side R_T / gamma. Only self.R_D (and optional log_alpha)
    require grad. Weight folds are recomputed per forward so gradients flow
    through R_D.

    Teacher-forced depth-K unroll:
      inputs: token embeddings e_{t..t+K-1}, target hidden a_t (rotated
      basis) for the first step; recurrent steps consume the draft's own
      (quantized) hidden output.
    """

    def __init__(self, sd, R_T, gamma, W_lm, w_bits=4, a_bits=4,
                 alpha_init=32.0, train_alpha=False, init="RT",
                 device="cuda:0"):
        super().__init__()
        self.dev = device
        D = sd["fc.weight"].shape[0]
        self.D = D
        # frozen originals (fp32 for stable folds)
        self.register_buffer("W_fc", sd["fc.weight"].float().to(device))
        self.register_buffer("b_fc", sd["fc.bias"].float().to(device))
        self.register_buffer("R_T", R_T.float().to(device))
        self.register_buffer("gamma", gamma.float().to(device))
        self.register_buffer("W_lm", W_lm.float().to(device))
        # single decoder layer weights (frozen; R_D-conjugated per forward)
        self.dec = {k: v.float().to(device) for k, v in sd.items()
                    if k.startswith("layers.0.")}
        self.register_buffer("E", sd["embed_tokens.weight"].float()
                             .to(device))
        # trainable rotation
        if init == "RT":
            R0 = R_T.float().clone()
        elif init == "identity":
            R0 = torch.eye(D)
        elif init == "hadamard":
            import scipy.linalg as sl
            R0 = torch.tensor(sl.hadamard(D) / D ** 0.5, dtype=torch.float32)
        else:
            g = torch.Generator().manual_seed(0)
            R0, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
        self.R_D = nn.Parameter(R0.to(device))
        self.log_alpha = nn.Parameter(
            torch.tensor(float(alpha_init)).log().to(device),
            requires_grad=train_alpha)
        self.w_bits, self.a_bits = w_bits, a_bits

    # -- exact folds (recomputed per forward; differentiable in R_D) -----
    def folded_first(self):
        alpha = self.log_alpha.exp()
        M = self.gamma.unsqueeze(1) * self.R_T           # D_γ · R_T
        W_e = self.W_fc[:, :self.D] / alpha
        W_h = self.W_fc[:, self.D:] @ M
        return torch.cat([W_e, W_h], dim=1)

    def folded_rec(self):
        alpha = self.log_alpha.exp()
        W_e = self.W_fc[:, :self.D] / alpha
        W_h = self.W_fc[:, self.D:] @ self.R_D
        return torch.cat([W_e, W_h], dim=1)

    def _proj(self, z, W):
        if self.w_bits < 16:
            W = w4_fake_quant_ste(W, self.w_bits)
        if self.a_bits < 16:
            z = a4_fake_quant_ste(z, self.a_bits)
        y = F.linear(z, W, self.b_fc)
        return y @ self.R_D                              # post-projection R_D

    def _decoder(self, x, attn_mask=None):
        """R_D-conjugated single LlamaDecoderLayer, causal, no KV cache
        (training unroll uses full-prefix attention each step). Quantized
        linears with STE."""
        Rd = self.R_D
        # explicit R_D conjugation via basis change (mathematically equal to
        # folding W' = R_Dᵀ·W·R_D into every decoder linear; quantizers act
        # on original-basis weights here — the EVAL path quantizes the
        # conjugated weights, an intentional training/eval proxy gap
        # recorded in the report)
        x_orig = x @ Rd.t()                               # to original basis
        h = self._llama_layer_original(x_orig, attn_mask)
        return h @ Rd                                     # back to R_D gauge

    def _llama_layer_original(self, x, attn_mask):
        d = self.dec
        pre = d["layers.0.input_layernorm.weight"]
        post = d["layers.0.post_attention_layernorm.weight"]

        def rms(v, w):
            return w * v * torch.rsqrt(
                v.pow(2).mean(-1, keepdim=True) + 1e-6)

        def q(name, v):
            W = d[f"layers.0.{name}.weight"]
            if self.w_bits < 16:
                W = w4_fake_quant_ste(W, self.w_bits)
            if self.a_bits < 16:
                v = a4_fake_quant_ste(v, self.a_bits)
            return v @ W.t()

        B, T, D = x.shape
        n_head, hd = 32, D // 32
        h1 = rms(x, pre)
        qh = q("self_attn.q_proj", h1).view(B, T, n_head, hd)
        kh = q("self_attn.k_proj", h1).view(B, T, n_head, hd)
        vh = q("self_attn.v_proj", h1).view(B, T, n_head, hd)
        qh, kh = _rope(qh, kh)
        att = torch.einsum("bqhd,bkhd->bhqk", qh, kh) / hd ** 0.5
        causal = torch.full((T, T), float("-inf"), device=x.device) \
            .triu(1)
        att = (att + causal).softmax(-1)
        o = torch.einsum("bhqk,bkhd->bqhd", att, vh).reshape(B, T, D)
        x = x + q("self_attn.o_proj", o)
        h2 = rms(x, post)
        g = q("mlp.gate_proj", h2)
        u = q("mlp.up_proj", h2)
        x = x + q("mlp.down_proj", F.silu(g) * u)
        return x

    def unroll(self, tok_ids, a_seq, K):
        """Teacher-forced depth-K unroll following the EAGLE-1 contract.

        tok_ids: (B, T+K) visited token ids — prefix window T plus the K
            teacher-trajectory tokens.
        a_seq: (B, T, D) rotated target hiddens a_i for the prefix window
            (draft prefill inputs are ALL first-path: [e_{i+1}, a_i]).
        Returns list over k=1..K of (logits, hidden_gauge).
        """
        alpha = self.log_alpha.exp()
        B, TK = tok_ids.shape
        T = a_seq.shape[1]
        E = self.E[tok_ids]                                # (B,T+K,D)
        Wf, Wr = self.folded_first(), self.folded_rec()
        # prefill: positions i=0..T-1 consume [e_{i+1}, a_i]  (parallel)
        e_pref = E[:, 1:T + 1] * alpha
        z_pref = torch.cat([e_pref, a_seq], dim=-1)
        y_pref = self._proj(z_pref, Wf)                    # (B,T,D) gauge
        outs = []
        seq = y_pref
        h_gauge = None
        for k in range(K):
            if k == 0:
                h_all = self._decoder(seq)
                h_gauge = h_all[:, -1:]
            else:
                e_k = E[:, T + k:T + k + 1] * alpha
                z = torch.cat([e_k, h_gauge], dim=-1)
                y = self._proj(z, Wr)
                seq = torch.cat([seq, y], dim=1)
                h_all = self._decoder(seq)
                h_gauge = h_all[:, -1:]
            logits = h_gauge.squeeze(1) @ (self.W_lm @ self.R_D).t()
            outs.append((logits, h_gauge))
        return outs

    def orthogonality_error(self):
        I = torch.eye(self.D, device=self.R_D.device)
        return float((self.R_D.t() @ self.R_D - I).norm())

    @torch.no_grad()
    def retract(self):
        """QR retraction back onto the orthogonal manifold."""
        Q, R = torch.linalg.qr(self.R_D.data)
        s = torch.sign(torch.diagonal(R))
        self.R_D.data = Q * s


def _rope(q, k, base=10000.0):
    B, T, H, Dh = q.shape
    pos = torch.arange(T, device=q.device, dtype=torch.float32)
    inv = 1.0 / base ** (torch.arange(0, Dh, 2, device=q.device,
                                      dtype=torch.float32) / Dh)
    ang = pos[:, None] * inv[None]
    cos = torch.cos(ang)[None, :, None, :]
    sin = torch.sin(ang)[None, :, None, :]

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        o = torch.empty_like(x)
        o[..., 0::2] = x1 * cos - x2 * sin
        o[..., 1::2] = x1 * sin + x2 * cos
        return o
    return rot(q), rot(k)
