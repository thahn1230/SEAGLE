"""Exact runtime-matched quantized ROTATION forward (study spec section 7).

SCOPE (2026-07-22): quantization-aware rotation learning — every model
weight is bitwise frozen; the ONLY trainable parameter is R_D / residual A
(optionally a scalar log-alpha for the EXPLICIT ablation only; the main
experiments select alpha by calibration sweep). The train_draft_core
pathway below is OUT OF SCOPE and retained solely for provenance.

ONE differentiable implementation of the deployed D4P3 draft whose forward
reproduces the runtime `ConcatSelectiveDraftAdapter(variant='folded',
first_hidden_mode='gamma_R1', quant_*='fake_w4a4', ar_r2r4=True,
embed_scale_alpha=alpha, first_fold_R=R_T)`; every quantizer is wrapped in a
straight-through estimator

    out = x + (Q_official(x.detach()) - x).detach()

so the FORWARD equals the official SpinQuant quantizer output exactly while
the backward is identity.

Weight folds replicate the runtime construction INCLUDING its cast order
(fp64 fold -> fp16 cast -> fp64 R2 conj -> fp16 -> fp32 R4 FWHT -> fp16;
P3 alpha divided in fp16 AFTER the cast, exactly like the adapter), so with
``exact=True`` the quantized weights are BITWISE equal to the runtime's
(Gate D). ``exact=False`` runs the same code in fp32 for training speed;
the residual deviation is ~1 quantization LSB on a small fraction of
elements and is measured/reported by the parity harness.

    W_first_preR = cast16[W_e | W_h ·fp64 D_γ·R_first], e-slice /= α (fp16)
    W_rec_preR   = cast16[W_e | W_h ·fp64 R_D],         e-slice /= α (fp16)
    projection: STE-quant(W_preR) fp16 GEMM + bias, then y·R_D fp32 (PostR)
    q,k        : cast16(W ·fp64 R_D)
    v          : cast16(R2 ⊗_head cast16(V ·fp64 R_D))
    o          : cast16((cast16(R_Dᵀ ·fp64 O)) ⊗_head R2ᵀ)
    gate,up    : cast16((W · diag(γ_l)) ·fp64 R_D)
    down       : FWHT_input(cast16(R_Dᵀ ·fp64 W)) (fp32 kernel, autograd)
    head       : cast16(W_lm ·fp64 R_D)             (fp16, never quantized)
    embedding  : ORIGINAL table · α (P3), fp16, never quantized
    post_attention_layernorm: weight 1 (γ_l fused into gate/up), eps 1e-6
    RoPE       : HF rotate_half; attention fp16 QK, fp32 softmax (cnets)

Capacity controls: ``train_draft_core=True`` registers the RAW ORIGINAL
draft weights (fc e/h split, q/k/v/o/gate/up/down) as Parameters — the
folds above consume them live, so any trained state remains runtime-
foldable and exportable as an original-basis state dict.

Draft KV: incremental per-depth chain with cached K/V; optional KV4
(runtime fake_quant_kv) on appended slices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fake_w4a4_draft as fq
from . import spinquant_bridge as sb
from .kv4_cache import fake_quant_kv

NH, HD = 32, 128
RMS_EPS = 1e-6


# ------------------------- STE-exact quant wrappers -------------------------

def ste_weight_quant(w, bits=4):
    with torch.no_grad():
        qw = fq._weight_fake_quant(w.detach(), bits)
    if not w.requires_grad:
        # no-grad path returns the OFFICIAL quantizer output directly:
        # the STE reconstruction w + (qw - w) re-rounds in fp16 and can
        # differ from qw by 1 ULP on grid-boundary elements (Gate D)
        return qw
    return w + (qw - w).detach()


class _ActQ:
    def __init__(self, bits=4):
        self.q = fq._act_quantizer(bits)

    def __call__(self, x):
        with torch.no_grad():
            x2 = x.detach()
            self.q.find_params(x2)
            qx = self.q(x2)
            self.q.free()
        if not x.requires_grad:
            return qx
        return x + (qx - x).detach()


def ste_kv4(x, bits=4):
    with torch.no_grad():
        qx = fake_quant_kv(x.detach(), bits=bits)
    if not x.requires_grad:
        return qx
    return x + (qx - x).detach()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, dim=HD, base=10000.0, device="cpu"):
    inv = 1.0 / base ** (torch.arange(0, dim, 2, device=device,
                                      dtype=torch.float32) / dim)
    ang = positions.float()[:, None] * inv[None]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


class _HadUSTE(torch.autograd.Function):
    """Input-side exact-Hadamard fold, forward via the runtime kernel,
    backward via the transpose transform (H orthogonal)."""

    @staticmethod
    def forward(ctx, W, had_K, K):
        sb.add_spinquant_to_syspath()
        from utils import hadamard_utils
        ctx.had_K, ctx.K = had_K, K
        with torch.no_grad():
            return hadamard_utils.matmul_hadU_cuda(W.float(), had_K, K) \
                .to(W.dtype)

    @staticmethod
    def backward(ctx, g):
        sb.add_spinquant_to_syspath()
        from utils import hadamard_utils
        gt = hadamard_utils.matmul_hadU_cuda(
            g.float().contiguous(), ctx.had_K, ctx.K).to(g.dtype)
        return gt, None, None


# ------------------------------- the module ---------------------------------

class ExactQuantizedRotationForward(nn.Module):
    CORE = ("W_e", "W_h", "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown")

    def __init__(self, sd, R_T, gamma, W_lm, rot, alpha_init=32.0,
                 train_alpha=False, w_bits=4, a_bits=4, draft_kv_bits=16,
                 train_draft_core=False, r2_seed=0, device="cuda:0",
                 first_fold_R=None, alpha_rec_init=None, r2_rot=None):
        super().__init__()
        self.dev = device
        self.rot = rot
        # GS/R2 study: optional learnable draft R2 (ResidualR2Rotation,
        # R2_D = R2_B @ C(B)). None = frozen baseline R2_B (buffer below),
        # bitwise-identical to the pre-study behavior.
        self.r2_rot = r2_rot
        self.w_bits, self.a_bits = w_bits, a_bits
        self.kv_bits = draft_kv_bits
        D = sd["fc.weight"].shape[0]
        self.D = D

        def cbuf(name, t):
            self.register_buffer(name, t.to(device))

        cbuf("R_T", R_T.float())
        cbuf("gamma64", gamma.double())
        cbuf("W_lm16", W_lm.half())
        cbuf("R_first", (first_fold_R if first_fold_R is not None
                         else R_T).float())
        cbuf("E", sd["embed_tokens.weight"].float())
        cbuf("R2_64", fq.baseline_r2(r2_seed))
        # fc.bias and post_attention_layernorm.weight are part of the
        # ORIGINAL EAGLE trainable set (model.parameters()); register them
        # as Parameters in QAT mode (folds consume them live either way).
        if train_draft_core:
            self.register_parameter(
                "b_fc", nn.Parameter(sd["fc.bias"].float().to(device)))
            self.register_parameter("gl64", nn.Parameter(
                sd["layers.0.post_attention_layernorm.weight"]
                .double().to(device)))
        else:
            cbuf("b_fc", sd["fc.bias"].float())
            cbuf("gl64",
                 sd["layers.0.post_attention_layernorm.weight"].double())
        sb.add_spinquant_to_syspath()
        from utils import hadamard_utils
        had_K, K = hadamard_utils.get_hadK(
            sd["layers.0.mlp.down_proj.weight"].shape[1])
        self.had_K = had_K.to(device) if had_K is not None else None
        self.had_KK = K

        core = dict(W_e=sd["fc.weight"][:, :D], W_h=sd["fc.weight"][:, D:],
                    Wq=sd["layers.0.self_attn.q_proj.weight"],
                    Wk=sd["layers.0.self_attn.k_proj.weight"],
                    Wv=sd["layers.0.self_attn.v_proj.weight"],
                    Wo=sd["layers.0.self_attn.o_proj.weight"],
                    Wgate=sd["layers.0.mlp.gate_proj.weight"],
                    Wup=sd["layers.0.mlp.up_proj.weight"],
                    Wdown=sd["layers.0.mlp.down_proj.weight"])
        self.train_draft_core = train_draft_core
        for name, t in core.items():
            t = t.float().to(device)          # fp32 master (== fp16 values)
            if train_draft_core:
                self.register_parameter(name, nn.Parameter(t.clone()))
            else:
                self.register_buffer(name, t)
        self.log_alpha = nn.Parameter(
            torch.tensor(float(alpha_init)).log().to(device),
            requires_grad=train_alpha)
        # exact python-float alpha for the fixed-alpha (train_alpha=False)
        # path: the runtime adapter divides the fp16 fold by the FULL-
        # precision scalar (and scales E in fp32) — an fp16-rounded or
        # exp(log(.)) round-tripped alpha only matches at powers of two
        self.alpha_exact = float(alpha_init)
        # EP3-P pathwise migration (m = D^beta): alpha_rec_init != None
        # switches the fold to the CURRENT deploy-adapter semantics —
        # e-slice divided in the fold dtype BEFORE the fp16 cast
        # (concat_selective_projection.py:465-472 divides the fp64 fold),
        # W_rec e-slice divided by alpha_rec, and the recurrent input's
        # e-slice rescaled by (alpha_rec/alpha_first) at runtime
        # (rec_embed_rescale). Incompatible with train_alpha.
        self.alpha_rec_exact = (float(alpha_rec_init)
                                if alpha_rec_init is not None else None)
        if self.alpha_rec_exact is not None:
            assert not train_alpha, "EP3-P pathwise alpha is fixed-only"
        self.aq = _ActQ(a_bits) if a_bits < 16 else None
        self.counters = dict(w_quant=0, a_quant=0, kv_quant=0)

    # ---- differentiable transformed weights, runtime cast order ------------
    def transformed_weights(self, exact=True):
        dd = torch.float64 if exact else torch.float32
        R = self.rot.R().to(self.dev).to(dd)
        Rf = self.R_first.to(dd)
        gam = self.gamma64.to(dd)
        gl = self.gl64.to(dd)
        # learnable R2_D when r2_rot is set (product formed in dd so that
        # B=0 reproduces the fp64 baseline bitwise in exact mode);
        # otherwise the frozen baseline R2_B buffer
        R2 = (self.r2_rot.R(dd).to(self.dev) if self.r2_rot is not None
              else self.R2_64.to(dd))

        def c16(x):
            return x.half()

        M = gam.unsqueeze(1) * Rf
        D = self.D
        if self.alpha_rec_exact is not None:
            # EP3-P pathwise fold, deploy-adapter order: the e-slice is
            # divided in the fold dtype BEFORE the fp16 cast (the runtime
            # divides its fp64 fold, then _make_linear casts), with
            # independent alpha_first / alpha_rec per path
            Wf_ = torch.cat([self.W_e.to(dd), self.W_h.to(dd) @ M], dim=1)
            Wr_ = torch.cat([self.W_e.to(dd), self.W_h.to(dd) @ R], dim=1)
            Wf = c16(torch.cat([Wf_[:, :D] / self.alpha_exact,
                                Wf_[:, D:]], dim=1))
            Wr = c16(torch.cat([Wr_[:, :D] / self.alpha_rec_exact,
                                Wr_[:, D:]], dim=1))
            q = c16(self.Wq.to(dd) @ R)
            k = c16(self.Wk.to(dd) @ R)
            v1 = c16(self.Wv.to(dd) @ R).to(dd)
            v = c16(torch.einsum("ab,hbc->hac", R2,
                                 v1.reshape(NH, HD, D)).reshape(NH * HD,
                                                                D))
            o1 = c16(R.t() @ self.Wo.to(dd)).to(dd)
            o = c16(torch.einsum("ohc,cb->ohb", o1.reshape(D, NH, HD),
                                 R2.t()).reshape(D, NH * HD))
            gate = c16((self.Wgate.to(dd) * gl.unsqueeze(0)) @ R)
            up = c16((self.Wup.to(dd) * gl.unsqueeze(0)) @ R)
            d1 = c16(R.t() @ self.Wdown.to(dd))
            down = _HadUSTE.apply(d1, self.had_K, self.had_KK)
            head = c16(self.W_lm16.to(dd) @ R)
            return dict(W_first=Wf, W_rec=Wr, q=q, k=k, v=v, o=o,
                        gate=gate, up=up, down=down, head=head,
                        R=R.float())
        Wf = c16(torch.cat([self.W_e.to(dd), self.W_h.to(dd) @ M], dim=1))
        Wr = c16(torch.cat([self.W_e.to(dd), self.W_h.to(dd) @ R], dim=1))
        if not self.log_alpha.requires_grad:
            # the runtime adapter divides its fp16 fold by the python-
            # float alpha ON CPU; CPU evaluates that as an fp32 division
            # by fp32(alpha) with CPU rounding. exact=True (deployment
            # parity) runs the division literally on CPU for bitwise
            # equality; the fp32-scalar GPU replica (identical on all but
            # rare rounding-boundary elements) serves the training path.
            a32 = torch.tensor(self.alpha_exact, dtype=torch.float32,
                               device=Wf.device)

            def div_a(x):
                if exact:
                    return (x.detach().cpu().half()
                            / self.alpha_exact).to(x.device)
                return (x.float() / a32).half()
        else:
            a16 = self.log_alpha.exp().half()

            def div_a(x):
                return x / a16
        Wf = torch.cat([div_a(Wf[:, :D]), Wf[:, D:]], dim=1)
        Wr = torch.cat([div_a(Wr[:, :D]), Wr[:, D:]], dim=1)

        q = c16(self.Wq.to(dd) @ R)
        k = c16(self.Wk.to(dd) @ R)
        v1 = c16(self.Wv.to(dd) @ R).to(dd)              # cast dance
        v = c16(torch.einsum("ab,hbc->hac", R2,
                             v1.reshape(NH, HD, D)).reshape(NH * HD, D))
        o1 = c16(R.t() @ self.Wo.to(dd)).to(dd)
        o = c16(torch.einsum("ohc,cb->ohb", o1.reshape(D, NH, HD),
                             R2.t()).reshape(D, NH * HD))
        gate = c16((self.Wgate.to(dd) * gl.unsqueeze(0)) @ R)
        up = c16((self.Wup.to(dd) * gl.unsqueeze(0)) @ R)
        d1 = c16(R.t() @ self.Wdown.to(dd))
        down = _HadUSTE.apply(d1, self.had_K, self.had_KK)
        head = c16(self.W_lm16.to(dd) @ R)
        return dict(W_first=Wf, W_rec=Wr, q=q, k=k, v=v, o=o, gate=gate,
                    up=up, down=down, head=head, R=R.float())

    # qanchor study: optional FROZEN per-site anchor scales (dict
    # site->-scale set by the trainer); STE quant under a fixed scale
    anchor_scales = None
    # qanchor Phase C: optional FP16 LoRA side-branches per site
    # (dict site -> (A [r,in], B [out,r]) Parameters); the base W4 codes
    # stay untouched => D_Q = 0 by construction
    lora = None

    def _lora_add(self, site, x, y):
        if self.lora is not None and site in self.lora:
            A, B = self.lora[site]
            y = y + (x.float() @ A.t() @ B.t()).to(y.dtype)
        return y

    def _qw(self, w, site=None):
        if self.w_bits < 16:
            self.counters["w_quant"] += 1
            if (self.anchor_scales is not None and site is not None
                    and site in self.anchor_scales):
                s = self.anchor_scales[site].to(w.device)
                with torch.no_grad():
                    qw = (s * torch.clamp(torch.round(
                        w.detach().float() / s), -8, 7)).to(w.dtype)
                if not w.requires_grad:
                    return qw
                return w + (qw - w).detach()
            return ste_weight_quant(w, self.w_bits)
        return w

    def _qa(self, x):
        if self.aq is not None:
            self.counters["a_quant"] += 1
            return self.aq(x)
        return x

    # qanchor Phase D hard budget: when set, retain grads on the folded
    # tensors of the LAST forward (STE identity => these equal the
    # gradients wrt the deployed quantized weights)
    capture_tw_grads = False

    def quantized_weights(self, exact=True):
        tw = self.transformed_weights(exact)
        if self.capture_tw_grads:
            for k in self.QSITES:
                if tw[k].requires_grad:
                    tw[k].retain_grad()
            self._tw_cap = tw
        out = {k: self._qw(tw[k], site=k) for k in
               ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up",
                "down")}
        out["head"] = tw["head"]
        return out, tw

    # ---- forward ----------------------------------------------------------
    def _proj(self, z, Wq16, R, site=None):
        zq = self._qa(z.half())
        y = F.linear(zq, Wq16, self.b_fc.half())
        if site is not None:
            y = self._lora_add(site, z.half(), y)
        return (y.float() @ R).to(y.dtype)          # PostProjectionR1

    def _attn_mlp(self, x, qw, pos, cache, attn_bias=None):
        B, T, D = x.shape
        h1 = x
        qh = self._lora_add("q", h1, F.linear(self._qa(h1), qw["q"])) \
            .view(B, T, NH, HD).transpose(1, 2)
        kh = self._lora_add("k", h1, F.linear(self._qa(h1), qw["k"])) \
            .view(B, T, NH, HD).transpose(1, 2)
        vh = self._lora_add("v", h1, F.linear(self._qa(h1), qw["v"])) \
            .view(B, T, NH, HD).transpose(1, 2)
        cos, sin = rope_cos_sin(pos, dim=HD, device=x.device)
        cos = cos[None, None].to(x.dtype)
        sin = sin[None, None].to(x.dtype)
        qh = qh * cos + rotate_half(qh) * sin
        kh = kh * cos + rotate_half(kh) * sin
        # deployed KV4 semantics (DraftKV4Patch): the CURRENT step's
        # attention sees its own K/V unquantized; the cache stores the
        # QUANTIZED append, which later steps read.
        if cache.get("k") is not None:
            k_all = torch.cat([cache["k"], kh], dim=2)
            v_all = torch.cat([cache["v"], vh], dim=2)
        else:
            k_all, v_all = kh, vh
        if self.kv_bits < 16:
            self.counters["kv_quant"] += 2
            kq = ste_kv4(kh, self.kv_bits)
            vq = ste_kv4(vh, self.kv_bits)
        else:
            kq, vq = kh, vh
        if cache.get("k") is not None:
            cache["k"] = torch.cat([cache["k"], kq], dim=2)
            cache["v"] = torch.cat([cache["v"], vq], dim=2)
        else:
            cache["k"], cache["v"] = kq, vq
        Tc = k_all.shape[2]
        att = (qh @ k_all.transpose(-1, -2)) / HD ** 0.5
        causal = torch.full((T, Tc), torch.finfo(att.dtype).min,
                            device=x.device, dtype=att.dtype) \
            .triu(Tc - T + 1)
        att = att + causal
        if attn_bias is not None:                 # padding mask (training)
            att = att + attn_bias.to(att.dtype)
        att = att.float().softmax(-1).to(x.dtype)
        o = (att @ v_all).transpose(1, 2).reshape(B, T, D)
        x = x + self._lora_add("o", o, F.linear(self._qa(o), qw["o"]))
        if getattr(self, "debug_capture", None) is not None:
            # Gate R2-A instrumentation: post-o_proj residual (the
            # attention-block OUTPUT, gauge-invariant under a correct
            # V/O R2 basis change; the pre-o_proj activation is basis-
            # dependent by design)
            self.debug_capture.append(x.detach().float().cpu())
        h32 = x.float()
        var = h32.pow(2).mean(-1, keepdim=True)
        h2 = (h32 * torch.rsqrt(var + RMS_EPS)).to(x.dtype)
        gt = self._lora_add("gate", h2, F.linear(self._qa(h2),
                                                 qw["gate"]))
        up = self._lora_add("up", h2, F.linear(self._qa(h2), qw["up"]))
        mid = F.silu(gt) * up
        if self.had_K is not None:
            sb.add_spinquant_to_syspath()
            from utils import hadamard_utils
            ms = mid.shape
            mid = hadamard_utils.matmul_hadU_cuda(
                mid.reshape(-1, ms[-1]), self.had_K, self.had_KK) \
                .reshape(ms)
        x = x + self._lora_add("down", mid,
                               F.linear(self._qa(mid), qw["down"]))
        return x

    def forward_chain(self, tok_ids, a_seq, K, pos_offset=0,
                      recur_tokens=None, rollout=None, exact=False):
        qw, tw = self.quantized_weights(exact)
        R = tw["R"]
        self._last_head = qw["head"]        # for head_logits() on targets
        a = (self.alpha_exact if not self.log_alpha.requires_grad
             else self.log_alpha.exp())
        # EP3-P: recurrent input e-slice rescale (runtime
        # rec_embed_rescale = alpha_rec/alpha_first, fp16 activation mul)
        rr = (None if self.alpha_rec_exact is None
              else float(self.alpha_rec_exact / self.alpha_exact))
        B, T = a_seq.shape[0], a_seq.shape[1]
        E = self.E[tok_ids] * a
        z = torch.cat([E[:, 1:T + 1].half(), a_seq.half()], dim=-1)
        y = self._proj(z, qw["W_first"], R, site="W_first")
        cache = {}
        pos = torch.arange(pos_offset, pos_offset + T, device=y.device)
        h_all = self._attn_mlp(y, qw, pos, cache)
        h_last = h_all[:, -1:]
        outs = []
        for k in range(K):
            logits = F.linear(h_last.squeeze(1).half(),
                              qw["head"]).float()
            if rollout is not None:
                chosen = rollout(logits).detach()
            elif recur_tokens is not None and k < recur_tokens.shape[1]:
                chosen = recur_tokens[:, k]
            else:
                chosen = logits.argmax(-1)
            outs.append((logits, h_last, chosen))
            if k == K - 1:
                break
            e_k = (self.E[chosen] * a).half().unsqueeze(1)
            if rr is not None:
                e_k = e_k * rr
            z = torch.cat([e_k, h_last.half()], dim=-1)
            y = self._proj(z, qw["W_rec"], R, site="W_rec")
            pos_k = torch.arange(pos_offset + T + k,
                                 pos_offset + T + k + 1, device=y.device)
            h_all = self._attn_mlp(y, qw, pos_k, cache)
            h_last = h_all[:, -1:]
        return outs

    QSITES = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up",
              "down")

    def qw_leaves(self, exact=False):
        """Detached quantized-weight LEAVES for gradient accumulation.
        The STE makes the quantizer a gradient identity, so micro-batches
        can run fwd/bwd against these leaves (accumulating .grad on them)
        and `fold_backward` then routes the summed gradients through the
        fold graph to the masters ONCE per optimizer step — bitwise the
        same forward and mathematically the same gradient as backprop
        through the full STE graph, without keeping the fold graph alive
        across micro-batches (memory) or re-running the w_clip search per
        micro-batch (time)."""
        with torch.no_grad():
            tw = self.transformed_weights(exact)
            out = {}
            for k in self.QSITES:
                w = tw[k]
                if self.w_bits < 16:
                    self.counters["w_quant"] += 1
                    w = fq._weight_fake_quant(w, self.w_bits)
                out[k] = w.requires_grad_(True)
            out["head"] = tw["head"].requires_grad_(True)
            R = tw["R"]
        return out, {"R": R}

    def fold_backward(self, leaves, exact=False):
        """Backprop accumulated leaf grads through the (rebuilt) fold
        graph into the trainable masters (STE: quantizer is identity)."""
        tw = self.transformed_weights(exact)
        # head excluded: W_lm is a frozen buffer (original EAGLE head is
        # frozen), so its fold has no grad path — identical to the full
        # STE backward where the head gradient dead-ends in the buffer.
        keys = [k for k in self.QSITES
                if leaves[k].grad is not None and tw[k].requires_grad]
        if not keys:
            return
        torch.autograd.backward(
            [tw[k] for k in keys],
            [leaves[k].grad.to(tw[k].dtype) for k in keys])

    def forward_train(self, tok_ids, feat_seq, pad_mask=None, exact=False,
                     qw=None, proj="first"):
        """Original-EAGLE single-step training pass (all rows first-path,
        exactly like eagle/train/main.py's causal forward): tok_ids (B,T+1)
        raw token ids, feat_seq (B,T) interface-input features. Returns
        h_all (B,T,D) draft hidden in the DEPLOYED basis; logits via
        `head_logits`. pad_mask (B,T) True=real token.

        qw: optional precomputed (qw, tw) from quantized_weights() — the
        folded/quantized weights are constant within one optimizer step,
        so gradient accumulation can reuse them (identical math; the
        official w_clip search dominates step time otherwise)."""
        if qw is None:
            qw, tw = self.quantized_weights(exact)
        else:
            qw, tw = qw
        R = tw["R"]
        a = (self.alpha_exact if not self.log_alpha.requires_grad
             else self.log_alpha.exp())
        B, T = feat_seq.shape[0], feat_seq.shape[1]
        E = self.E[tok_ids] * a
        e_part = E[:, 1:T + 1].half()
        if proj != "first" and self.alpha_rec_exact is not None:
            e_part = e_part * float(self.alpha_rec_exact
                                    / self.alpha_exact)
        z = torch.cat([e_part, feat_seq.half()], dim=-1)
        # proj="rec": multi-step rollout rows consume the draft's OWN
        # hidden through the deployed recurrent fold (no interface fold)
        y = self._proj(z, qw["W_first" if proj == "first" else "W_rec"],
                       R)
        pos = torch.arange(0, T, device=y.device)
        bias = None
        if pad_mask is not None:
            bias = torch.zeros(B, 1, 1, T, device=y.device)
            bias.masked_fill_(~pad_mask[:, None, None, :],
                              torch.finfo(torch.float16).min / 2)
        if self.training and torch.is_grad_enabled():
            # activation checkpointing: numerically identical forward,
            # recomputed in backward (full-T attention matrices dominate
            # training memory on 24 GB cards)
            h_all = torch.utils.checkpoint.checkpoint(
                lambda yy: self._attn_mlp(yy, qw, pos, {},
                                          attn_bias=bias),
                y, use_reentrant=False)
        else:
            h_all = self._attn_mlp(y, qw, pos, {}, attn_bias=bias)
        self._last_head = qw["head"]
        return h_all

    def head_logits(self, h):
        return F.linear(h.half(), self._last_head).float()

    def export_original_state(self):
        """Original-basis draft core weights (runtime-foldable export)."""
        out = {n: getattr(self, n).detach().cpu().half()
               for n in self.CORE}
        out["b_fc"] = self.b_fc.detach().cpu().half()
        out["gl"] = self.gl64.detach().cpu().half()
        return out


# scope-correction alias (old name used by armed pipelines)
ExactQATRotatedDraft = ExactQuantizedRotationForward
