"""Draft-side SpinQuant (R1_D/R2_D) rotated-quantized forward for the
Llama-3.1 DFlash draft — shared by the rotation trainer and deployment
(trainer forward == deploy forward).

SpinQuant-faithful construction, adapted to the DFlash draft topology:

γ-fusion (norms → adjacent linears), done ONCE on frozen weights:
  input_ln γ_i  → q/k/v NOISE views          (per layer)
  hidden_norm γ → k/v CTX views              (shared across layers)
  post_ln γ_i   → gate/up                    (per layer)
  final norm γ_f → folded into the OUTPUT boundary matrix (shared head
                   cannot absorb it: B_out = D_γf · R1_D^T, one GEMM)
  NOTE: because k_proj/v_proj serve two branches with different fused γ,
  ctx-specific K/V weight views are MANDATORY-CORRECTNESS here (not a
  SEAGLE optimization): the same structural fact proven in the transfer
  study now arises from vanilla SpinQuant norm fusion alone.

R1_D conjugation (row-vector, y = xW^T):
  residual-reading (q/k/v noise, gate/up): W' = W_fused @ R1_D
  residual-writing (o_proj, down_proj):    W' = R1_D^T @ W
  fc: unaffected (input = target-basis hidden; folded with R1_T separately)
  boundaries (RD3): input  e @ R1_D   (shared embed output)
                    output norm_bare(h) @ (D_γf R1_D^T) before shared head
R2_D per layer [head_dim²]: v_proj output-side per-head (BOTH ctx and noise
  views) + o_proj input-side per-head. Valid: no norm sits between V and
  attention·V (q_norm/k_norm touch only Q/K).
R4-analog for draft down_proj input: online Hadamard (fixed), matching
  SpinQuant W4A4 practice; weight side folded into down_proj.

Quantization (STE): W4 per-channel sym RTN on the ROTATED weights per
forward (gradients flow to R1_D/R2_D); A4 per-token asym STE at every
linear input. identity-R init ⇒ FP-exact function (Gate D basis).
"""
import torch
from torch import nn

from . import SPINQUANT_ROOT  # noqa: F401
from .rc import rtn_sym_perchannel, act_fake_ste

D = 4096
HD = 128


def _had_matrix(n, device):
    from utils.hadamard_utils import random_hadamard_matrix
    return random_hadamard_matrix(n, device)


class RotQuantDraft(nn.Module):
    """DFlash draft with trainable R1_D/R2_D and STE W4A4, block-parallel
    forward compatible with vsq_specforge_port.specforge_block_forward
    (signature: position_ids, noise_embedding, target_hidden,
    attention_mask, ...)."""

    def __init__(self, draft, w_bits=4, a_bits=4, use_r2=True,
                 train_rotations=True, r4_draft=True, device="cuda",
                 train_weights=False):
        super().__init__()
        self.cfg = dict(w_bits=w_bits, a_bits=a_bits, use_r2=use_r2,
                        r4_draft=r4_draft, ctx_a_bits=None,
                        fc_p2=False)
        self.rc_matrix_buf = None
        # FIDI I7: per-channel SmoothQuant-like scale on the ctx K/V input
        # (post-R_C basis). FP-preserving pair: x*(s) vs W/(s) on input cols.
        self.ctx_smooth_buf = None
        self.block_size = draft.block_size
        self.mask_token_id = draft.mask_token_id
        self.target_layer_ids = draft.target_layer_ids
        self.n_layers = len(draft.layers)
        dev = device

        # ---- frozen γ-fused weight views (fp32 buffers)
        g_hid = draft.hidden_norm.weight.data.float().to(dev)
        self.register_buffer("fc_w", draft.fc.weight.data.float().to(dev))
        self.register_buffer("gamma_f",
                             draft.norm.weight.data.float().to(dev))
        self.eps = draft.norm.variance_epsilon \
            if hasattr(draft.norm, "variance_epsilon") else 1e-6
        for i, layer in enumerate(draft.layers):
            att, mlp = layer.self_attn, layer.mlp
            g_in = layer.input_layernorm.weight.data.float().to(dev)
            g_po = layer.post_attention_layernorm.weight.data.float().to(dev)
            B = lambda n, t: self.register_buffer(f"{n}_{i}", t)
            B("wq", att.q_proj.weight.data.float().to(dev) * g_in)
            B("wk_noise", att.k_proj.weight.data.float().to(dev) * g_in)
            B("wv_noise", att.v_proj.weight.data.float().to(dev) * g_in)
            B("wk_ctx", att.k_proj.weight.data.float().to(dev) * g_hid)
            B("wv_ctx", att.v_proj.weight.data.float().to(dev) * g_hid)
            B("wo", att.o_proj.weight.data.float().to(dev))
            B("wg", mlp.gate_proj.weight.data.float().to(dev) * g_po)
            B("wu", mlp.up_proj.weight.data.float().to(dev) * g_po)
            B("wd", mlp.down_proj.weight.data.float().to(dev))
            B("qn", att.q_norm.weight.data.float().to(dev))
            B("kn", att.k_norm.weight.data.float().to(dev))
        self.rotary = draft.rotary_emb
        inter = draft.layers[0].mlp.down_proj.weight.shape[1]
        if r4_draft:
            H4 = _had_matrix(inter, dev).float()
            self.register_buffer("had4", H4)
        # ---- trainable rotations
        lin1 = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            lin1.weight.copy_(torch.eye(D))
        self.r1 = nn.utils.parametrizations.orthogonal(
            lin1, orthogonal_map="cayley").to(dev)
        self.r2 = nn.ModuleList()
        for i in range(self.n_layers):
            l2 = nn.Linear(HD, HD, bias=False)
            with torch.no_grad():
                l2.weight.copy_(torch.eye(HD))
            self.r2.append(nn.utils.parametrizations.orthogonal(
                l2, orthogonal_map="cayley").to(dev))
        if not train_rotations:
            for p in self.parameters():
                p.requires_grad_(False)
        if train_weights:
            # promote the gamma-fused weight buffers to trainable parameters
            # (QAT family Q2/Q3/Q5: weights train, rotations stay frozen)
            names = [f"{n}_{i}" for i in range(self.n_layers)
                     for n in ("wq", "wk_noise", "wv_noise", "wk_ctx",
                               "wv_ctx", "wo", "wg", "wu", "wd")] + ["fc_w"]
            for nm in names:
                t = getattr(self, nm)
                delattr(self, nm)
                self.register_parameter(nm, nn.Parameter(t))

    def R1(self):
        return self.r1.weight.float()

    def R2(self, i):
        return self.r2[i].weight.float() if self.cfg["use_r2"] else None

    @torch.no_grad()
    def freeze_for_eval(self):
        """Precompute every rotated+quantized weight ONCE (bf16 cache) —
        eval-time rotations are frozen, so per-forward rotation/quant temps
        are pure waste (and were the OOM source on long sequences). Frees
        the fp32 source buffers afterwards."""
        R1 = self.R1()
        F = {}
        F["fc"] = self._wq(self.fc_w, "fc").to(torch.bfloat16)
        for i in range(self.n_layers):
            R2 = self.R2(i)
            F[f"wq_{i}"] = self._wq(getattr(self, f"wq_{i}") @ R1, "q").to(
                torch.bfloat16)
            F[f"wk_noise_{i}"] = self._wq(
                getattr(self, f"wk_noise_{i}") @ R1, "k").to(torch.bfloat16)
            F[f"wv_noise_{i}"] = self._wq(self._headwise(
                getattr(self, f"wv_noise_{i}"), R2, "out") @ R1, "v").to(
                torch.bfloat16)
            Rc = self.rc_matrix_buf
            wkc_src = getattr(self, f"wk_ctx_{i}")
            wvc_src = self._headwise(getattr(self, f"wv_ctx_{i}"), R2, "out")
            if Rc is not None:
                wkc_src = wkc_src @ Rc
                wvc_src = wvc_src @ Rc
            if self.ctx_smooth_buf is not None:
                wkc_src = wkc_src / self.ctx_smooth_buf[None, :]
                wvc_src = wvc_src / self.ctx_smooth_buf[None, :]
            F[f"wk_ctx_{i}"] = self._wq(wkc_src, "k").to(torch.bfloat16)
            F[f"wv_ctx_{i}"] = self._wq(wvc_src, "v").to(torch.bfloat16)
            F[f"wo_{i}"] = self._wq(R1.t() @ self._headwise(
                getattr(self, f"wo_{i}"), R2, "in"), "o").to(torch.bfloat16)
            F[f"wg_{i}"] = self._wq(
                getattr(self, f"wg_{i}") @ R1, "gate").to(torch.bfloat16)
            F[f"wu_{i}"] = self._wq(
                getattr(self, f"wu_{i}") @ R1, "up").to(torch.bfloat16)
            wd = getattr(self, f"wd_{i}")
            if self.cfg["r4_draft"]:
                wd = wd @ self.had4
            F[f"wd_{i}"] = self._wq(R1.t() @ wd, "down").to(torch.bfloat16)
        self._F = F
        self.register_buffer("R1_frozen", R1.clone())
        self.register_buffer("gamma_f_b", self.gamma_f.clone())
        for i in range(self.n_layers):
            for n in ("wq", "wk_noise", "wv_noise", "wk_ctx", "wv_ctx",
                      "wo", "wg", "wu", "wd"):
                delattr(self, f"{n}_{i}")
        del self.fc_w
        torch.cuda.empty_cache()
        return self

    # ---- quant helpers (STE)
    # comp: FIDI §16 component-selective W4A4 — components named in
    # cfg["fp_components"] ({"fc","q","k","v","o","gate","up","down"})
    # bypass BOTH their weight and input quantizers; empty/absent = no-op.
    def _fp(self, comp):
        if comp is None:
            return False
        S = self.cfg.get("fp_components", ())
        if isinstance(comp, tuple):        # ("k", layer) — "k" or "k@2"
            c, li = comp
            return c in S or f"{c}@{li}" in S
        return comp in S

    def _wq(self, w, comp=None):
        if self._fp(comp):
            return w
        return rtn_sym_perchannel(w, self.cfg["w_bits"])

    def _aq(self, x, comp=None):
        if self._fp(comp):
            return x
        return act_fake_ste(x, self.cfg["a_bits"])

    @staticmethod
    def _rms(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def _headwise(self, w, R2, side):
        """Apply per-head R2 to a projection weight (fp32).
        side='out': rows are heads*HD (v_proj); side='in': cols (o_proj)."""
        if R2 is None:
            return w
        if side == "out":
            h = w.shape[0] // HD
            wv = w.view(h, HD, -1)
            return torch.einsum("st,hto->hso", R2.t(), wv).reshape_as(w)
        h = w.shape[1] // HD
        wo = w.view(w.shape[0], h, HD)
        return torch.einsum("oht,ts->ohs", wo, R2).reshape(w.shape)

    def forward(self, position_ids, noise_embedding=None, target_hidden=None,
                attention_mask=None, use_cache=False, is_causal=False,
                **kw):
        frozen = hasattr(self, "_F")
        R1 = self.R1_frozen if frozen else self.R1()
        getw = (lambda key, build: self._F[key].float()) if frozen else             (lambda key, build: build())
        xdt = noise_embedding.dtype
        # H_t path: fc (target-basis input; quantized), bare hidden RMSNorm
        # (γ moved into ctx K/V views)
        cab = self.cfg.get("ctx_a_bits") or self.cfg["a_bits"]
        tf = target_hidden.float()
        if self._fp("fc"):
            thin = tf
        elif self.cfg.get("fc_p2"):
            B0, S0, MD = tf.shape
            thin = act_fake_ste(tf.reshape(B0, S0, 5, MD // 5), cab
                                ).reshape(B0, S0, MD)
        else:
            thin = act_fake_ste(tf, cab)
        Wfc = getw("fc", lambda: self._wq(self.fc_w, "fc"))
        cap = getattr(self, "_cap", None)   # FIDI capture tap (None = off)
        capd = getattr(self, "_cap_deep", None)  # FIDI §17 AW-input taps
        Zt = thin @ Wfc.t()
        if cap:
            cap("S3_Zt_dep", Zt)
        Ht = self._rms(Zt, self.eps)
        if cap:
            cap("S3_Ht_dep", Ht)
        if self.rc_matrix_buf is not None:
            Ht = Ht @ self.rc_matrix_buf
            if cap:
                cap("S3_Ht_rc_dep", Ht)
        if self.ctx_smooth_buf is not None:
            Ht = Ht * self.ctx_smooth_buf   # inverse folded into ctx K/V views
        # input boundary: shared-embed output -> draft basis
        h = (noise_embedding.float() @ R1)
        S = target_hidden.shape[1]
        cos, sin = self.rotary(noise_embedding, position_ids)
        cos, sin = cos.float(), sin.float()
        B_, Q = h.shape[:2]
        for i in range(self.n_layers):
            R2 = self.R2(i)
            xn_raw = self._rms(h, 1e-6)
            if capd:
                capd(f"xn_l{i}", xn_raw)
            xn_q = self._aq(xn_raw, ("q", i))
            xn_k = self._aq(xn_raw, ("k", i))
            xn_v = self._aq(xn_raw, ("v", i))
            wq = getw(f"wq_{i}", lambda: self._wq(
                getattr(self, f"wq_{i}") @ R1, ("q", i)))
            wkn = getw(f"wk_noise_{i}", lambda: self._wq(
                getattr(self, f"wk_noise_{i}") @ R1, ("k", i)))
            wvn = getw(f"wv_noise_{i}", lambda: self._wq(self._headwise(
                getattr(self, f"wv_noise_{i}"), R2, "out") @ R1, ("v", i)))
            xc_k = Ht if self._fp(("k", i)) else act_fake_ste(Ht, cab)
            xc_v = Ht if self._fp(("v", i)) else act_fake_ste(Ht, cab)
            Rc = self.rc_matrix_buf
            Sm = self.ctx_smooth_buf

            def _ctx_build(src, comp):
                w = src if Rc is None else src @ Rc
                if Sm is not None:
                    w = w / Sm[None, :]
                return self._wq(w, comp)
            wkc = getw(f"wk_ctx_{i}", lambda: _ctx_build(
                getattr(self, f"wk_ctx_{i}"), ("k", i)))
            wvc = getw(f"wv_ctx_{i}", lambda: _ctx_build(self._headwise(
                getattr(self, f"wv_ctx_{i}"), R2, "out"), ("v", i)))
            q = (xn_q @ wq.t()).view(B_, Q, -1, HD)
            q = self._rms(q) * getattr(self, f"qn_{i}")
            kc_, kn_ = xc_k @ wkc.t(), xn_k @ wkn.t()
            vc_, vn_ = xc_v @ wvc.t(), xn_v @ wvn.t()
            if cap:
                cap(f"S4_kctx_lin_l{i}", kc_)
                cap(f"S4_vctx_lin_l{i}", vc_)
                cap(f"S4_kdraft_lin_l{i}", kn_)
                cap(f"S4_vdraft_lin_l{i}", vn_)
            k = torch.cat([kc_, kn_], dim=1)
            v = torch.cat([vc_, vn_], dim=1)
            KVL = k.shape[1]
            k = self._rms(k.view(B_, KVL, -1, HD)) * getattr(self, f"kn_{i}")
            v = v.view(B_, KVL, -1, HD)
            # RoPE: cos/sin computed for full kv positions;
            # q uses the last Q entries
            def rot(t):
                t1, t2 = t[..., :HD // 2], t[..., HD // 2:]
                return torch.cat((-t2, t1), dim=-1)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            cq, sq = cos[:, -Q:].unsqueeze(1), sin[:, -Q:].unsqueeze(1)
            q = q * cq + rot(q) * sq
            k = k * cos.unsqueeze(1) + rot(k) * sin.unsqueeze(1)
            if cap:
                cap(f"S4_k_stored_l{i}", k)   # post k_norm + RoPE, [B,h,KVL,HD]
                cap(f"S4_v_stored_l{i}", v)   # raw projection, [B,h,KVL,HD]
            kr = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            vr = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, kr, vr, attn_mask=attention_mask, is_causal=False)
            a = attn.transpose(1, 2).reshape(B_, Q, -1)
            if capd:
                capd(f"attn_out_l{i}", a)
            wo = getw(f"wo_{i}", lambda: self._wq(
                R1.t() @ self._headwise(getattr(self, f"wo_{i}"),
                                        R2, "in"), ("o", i)))
            h = h + self._aq(a, ("o", i)) @ wo.t()
            xp_raw = self._rms(h, 1e-6)
            if capd:
                capd(f"xp_l{i}", xp_raw)
            xp_g = self._aq(xp_raw, ("gate", i))
            xp_u = self._aq(xp_raw, ("up", i))
            wg = getw(f"wg_{i}", lambda: self._wq(
                getattr(self, f"wg_{i}") @ R1, ("gate", i)))
            wu = getw(f"wu_{i}", lambda: self._wq(
                getattr(self, f"wu_{i}") @ R1, ("up", i)))
            z = torch.nn.functional.silu(xp_g @ wg.t()) * (xp_u @ wu.t())
            if self.cfg["r4_draft"]:
                z = z @ self.had4
            if frozen:
                wd = getw(f"wd_{i}", None)
            else:
                wd = getattr(self, f"wd_{i}")
                if self.cfg["r4_draft"]:
                    wd = wd @ self.had4
                wd = self._wq(R1.t() @ wd, ("down", i))
            if capd:
                capd(f"z_l{i}", z)   # post-had4 when r4_draft: deployed input
            h = h + self._aq(z, ("down", i)) @ wd.t()
        # output boundary: bare final norm, then D_γf · R1^T, shared head
        out = self._rms(h, self.eps) @ R1.t() * self.gamma_f
        return out.to(xdt)
