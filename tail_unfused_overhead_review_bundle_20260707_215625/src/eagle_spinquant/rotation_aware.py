"""Rotation-aware EAGLE variants (D/E/F) — see docs/rotation_aware_eagle_overview.md.

Bases (row-vector convention, x' = x @ M):
  S      = diag(1/gamma_f) @ R1          (T_h;   h_hat = h @ S)
  S_inv  = R1.T @ diag(gamma_f)          (T_h_inv)
Folds for a linear L(x) = x @ W.T + b:
  in-fold  (consume x@S):  W' = W @ diag(gamma_f) @ R1        (== Variant-B fold)
  out-fold (emit    y@S):  W'' = R1.T @ diag(1/gamma_f) @ W,  b'' = b @ S
All folds are verified on random fp64 tensors by verify_fold_algebra() —
call it once before building anything real.

Variants:
  D1 : folded fc always; recycled features converted to S-basis at the
       recycled INPUT edge (T_h); original head scores raw f.        (oracle)
  D2 : folded fc always; ALL draft outputs converted to f_hat at the
       OUTPUT edge; rotated head scores f_hat.                        (oracle)
  E* : D1 loop + embedding transformed (e@R1 / (e/gamma)@R1 /
       e@R1 with co-folded fc e-block).                       (branch probe)
  F  : whole-draft algebraic conversion, mode 'gamma' (stream basis S,
       single path, RMSNorm inexact) or 'r1' (stream basis R1, exact,
       but external/recycled fc paths split).                (feasibility)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .study import EventAccum, VariantAdapter, fold_matrix

D = 4096


# ---------------------------------------------------------------------------
# matrices and folds (all fp64)
# ---------------------------------------------------------------------------

def s_matrix(R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """S = diag(1/gamma) @ R1 (fp64)."""
    return R1.double() / gamma.double().unsqueeze(1)


def s_inv_matrix(R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """S^-1 = R1.T @ diag(gamma) (fp64)."""
    return R1.double().t() * gamma.double().unsqueeze(0)


def in_fold(W: torch.Tensor, R1: torch.Tensor, gamma=None) -> torch.Tensor:
    """W' = W @ diag(gamma) @ R1 (gamma=None -> pure R1). Consumes S/R1 basis."""
    Wd = W.double()
    if gamma is not None:
        Wd = Wd * gamma.double().unsqueeze(0)
    return Wd @ R1.double()


def out_fold(W: torch.Tensor, R1: torch.Tensor, gamma=None) -> torch.Tensor:
    """W'' = R1.T @ diag(1/gamma) @ W (gamma=None -> pure R1). Emits S/R1 basis."""
    Wd = W.double()
    if gamma is not None:
        Wd = Wd / gamma.double().unsqueeze(1)
    return R1.double().t() @ Wd


def vec_to_basis(b: torch.Tensor, R1: torch.Tensor, gamma=None) -> torch.Tensor:
    """b'' = b @ S (row vector into S/R1 basis)."""
    bd = b.double()
    if gamma is not None:
        bd = bd / gamma.double()
    return bd @ R1.double()


def t_h(x: torch.Tensor, R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return (x.double() / gamma.double()) @ R1.double()


def t_h_inv(x: torch.Tensor, R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return (x.double() @ R1.double().t()) * gamma.double()


def verify_fold_algebra(seed: int = 0, n: int = 64, tol: float = 1e-9) -> dict:
    """Assert every fold identity on random fp64 tensors. Raises on failure."""
    g = torch.Generator().manual_seed(seed)
    R, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    gam = torch.rand(n, generator=g, dtype=torch.float64) + 0.25
    W = torch.randn(n, n, generator=g, dtype=torch.float64)
    b = torch.randn(n, generator=g, dtype=torch.float64)
    x = torch.randn(3, n, generator=g, dtype=torch.float64)
    S = s_matrix(R, gam)
    checks = {}
    # S @ S_inv = I
    checks["S_inv"] = (S @ s_inv_matrix(R, gam) - torch.eye(n).double()).abs().max().item()
    # in-fold: (x@S) @ in_fold(W).T == x @ W.T
    checks["in_fold_gamma"] = ((x @ S) @ in_fold(W, R, gam).t() - x @ W.t()).abs().max().item()
    checks["in_fold_r1"] = ((x @ R) @ in_fold(W, R).t() - x @ W.t()).abs().max().item()
    # out-fold: x @ out_fold(W).T (+ b@S) == (x @ W.T + b) @ S
    checks["out_fold_gamma"] = (x @ out_fold(W, R, gam).t() + vec_to_basis(b, R, gam)
                                - (x @ W.t() + b) @ S).abs().max().item()
    checks["out_fold_r1"] = (x @ out_fold(W, R).t() - (x @ W.t()) @ R).abs().max().item()
    # t_h / t_h_inv roundtrip and matrix consistency
    checks["t_h_matches_S"] = (t_h(x, R, gam) - x @ S).abs().max().item()
    checks["roundtrip"] = (t_h_inv(t_h(x, R, gam), R, gam) - x).abs().max().item()
    bad = {k: v for k, v in checks.items() if v > tol}
    if bad:
        raise AssertionError(f"fold algebra check failed: {bad}")
    return checks


def build_rotated_head(lm_head_weight, R1, gamma, device, dtype) -> nn.Linear:
    """Head that scores S-basis features: W_rot = W @ diag(gamma) @ R1
    (input-fold). Identical to SpinQuant's fused+rotated lm_head — the basis
    ledger verifies that numerically against the real rotated model."""
    W = in_fold(lm_head_weight, R1.cpu(), gamma.cpu())
    V, Din = W.shape
    head = nn.Linear(Din, V, bias=False)
    head.weight.data = W.to(dtype)
    head = head.to(device=device, dtype=dtype)
    for p in head.parameters():
        p.requires_grad = False
    return head


# ---------------------------------------------------------------------------
# D1 / D2 / E adapters
# ---------------------------------------------------------------------------

class _EmbedWrap(nn.Module):
    """Wraps draft embed_tokens; emits e / e@R1 / (e/gamma)@R1 (fp32 math)."""

    def __init__(self, emb: nn.Embedding, R1, gamma, mode: str):
        super().__init__()
        self.emb = emb
        self.mode = mode
        self.register_buffer("R1", R1.to(torch.float32))
        self.register_buffer("gamma", gamma.to(torch.float32))

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        e = self.emb(ids)
        x = e.to(torch.float32)
        if self.mode in ("r1", "r1_cofold"):
            x = x @ self.R1
        elif self.mode == "r1_gamma":
            x = (x / self.gamma) @ self.R1
        else:
            raise ValueError(self.mode)
        return x.to(e.dtype)


class RotatedLoopAdapter(VariantAdapter):
    """Variants D1/D2 (+ E embedding modes on top of the D1 loop).

    fc h-block is folded (B-style) for the WHOLE run; the recycled features
    are made basis-consistent by an explicit conversion:
      score_basis='original' (D1): recycled INPUT edge converts f -> T_h(f);
        head = original, scores raw f.
      score_basis='rotated'  (D2): OUTPUT edge converts every draft output
        f -> f_hat; head = rotated, scores f_hat; recycled f_hat feeds the
        folded fc directly.
    embed_mode: 'original' | 'r1' | 'r1_gamma' | 'r1_cofold'."""

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 score_basis="original", embed_mode="original"):
        super().__init__(ea_model, stash, device, dtype)
        self.score_basis = score_basis
        self.embed_mode = embed_mode
        self.name = ("D1" if score_basis == "original" else "D2")
        if embed_mode != "original":
            self.name = f"E_{embed_mode}"
        if score_basis == "rotated":
            self.head = build_rotated_head(stash["lm_head_weight"], self.R1.cpu(),
                                           stash["gamma_f"], device, dtype)
        self.gamma32 = stash["gamma_f"].to(device).to(torch.float32)
        self.n_conversions = 0
        self._first = False

    def arm(self):
        self._first = True

    def install(self):
        fc = self.ea_layer.fc
        Dh = fc.weight.shape[0]
        self._w_backup = fc.weight.data.clone()
        M = fold_matrix(self.R1.to(fc.weight.device), self.gamma.to(fc.weight.device))
        W_h = fc.weight.data[:, Dh:2 * Dh].double()
        fc.weight.data[:, Dh:2 * Dh] = (W_h @ M).to(fc.weight.dtype)
        if self.embed_mode == "r1_cofold":
            # e-block consumes e@R1: W_e' = W_e @ R1
            W_e = fc.weight.data[:, :Dh].double()
            fc.weight.data[:, :Dh] = (W_e @ self.R1.double().to(fc.weight.device)) \
                .to(fc.weight.dtype)
        if self.embed_mode != "original":
            self._orig_embed = self.ea_layer.embed_tokens
            self.ea_layer.embed_tokens = _EmbedWrap(
                self._orig_embed, self.R1, self.gamma32, self.embed_mode
            ).to(fc.weight.device)

        adapter = self
        self._orig_forward = self.ea_layer.forward

        def patched_forward(hidden_states, *a, **k):
            if adapter.score_basis == "original":       # D1: convert recycled INPUT
                if adapter._first:
                    adapter._first = False
                else:
                    end = adapter.unrot.start()
                    hs = hidden_states.to(torch.float32)
                    hidden_states = ((hs / adapter.gamma32) @ adapter.R1) \
                        .to(hidden_states.dtype)
                    end.record()
                    adapter.n_conversions += 1
                return adapter._orig_forward(hidden_states, *a, **k)
            # D2: convert every OUTPUT to f_hat
            out = adapter._orig_forward(hidden_states, *a, **k)
            h = out[0] if isinstance(out, tuple) else out
            end = adapter.unrot.start()
            h32 = h.to(torch.float32)
            h2 = ((h32 / adapter.gamma32) @ adapter.R1).to(h.dtype)
            end.record()
            adapter.n_conversions += 1
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2

        self.ea_layer.forward = patched_forward
        return super().install()

    def uninstall(self):
        super().uninstall()
        if hasattr(self, "_orig_forward"):
            self.ea_layer.forward = self._orig_forward
            del self._orig_forward
        if hasattr(self, "_w_backup"):
            self.ea_layer.fc.weight.data.copy_(self._w_backup)
            del self._w_backup
        if hasattr(self, "_orig_embed"):
            self.ea_layer.embed_tokens = self._orig_embed
            del self._orig_embed

    # CSV metadata
    def meta(self):
        return {
            "original_lm_head_used": self.score_basis == "original",
            "rotated_lm_head_used": self.score_basis == "rotated",
            "embedding_basis": {"original": "original", "r1": "e@R1_UNfolded",
                                "r1_gamma": "(e/gamma)@R1_UNfolded",
                                "r1_cofold": "e@R1_cofolded"}[self.embed_mode],
            "recycled_feature_basis": ("original_converted_at_input_Th" if
                                       self.score_basis == "original"
                                       else "rotated_f_hat"),
        }


# ---------------------------------------------------------------------------
# F: whole-draft algebraic conversion
# ---------------------------------------------------------------------------

def convert_draft_state(sd: dict, R1: torch.Tensor, gamma: torch.Tensor,
                        mode: str) -> tuple[dict, dict]:
    """Convert a draft state dict to a rotated-basis draft.

    mode='gamma': stream basis S (single fc path; RMSNorm exactness NOT
                  guaranteed — that is the experiment).
    mode='r1'  : stream basis R1 (exact; returns a SEPARATE external fc weight
                 in extra['fc_ext'] because external h_hat is S-basis).
    Returns (new_state_dict, extra) with extra['head_weight_fold'] describing
    the head fold to use ('gamma' or 'r1')."""
    assert mode in ("gamma", "r1")
    R1 = R1.double()
    gamma = gamma.double()
    gm = gamma if mode == "gamma" else None
    out = {k: v.clone() for k, v in sd.items()}
    extra = {"mode": mode, "head_weight_fold": mode}

    # embedding rows -> stream basis
    out["embed_tokens.weight"] = _cast(in_basis_rows(sd["embed_tokens.weight"],
                                                     R1, gm), sd["embed_tokens.weight"])

    # fc: e-block/h-block input folds, then whole output fold
    W = sd["fc.weight"].double()
    W_e, W_h = W[:, :D], W[:, D:2 * D]
    We_f = in_fold(W_e, R1, gm)
    Wh_f = in_fold(W_h, R1, gamma)          # h-block ALWAYS consumes S (h_hat)
    if mode == "gamma":
        fc_full = torch.cat([We_f, Wh_f], dim=1)
        out["fc.weight"] = _cast(out_fold(fc_full, R1, gamma), sd["fc.weight"])
        out["fc.bias"] = _cast(vec_to_basis(sd["fc.bias"], R1, gamma), sd["fc.bias"])
    else:
        # recycled path: h-block consumes R1-basis (the stream basis)
        Wh_rec = in_fold(W_h, R1)           # pure R1
        fc_rec = torch.cat([We_f, Wh_rec], dim=1)
        fc_ext = torch.cat([We_f, Wh_f], dim=1)
        out["fc.weight"] = _cast(out_fold(fc_rec, R1), sd["fc.weight"])
        extra["fc_ext"] = _cast(out_fold(fc_ext, R1), sd["fc.weight"])
        out["fc.bias"] = _cast(vec_to_basis(sd["fc.bias"], R1), sd["fc.bias"])

    p = "layers.0."
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        out[p + name + ".weight"] = _cast(
            in_fold(sd[p + name + ".weight"], R1, gm), sd[p + name + ".weight"])
    out[p + "self_attn.o_proj.weight"] = _cast(
        out_fold(sd[p + "self_attn.o_proj.weight"], R1, gm),
        sd[p + "self_attn.o_proj.weight"])

    # fuse gamma_l of post_attention_layernorm into gate/up, then input-fold
    gamma_l = sd[p + "post_attention_layernorm.weight"].double()
    for name in ("mlp.gate_proj", "mlp.up_proj"):
        Wg = sd[p + name + ".weight"].double() * gamma_l.unsqueeze(0)
        out[p + name + ".weight"] = _cast(in_fold(Wg, R1, gm), sd[p + name + ".weight"])
    out[p + "post_attention_layernorm.weight"] = torch.ones_like(
        sd[p + "post_attention_layernorm.weight"])
    out[p + "mlp.down_proj.weight"] = _cast(
        out_fold(sd[p + "mlp.down_proj.weight"], R1, gm),
        sd[p + "mlp.down_proj.weight"])
    return out, extra


def in_basis_rows(E: torch.Tensor, R1: torch.Tensor, gamma=None) -> torch.Tensor:
    """rows e -> e @ S (gamma) or e @ R1 (gamma=None)."""
    Ed = E.double()
    if gamma is not None:
        Ed = Ed / gamma.double().unsqueeze(0)
    return Ed @ R1.double()


def _cast(t, like):
    return t.to(like.dtype)


class AlgebraicDraftAdapter(VariantAdapter):
    """Variant F: load the converted draft; no runtime conversions.
    mode='gamma' (F_R_gamma): single path, rotated head, expects inexactness
    at the RMSNorm. mode='r1' (F_R_only): exact, but two fc paths (external
    vs recycled) — swap via .data pointer like B2."""

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 mode="gamma"):
        super().__init__(ea_model, stash, device, dtype)
        self.mode = mode
        self.name = "F_R_gamma" if mode == "gamma" else "F_R_only"
        gamma = stash["gamma_f"]
        if mode == "gamma":
            self.head = build_rotated_head(stash["lm_head_weight"], self.R1.cpu(),
                                           gamma, device, dtype)
        else:
            W = in_fold(stash["lm_head_weight"], self.R1.cpu().double())
            head = nn.Linear(W.shape[1], W.shape[0], bias=False)
            head.weight.data = W.to(dtype)
            self.head = head.to(device=device, dtype=dtype)
        self._gamma_cpu = gamma
        self.n_conversions = 0  # zero by design

    def install(self):
        ea = self.ea_layer
        self._sd_backup = {k: v.detach().cpu().clone()
                           for k, v in ea.state_dict().items()}
        conv, extra = convert_draft_state(
            {k: v.detach().cpu() for k, v in ea.state_dict().items()},
            self.R1.cpu(), self._gamma_cpu, self.mode)
        ea.load_state_dict(conv, strict=True)
        ea.to(self.device)
        if self.mode == "r1":
            fc = ea.fc
            self.W_rec = fc.weight.data
            self.W_ext = extra["fc_ext"].to(fc.weight.device, fc.weight.dtype)
            self._pending_external = False
            adapter = self
            self._orig_forward = ea.forward

            def patched_forward(hidden_states, *a, **k):
                if adapter._pending_external:
                    fc.weight.data = adapter.W_ext
                    adapter._pending_external = False
                else:
                    fc.weight.data = adapter.W_rec
                return adapter._orig_forward(hidden_states, *a, **k)
            ea.forward = patched_forward
        return super().install()

    def arm(self):
        if self.mode == "r1":
            self._pending_external = True

    def uninstall(self):
        super().uninstall()
        if hasattr(self, "_orig_forward"):
            self.ea_layer.forward = self._orig_forward
            del self._orig_forward
        if hasattr(self, "W_rec"):
            self.ea_layer.fc.weight.data = self.W_rec
        self.ea_layer.load_state_dict(self._sd_backup, strict=True)
        self.ea_layer.to(self.device)
        del self._sd_backup

    def meta(self):
        return {
            "original_lm_head_used": False,
            "rotated_lm_head_used": self.mode == "gamma",
            "embedding_basis": ("e@S_cofolded" if self.mode == "gamma"
                                else "e@R1_cofolded"),
            "recycled_feature_basis": ("rotated_f_hat_native" if self.mode == "gamma"
                                       else "R1_native_two_path"),
        }


class GNativeAdapter(VariantAdapter):
    """Variant G evaluation: a TRAINED rotation-native draft (weights loaded
    by the runner via --draft-ckpt). No folds, no transforms, ONE fc path;
    h_hat in, f_hat recycled natively, rotated head scores."""
    name = "G_native"

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16):
        super().__init__(ea_model, stash, device, dtype)
        self.head = build_rotated_head(stash["lm_head_weight"], self.R1.cpu(),
                                       stash["gamma_f"], device, dtype)
        self.n_conversions = 0

    def meta(self):
        return {"original_lm_head_used": False, "rotated_lm_head_used": True,
                "embedding_basis": "as-trained (init: e@S cofolded)",
                "recycled_feature_basis": "rotated_f_hat_native_trained"}


# ---------------------------------------------------------------------------
# standalone draft helpers (for the basis ledger / F localization)
# ---------------------------------------------------------------------------

def build_standalone_draft(draft_path, device, dtype=torch.float32):
    import os
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    cfg = EConfig.from_pretrained(os.path.join(draft_path, "config.json"))
    d = Model(cfg, bias=True)
    sd = torch.load(os.path.join(draft_path, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    d.load_state_dict(sd, strict=True)
    d = d.to(device=device, dtype=dtype).eval()
    d.diff_device = False
    d.init_tree()
    d.reset_kv()
    return d


@torch.no_grad()
def level_capture(draft, hidden, input_ids_full, head, max_levels=None):
    """One topK_genrate; returns (input_features, output_features) per call.
    Call 1 = external hidden; calls 2..L = recycled."""
    ins, outs = [], []
    orig_fwd = draft.forward

    def rec_fwd(hs, *a, **k):
        ins.append(hs.detach().float().cpu())
        out = orig_fwd(hs, *a, **k)
        o = out[0] if isinstance(out, tuple) else out
        outs.append(o.detach().float().cpu())
        return out
    draft.forward = rec_fwd
    draft.reset_kv()
    draft.reset()
    draft.topK_genrate(hidden, input_ids_full, head, None)
    draft.forward = orig_fwd
    return ins, outs
