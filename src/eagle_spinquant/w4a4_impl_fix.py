"""Implementation-correctness mode: w4a4_unfused_target_fused_draft_head.

TARGET: SpinQuant-style (fake or real) W4A4 transformer blocks, but the FINAL
TAIL is UNFUSED at runtime:
    h_hat     = RMSNorm0(x_R)                    (scale-free rotated hidden)
    h_unfused = (h_hat @ R1.T) * gamma_f         (== original post-norm hidden h)
    logits    = h_unfused @ W_lm_original.T
The hidden sent to the EAGLE draft is h_unfused (NOT h_hat).

DRAFT: fused lm_head kept (W_lm @ diag(gamma_f) @ R1, expects a scale-free
rotated feature). Projection (fc) pre-transforms are EXPLICIT runtime ops, NOT
folded into fc weights:
    first (external) forward : e -> e @ R1.T ,  hidden -> Identity   => [R1.T, I]
    recurrent (recycled)     : e -> e @ R1.T ,  hidden -> hidden @ R1.T => [R1.T, R1.T]

Everything is instance-attribute patching; third_party is never edited. This is
a correctness harness, not an optimized or claimed-optimal design.
"""

from __future__ import annotations

import torch

from . import rotation_aware as ra
from .study import VariantAdapter
from .tail_unfused import rmsnorm0

D = 4096


# ---------------------------------------------------------------------------
# target: unfused final tail
# ---------------------------------------------------------------------------

class UnfusedTailAdapter:
    """Patch base_model.model.norm.forward to emit h_unfused and restore the
    ORIGINAL lm_head. Saves the most recent h_hat / h_unfused for diagnostics.
    Also exposes fused-tail and unfused-tail logit computations for Test A."""

    def __init__(self, ea_model, stash, rot_dtype=torch.float16,
                 expose_h_hat=False):
        self.expose_h_hat = expose_h_hat   # ablation 4: pass h_hat, not h_unfused
        self.ea_model = ea_model
        self.norm = ea_model.base_model.model.norm
        self.lm_head = ea_model.base_model.lm_head
        self.eps = float(self.norm.variance_epsilon)
        dev = self.lm_head.weight.device
        R1 = stash["R1"].to(dev)
        self.R1t = R1.t().contiguous().to(rot_dtype)              # R1^T
        self.gamma = stash["gamma_f"].to(dev).to(torch.float16)
        self.W_lm_orig = stash["lm_head_weight"].to(dev).to(torch.float16)
        # fused head that the FUSED tail would use on h_hat: W_lm @ diag(g) @ R1
        self.W_fused = ra.in_fold(stash["lm_head_weight"].to(dev).double(),
                                  R1.double(),
                                  stash["gamma_f"].to(dev).double()).to(
            dev).to(torch.float16)
        self.last_h_hat = None
        self.last_h_unfused = None
        self._installed = False

    def _norm_forward(self, x_R):
        h_hat = rmsnorm0(x_R, self.eps)                          # scale-free rotated
        h_unfused = (h_hat.to(self.R1t.dtype) @ self.R1t).to(x_R.dtype) * self.gamma
        self.last_h_hat = h_hat.detach()
        self.last_h_unfused = h_unfused.detach()
        # ablation 4 exposes h_hat to the draft (and, to keep target logits
        # correct with the original lm_head, this is a KNOWN-WRONG control).
        return h_hat if self.expose_h_hat else h_unfused

    def install(self):
        assert not self._installed
        self._orig_norm_forward = self.norm.forward
        self._orig_lm_weight = self.lm_head.weight.data
        self.norm.forward = self._norm_forward
        # unfused tail -> original head on h_unfused; ablation4 (expose h_hat) ->
        # fused head on h_hat, so TARGET logits stay correct either way.
        self.lm_head.weight.data = (self.W_fused if self.expose_h_hat
                                    else self.W_lm_orig)
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        self.norm.forward = self._orig_norm_forward
        self.lm_head.weight.data = self._orig_lm_weight
        self._installed = False

    @torch.no_grad()
    def logits_from_xR(self, x_R):
        """Return (fused_logits, unfused_logits) on the SAME rotated residual."""
        h_hat = rmsnorm0(x_R, self.eps)
        fused = h_hat.to(torch.float16) @ self.W_fused.t()
        h_unf = (h_hat.to(self.R1t.dtype) @ self.R1t).to(x_R.dtype) * self.gamma
        unfused = h_unf.to(torch.float16) @ self.W_lm_orig.t()
        return fused, unfused


# ---------------------------------------------------------------------------
# draft: explicit projection pre-transforms + fused head
# ---------------------------------------------------------------------------

class DraftProjTransformAdapter(VariantAdapter):
    """Patch ea_layer.fc.forward to apply [R1.T, I] on the FIRST (external)
    forward and [R1.T, R1.T] on recurrent forwards (branch chosen by a per-
    topK_genrate call counter). Substitute the FUSED draft head. Log traces.

    transform overrides (for ablations): first_hidden in {"I","R1T"},
    embed in {"R1T","I"}, recurrent_hidden in {"R1T","I"}.
    """
    name = "w4a4_unfused_target_fused_draft_head"

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 first_embed="R1T", first_hidden="I",
                 rec_embed="R1T", rec_hidden="R1T",
                 head_mode="fused", trace=False):
        super().__init__(ea_model, stash, device, dtype)
        dev = device
        R1 = stash["R1"].to(dev)
        self.R1t = R1.t().contiguous().to(torch.float32)
        self.first_embed = first_embed
        self.first_hidden = first_hidden
        self.rec_embed = rec_embed
        self.rec_hidden = rec_hidden
        self.head_mode = head_mode
        if head_mode == "fused":
            self.head = ra.build_rotated_head(stash["lm_head_weight"], R1,
                                              stash["gamma_f"], dev, dtype)
            self.head_desc = dict(lm_head_mode="fused_W_lm@diag(gamma_f)@R1",
                                  feature_basis="scale_free_rotated_a@R1",
                                  has_R1T_absorbed=True, has_gamma_f_absorbed=True)
        else:  # "unfused" ablation: original head
            from .rotation_interface import build_original_head
            self.head = build_original_head(stash["lm_head_weight"], dev, dtype)
            self.head_desc = dict(lm_head_mode="unfused_original_W_lm",
                                  feature_basis="original_f",
                                  has_R1T_absorbed=False, has_gamma_f_absorbed=False)
        self._fc_idx = 0
        self.trace = trace
        self.proj_trace = []
        self.head_trace = []
        self._prompt_id = -1
        self._depth = 0

    def _apply(self, x, kind):
        if kind == "R1T":
            return (x.to(torch.float32) @ self.R1t).to(x.dtype)
        return x                                                 # "I"

    def set_context(self, prompt_id):
        self._prompt_id = prompt_id

    def arm(self):
        self._fc_idx = 0
        self._depth = 0

    def install(self):
        fc = self.ea_layer.fc
        adapter = self
        self._orig_fc_forward = fc.forward

        def fc_forward(z, *a, **k):
            e = z[..., :D]
            h = z[..., D:]
            first = (adapter._fc_idx == 0)
            e_kind = adapter.first_embed if first else adapter.rec_embed
            h_kind = adapter.first_hidden if first else adapter.rec_hidden
            hn_before = float(h.detach().float().norm())
            e2 = adapter._apply(e, e_kind)
            h2 = adapter._apply(h, h_kind)
            hn_after = float(h2.detach().float().norm())
            if adapter.trace and adapter._prompt_id >= 0 and len(adapter.proj_trace) < 200:
                adapter.proj_trace.append(dict(
                    prompt_id=adapter._prompt_id, tree_depth=adapter._depth,
                    is_first_external_target_forward=bool(first),
                    is_recycled_draft_forward=bool(not first),
                    embedding_transform=e_kind,
                    hidden_transform=h_kind,
                    hidden_source=("external_target_h_unfused" if first
                                   else "recycled_draft_feature"),
                    hidden_norm_before=round(hn_before, 4),
                    hidden_norm_after=round(hn_after, 4),
                    notes="[R1.T, I]" if (e_kind == "R1T" and h_kind == "I")
                          else f"[{e_kind}, {h_kind}]"))
            adapter._fc_idx += 1
            adapter._depth += 1
            return adapter._orig_fc_forward(torch.cat([e2, h2], -1), *a, **k)

        fc.forward = fc_forward

        # wrap topK_genrate: arm + substitute head + record head trace
        assert self._orig_topk is None
        self._orig_topk = self.ea_layer.topK_genrate

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            adapter.arm()
            if adapter.trace and adapter._prompt_id >= 0 and len(adapter.head_trace) < 50:
                adapter.head_trace.append(dict(
                    prompt_id=adapter._prompt_id, **adapter.head_desc,
                    feature_norm=round(float(hidden_states.detach().float().norm()), 4)))
            return adapter._orig_topk(hidden_states, input_ids, adapter.head,
                                      logits_processor, *a, **k)
        self.ea_layer.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None
        if hasattr(self, "_orig_fc_forward"):
            self.ea_layer.fc.forward = self._orig_fc_forward
            del self._orig_fc_forward

    def meta(self):
        return dict(first_embed=self.first_embed, first_hidden=self.first_hidden,
                    rec_embed=self.rec_embed, rec_hidden=self.rec_hidden,
                    **self.head_desc)
