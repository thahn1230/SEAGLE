"""Pure-R1 rotation-aware EAGLE (Tasks 3-4).

The draft is conjugated by R1 so its ENTIRE stream lives in the R1 basis:
  embedding rows -> e_R = e @ R1
  fc             -> consumes [e_R, h_R], emits x1_R = x1 @ R1
  attn / mlp     -> R1-conjugated (gamma_l fused into gate/up; RMSNorm0 commutes
                    with orthogonal R1 exactly)
  output feature -> f_R = f @ R1
  lm_head        -> W_lm @ R1  (scores f_R)

Crucially this is SINGLE-PATH: the external hidden it consumes is h_R = h @ R1
(supplied by the T5 EAGLE-friendly tail), and the recycled feature is
f_R = f @ R1 — BOTH R1-basis, BOTH consumed by the same fc h-block W_h @ R1.
Contrast the frozen-draft world where the target emits h_hat = (h/gamma_f)@R1
(scale-free), which differs from f_R by diag(gamma_f) and forces the B2/F_R_only
two-path split.

The R1-conjugation reuses rotation_aware.convert_draft_state(mode='r1'); we keep
ONLY its single recycled path (out['fc.weight']) and discard the external
fc_ext, because with the T5 tail the external hidden is already R1-basis.
"""

from __future__ import annotations

import torch

from . import rotation_aware as ra
from .study import VariantAdapter


def build_pure_r1_draft_state(sd: dict, R1: torch.Tensor,
                              gamma: torch.Tensor) -> dict:
    """R1-conjugated draft state dict (single fc path). sd is the ORIGINAL draft
    state dict; R1/gamma are the target's R1 and final-norm gamma_f (gamma is
    used only to fuse the draft's OWN layer norms — the fc h-block is pure R1)."""
    conv, _extra = ra.convert_draft_state(sd, R1, gamma, mode="r1")
    # out['fc.weight'] already = out_fold(cat([W_e@R1, W_h@R1]), R1); this is the
    # single path that consumes R1-basis e_R and h_R. Discard extra['fc_ext'].
    return conv


class PureR1Adapter(VariantAdapter):
    """Load the R1-conjugated draft into ea_layer and score with W_lm @ R1.
    NO runtime hidden transform, NO fc two-path swap — the T5 tail feeds h_R
    directly. Pair with tail_unfused.TailAdapter(mode='eagle_friendly_hR')."""
    name = "pure_r1"

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16):
        super().__init__(ea_model, stash, device, dtype)
        R1_cpu = self.R1.detach().cpu().double()       # [D,D]
        W = ra.in_fold(stash["lm_head_weight"].cpu(), R1_cpu)   # W_lm @ R1
        import torch.nn as nn
        head = nn.Linear(W.shape[1], W.shape[0], bias=False)
        head.weight.data = W.to(dtype)
        self.head = head.to(device=device, dtype=dtype)
        for p in self.head.parameters():
            p.requires_grad = False
        self.n_conversions = 0

    def install(self):
        ea = self.ea_layer
        self._sd_backup = {k: v.detach().cpu().clone()
                           for k, v in ea.state_dict().items()}
        conv = build_pure_r1_draft_state(
            {k: v.detach().cpu() for k, v in ea.state_dict().items()},
            self.R1.cpu(), self.gamma.cpu())
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype) for k, v in conv.items()},
                           strict=True)
        ea.to(self.device)
        return super().install()

    def uninstall(self):
        super().uninstall()
        self.ea_layer.load_state_dict(self._sd_backup, strict=True)
        self.ea_layer.to(self.device)
        del self._sd_backup

    def meta(self):
        return {"original_lm_head_used": False, "rotated_lm_head_used": True,
                "embedding_basis": "e_R = e@R1", "recycled_feature_basis": "f_R = f@R1",
                "external_hidden_basis": "h_R = h@R1", "fc_paths": 1}
