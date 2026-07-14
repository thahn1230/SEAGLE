"""The EAGLE-1 <-> rotated-target interface adapter.

The rotated+quantized target emits h_hat (R1 basis, gamma_f folded into lm_head):

    h_hat = (h / gamma_f) @ R1               (h = original post-final-norm feature)

so the exact inverse the EAGLE draft needs is

    h = (h_hat @ R1^T) * gamma_f             <-- unrotate_hidden()

Variants (docs/01 S3):
  naive   : feed h_hat straight to the stock draft (basis mismatch; the failure).
  unrotate: (Variant A) apply unrotate_hidden() at the draft entry, stock draft.
  conjugate: (Variant B) fold the map into the draft fc h-block (see draft_conjugation).

The draft always scores its OWN predicted features (original basis) through the
ORIGINAL lm_head, never the rotated one — so we substitute an original-basis head.

Discovered subtlety (see docs/04): folding into fc (Variant B) is exact only for the
FIRST draft token; EAGLE recycles the draft's original-basis predictions back through
the same fc during tree expansion, so a faithful Variant B must also compensate the
recycled features. Variant A has no such issue and is the correctness reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# core linear-algebra
# ---------------------------------------------------------------------------

def unrotate_hidden(h_rot: torch.Tensor, R1: torch.Tensor, gamma_f: torch.Tensor) -> torch.Tensor:
    """h = (h_hat @ R1^T) * gamma_f. R1:[D,D] orthogonal, gamma_f:[D]."""
    R1 = R1.to(h_rot.device, h_rot.dtype)
    gamma_f = gamma_f.to(h_rot.device, h_rot.dtype)
    return torch.matmul(h_rot, R1.t()) * gamma_f


def rotate_hidden(h: torch.Tensor, R1: torch.Tensor, gamma_f: torch.Tensor) -> torch.Tensor:
    """Inverse of unrotate_hidden: h_hat = (h / gamma_f) @ R1."""
    R1 = R1.to(h.device, h.dtype)
    gamma_f = gamma_f.to(h.device, h.dtype)
    return torch.matmul(h / gamma_f, R1)


# ---------------------------------------------------------------------------
# mathematical sanity checks
# ---------------------------------------------------------------------------

def check_orthogonal(R: torch.Tensor) -> dict:
    R = R.double()
    n = R.shape[0]
    I = torch.eye(n, dtype=R.dtype, device=R.device)
    err = (R.t() @ R - I).abs().max().item()
    return {"n": n, "max_abs_RtR_minus_I": err, "is_orthogonal": err < 1e-6}


def check_attention_preservation(D: int = 128, seq: int = 16, seed: int = 0) -> dict:
    """(Q@R3)(K@R3)^T == Q@K^T for orthogonal R3 (RoPE-shared rotation)."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(seq, D, generator=g, dtype=torch.float64)
    K = torch.randn(seq, D, generator=g, dtype=torch.float64)
    R3, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    base = Q @ K.t()
    rot = (Q @ R3) @ (K @ R3).t()
    err = (base - rot).abs().max().item()
    return {"D": D, "max_abs_score_diff": err, "preserved": err < 1e-8}


def check_unrotation_roundtrip(D: int = 4096, seq: int = 8, seed: int = 0) -> dict:
    """unrotate(rotate(h)) == h for random orthogonal R1 and positive gamma_f."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(seq, D, generator=g, dtype=torch.float64)
    R1, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    gamma_f = torch.rand(D, generator=g, dtype=torch.float64) + 0.5
    h_hat = rotate_hidden(h, R1, gamma_f)
    h_rec = unrotate_hidden(h_hat, R1, gamma_f)
    err = (h - h_rec).abs().max().item()
    return {"D": D, "max_abs_roundtrip_err": err, "ok": err < 1e-9}


# ---------------------------------------------------------------------------
# original-basis head for the draft
# ---------------------------------------------------------------------------

def build_original_head(lm_head_weight: torch.Tensor, device, dtype) -> nn.Linear:
    """nn.Linear(D->V, bias=False) holding the ORIGINAL (pre-rotation) lm_head.
    The draft predicts original-basis features, so it must be scored by this, not
    by the rotated lm_head that the target's verifier uses."""
    V, D = lm_head_weight.shape
    head = nn.Linear(D, V, bias=False)
    head.weight.data = lm_head_weight.to(dtype).clone()
    head = head.to(device=device, dtype=dtype)
    for p in head.parameters():
        p.requires_grad = False
    return head


# ---------------------------------------------------------------------------
# draft interface wrapper (single chokepoint: ea_layer.topK_genrate)
# ---------------------------------------------------------------------------

class DraftInterfaceAdapter:
    """Wraps EaModel.ea_layer.topK_genrate so that, regardless of caller
    (ea_model.forward or utils.update_inference_inputs), the draft receives the
    right-basis hidden and is scored by the original-basis head.

    variant:
      'naive'     : pass h_hat through unchanged (measures the break).
      'unrotate'  : h = (h_hat @ R1^T) * gamma_f  (Variant A).
      'conjugate' : pass h_hat through; the fc h-block is pre-folded elsewhere
                    (Variant B). Only the external hidden is fed here, so this is
                    exact for the first token; recycled features are handled by
                    the draft itself (see draft_conjugation for the caveat)."""

    def __init__(self, ea_model, R1, gamma_f, original_head_weight, variant="unrotate"):
        self.ea_model = ea_model
        self.variant = variant
        device = ea_model.base_model.lm_head.weight.device
        dtype = ea_model.base_model.dtype
        self.R1 = R1.to(device).to(torch.float32) if R1 is not None else None
        self.gamma_f = gamma_f.to(device).to(torch.float32) if gamma_f is not None else None
        self.head = build_original_head(original_head_weight, device, dtype)
        self._orig_topk = ea_model.ea_layer.topK_genrate
        self._installed = False

    def install(self):
        adapter = self

        def wrapped_topK_genrate(hidden_states, input_ids, head, logits_processor,
                                 *args, **kwargs):
            if adapter.variant == "unrotate":
                hs = hidden_states.to(torch.float32)
                hs = unrotate_hidden(hs, adapter.R1, adapter.gamma_f)
                hidden_states = hs.to(hidden_states.dtype)
            # naive/conjugate: pass hidden through unchanged.
            # Always score with the original-basis head, ignore the passed one.
            return adapter._orig_topk(hidden_states, input_ids, adapter.head,
                                      logits_processor, *args, **kwargs)

        self.ea_model.ea_layer.topK_genrate = wrapped_topK_genrate
        self._installed = True
        return self

    def uninstall(self):
        if self._installed:
            self.ea_model.ea_layer.topK_genrate = self._orig_topk
            self._installed = False
