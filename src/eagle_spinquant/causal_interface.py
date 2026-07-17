"""Phase C causal-panel adapters: restored vs exposed rotated interface.

The rotated target exposes a_t = n·R1 (pre-gamma normalized, rotated).
  - EXPOSED interface (C2/C4): the quantized first projection consumes a_t
    directly with the folded weight [W_e | W_h·D_γ·R1]
    (= ConcatSelectiveDraftAdapter, first_hidden_mode="gamma_R1").
  - RESTORED interface (C1/C3): a_t is first restored to the ORIGINAL
    EAGLE basis in fp32 — h_t = (a_t @ R1ᵀ) * γ — and the quantized first
    projection then consumes h_t with the ORIGINAL weight [W_e | W_h]
    (= first_hidden_mode="identity"). Pre-quantization the two paths are
    exactly equivalent (Gate B/H); under quantization they differ only in
    which basis the quantizers see.
"""
import torch

from .concat_selective_projection import ConcatSelectiveDraftAdapter


class RestoredInterfaceCSAdapter(ConcatSelectiveDraftAdapter):
    """Restore the rotated target feature to the original EAGLE basis
    (including final-RMSNorm gamma semantics) BEFORE the draft consumes it.
    Must be constructed with first_hidden_mode='identity'."""

    def __init__(self, *a, **k):
        k.setdefault("first_hidden_mode", "identity")
        assert k["first_hidden_mode"] == "identity"
        super().__init__(*a, **k)
        self.name = "concat_selective_restored_interface"
        self.n_restores = 0

    def transform_hidden(self, hs):
        x = hs.to(torch.float32) @ self.R1.to(hs.device, torch.float32).t()
        x = x * self.gamma.to(hs.device, torch.float32)
        self.n_restores += 1
        return x.to(hs.dtype)
