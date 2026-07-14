"""Variant B: fold the unrotation into the EAGLE draft's first projection (fc).

Draft forward (cnets.py:592-593):  fc(cat([e, h]))  with fc: nn.Linear(2D -> D).
In nn.Linear terms y = x @ W^T; x = [e | h] so W[:, :D] is the embedding block and
W[:, D:2D] is the hidden block. We want the draft to consume h_hat directly:

    h = unrotate(h_hat) = (h_hat @ R1^T) * gamma_f = h_hat @ (R1^T @ diag(gamma_f))

Require h_hat @ W_h_new^T == h @ W_h_old^T for all h_hat, which gives

    W_h_new = W_h_old @ (diag(gamma_f) @ R1)          # nn.Linear [D_out, D_in]

The embedding block is untouched (the draft's embedding is the original, unrotated
one). See docs/01 S2 and the recycling caveat in docs/04.
"""

from __future__ import annotations

import copy

import torch


def conjugate_draft_fc(ea_layer, R1: torch.Tensor, gamma_f: torch.Tensor) -> None:
    """In-place fold of the unrotation into fc's hidden block. `ea_layer` is the
    EAGLE draft (cnets.Model). Idempotent guard via a flag."""
    if getattr(ea_layer, "_fc_conjugated", False):
        raise RuntimeError("fc already conjugated; refusing to double-apply")
    fc = ea_layer.fc
    D = fc.weight.shape[0]           # out_features == hidden_size
    assert fc.weight.shape[1] == 2 * D, f"unexpected fc shape {fc.weight.shape}"
    dev = fc.weight.device
    R1 = R1.to(dev, torch.float64)
    gamma_f = gamma_f.to(dev, torch.float64)
    M = gamma_f.unsqueeze(1) * R1     # diag(gamma_f) @ R1  -> [D, D]
    W_h_old = fc.weight.data[:, D:2 * D].to(torch.float64)
    W_h_new = W_h_old @ M
    fc.weight.data[:, D:2 * D] = W_h_new.to(fc.weight.dtype)
    ea_layer._fc_conjugated = True


@torch.no_grad()
def conjugation_identity_check(ea_layer, R1, gamma_f, D=None, seq=8, seed=0,
                              device="cuda", dtype=torch.float32) -> dict:
    """Full-precision identity: for a random hidden h and matching input_ids,
    compare the FIRST-forward draft feature/logit from
      (a) stock draft consuming h            (original basis)
      (b) conjugated draft consuming h_hat   (rotated basis)
    Reports max abs err, relative err, cosine similarity, and logit KL. This is a
    single-forward check (Variant B is exact here by construction)."""
    from .rotation_interface import rotate_hidden, build_original_head

    ea_layer = copy.deepcopy(ea_layer).to(device=device, dtype=dtype).eval()
    if D is None:
        D = ea_layer.fc.weight.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(1, seq, D, generator=g, dtype=torch.float32).to(device, dtype)
    input_ids = torch.randint(0, ea_layer.vocab_size, (1, seq + 1), generator=g).to(device)

    R1_t = R1.to(device, torch.float32)
    gamma_t = gamma_f.to(device, torch.float32)
    h_hat = rotate_hidden(h.float(), R1_t, gamma_t).to(dtype)

    # (a) stock draft on h
    ea_layer.reset()
    if hasattr(ea_layer, "stable_kv"):
        ea_layer.stable_kv = None
    out_a, _ = ea_layer(h, input_ids=input_ids[:, 1:], use_cache=True)
    feat_a = out_a[:, -1].float()

    # (b) conjugated draft on h_hat
    conj = copy.deepcopy(ea_layer)
    conjugate_ok = True
    conjugate_draft_fc(conj, R1_t, gamma_t)
    conj.reset()
    if hasattr(conj, "stable_kv"):
        conj.stable_kv = None
    out_b, _ = conj(h_hat, input_ids=input_ids[:, 1:], use_cache=True)
    feat_b = out_b[:, -1].float()

    abs_err = (feat_a - feat_b).abs().max().item()
    rel_err = (abs_err / (feat_a.abs().max().item() + 1e-12))
    cos = torch.nn.functional.cosine_similarity(
        feat_a.flatten(), feat_b.flatten(), dim=0).item()

    # logit KL through a random-but-shared original head (uses draft's embed dim)
    V = ea_layer.vocab_size
    head_w = torch.randn(V, D, generator=g, dtype=torch.float32).to(device) * (D ** -0.5)
    head = build_original_head(head_w, device, torch.float32)
    la = torch.log_softmax(head(feat_a), dim=-1)
    lb = torch.log_softmax(head(feat_b), dim=-1)
    kl = torch.nn.functional.kl_div(lb, la, log_target=True, reduction="batchmean").item()

    return {
        "feature_max_abs_err": abs_err,
        "feature_rel_err": rel_err,
        "feature_cosine_sim": cos,
        "logit_kl": kl,
        "seq": seq, "D": D,
        "single_forward_equivalent": abs_err < 1e-2,  # bf16/fp32 roundoff tolerant
    }
