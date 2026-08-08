"""Interface arms between a (rotated/quantized) target and the DFlash draft.

Basis convention (row-vector, F.linear y = x W^T):
  rotated target residual: H^rot = H R1
  fold:  fc.weight[:, i*D:(i+1)*D]  <-  W_i @ R1
  explicit reference:      H = H^rot @ R1^T  per source branch
  embed/head restore (stock draft under rotated target):
      noise_embedding = embed_rot(ids) @ R1^T      (rotated rows -> original)
      draft_logits    = lm_head_rot(h_draft @ R1)  (original -> rotated head)
All explicit transforms run in fp32 (SEAGLE PostProjectionR1 convention).
"""
import copy

import torch


def make_ctx_transform(mode, R1=None, m=5, D=4096, mp3_scales=None):
    """Returns callable([B,S,m*D]) or None.
    mode: 'stock'|'naive'|'folded' -> None (no runtime ctx transform)
          'explicit'               -> per-branch @ R1^T (undo target rotation)
          'rotate'                 -> per-branch @ R1 (interface-only rotation
                                      at an UNROTATED target; pair with
                                      fold_wc so FP is preserved)
    mp3_scales: optional [m] tensor multiplied per branch (after transform).
    """
    R1t = R1.to(torch.float32).t().contiguous() if R1 is not None else None
    scales = None
    if mp3_scales is not None:
        scales = torch.as_tensor(mp3_scales, dtype=torch.float32)

    if mode in ("stock", "naive", "folded") and scales is None:
        return None

    R1f = R1.to(torch.float32).contiguous() if R1 is not None else None

    def _t(H):
        B, S, MD = H.shape
        x = H.reshape(B, S, m, D).to(torch.float32)
        if mode == "explicit":
            x = x @ R1t.to(H.device)
        elif mode == "rotate":
            x = x @ R1f.to(H.device)
        if scales is not None:
            x = x * scales.to(H.device).view(1, 1, m, 1)
        return x.reshape(B, S, MD).to(H.dtype)

    return _t


@torch.inference_mode()
def fold_wc(draft, R1, m=5, D=4096, mp3_scales=None):
    """Deep-copied draft with per-source-block rotation (and optional 1/m_i)
    folded into fc. fp64 staging."""
    d2 = copy.deepcopy(draft)
    W = d2.fc.weight.data.to(torch.float64)
    R = R1.to(torch.float64).to(W.device)
    for i in range(m):
        blk = W[:, i * D:(i + 1) * D] @ R
        if mp3_scales is not None:
            blk = blk / float(mp3_scales[i])
        W[:, i * D:(i + 1) * D] = blk
    d2.fc.weight.data = W.to(draft.fc.weight.dtype)
    return d2


def make_embed_head_restore(target, R1):
    """(embed_fn, head_fn) for a STOCK draft under a ROTATED target.
    Explicit fp32 boundary transforms (classified F1 in the folding audit:
    mathematically absorbable into draft-only weight views)."""
    R1f = R1.to(torch.float32)
    dev = target.device
    R1c = R1f.to(dev)
    embed = target.model.embed_tokens
    head = target.lm_head

    def embed_fn(ids):
        e = embed(ids)
        return (e.to(torch.float32) @ R1c.t()).to(e.dtype)

    def head_fn(h):
        return head((h.to(torch.float32) @ R1c).to(h.dtype))

    return embed_fn, head_fn
