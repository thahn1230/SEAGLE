"""Quantized-anchor machinery (QAT quantized-anchor causal study, 2026-08-14).

Defines the FROZEN anchor quantizer over the deployed FOLDED weight basis:

  anchor scale s = per-out-channel symmetric MSE-clip scale captured ONCE
                   from the PTQ anchor's folded weights (official
                   SpinQuant WeightQuantizer semantics, maxq=7)
  codes  c(W)   = clamp(round(fold(W)/s), -(maxq+1), maxq)   # in [-8, 7]
  c0            = c(W0)  — the deployed anchor code configuration

With rotations/scales frozen, folds are fixed linear maps, so cells are
well-defined deployment-visible objects for the whole study. All anchor
experiments (training AND evaluation of constructed/anchored models) use
these frozen scales; activations stay dynamic per-token asymmetric.

Provides:
  capture_anchor(tw)           anchor state {site: {scale, c0}}
  frozen_fake_quant(w, s)      STE fake-quant under a FROZEN scale
  codes(w, s)                  int8 codes
  cell_loss(tw, anchor, rho)   L_cell = mean relu(|u - c0| - rho)^2
  drift_metrics(...)           H_Q, D_Q, D_FP (+ per-site)
  save/load_anchor             checkpoint-stable anchor state
"""
from __future__ import annotations

import torch

from . import fake_w4a4_draft as fq
from . import spinquant_bridge as sb

MAXQ = 7
QSITES = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")


def _official_quantizer():
    sb.add_spinquant_to_syspath()
    from utils import quant_utils
    q = quant_utils.WeightQuantizer()
    q.configure(bits=4, perchannel=True, sym=True, mse=True)
    return q


def capture_anchor(tw):
    """tw: dict of FOLDED anchor weights (transformed_weights output).
    Runs the official MSE-clip search ONCE per site; returns
    {site: {"scale": (out,1) fp32, "c0": int8 codes}}. Deterministic."""
    anchor = {}
    for k in QSITES:
        w = tw[k].detach().float()
        q = _official_quantizer()
        q.find_params(w)
        deq, c, s = q.fake_quantize(w)
        anchor[k] = dict(scale=s.detach().float().cpu(),
                         c0=c.detach().to(torch.int8).cpu())
        # invariant: official dequant == frozen reconstruction, bitwise
        assert torch.equal(deq.float().cpu(),
                           (anchor[k]["scale"] * anchor[k]["c0"].float()))
    return anchor


def codes(w, scale, maxq=MAXQ):
    """int codes of a folded weight under a FROZEN scale (official
    convention: clamp(round(w/s), -(maxq+1), maxq))."""
    return torch.clamp(torch.round(w.float() / scale.to(w.device)),
                       -(maxq + 1), maxq).to(torch.int8)


def frozen_fake_quant(w, scale, maxq=MAXQ):
    """STE fake-quant under a FROZEN anchor scale: forward = s*clamp(
    round(w/s)), backward = identity. Matches the official sym
    fake_quantize applied with the same fixed scale, bitwise."""
    s = scale.to(w.device)
    with torch.no_grad():
        qw = (s * torch.clamp(torch.round(w.detach().float() / s),
                              -(maxq + 1), maxq)).to(w.dtype)
    if not w.requires_grad:
        return qw
    return w + (qw - w).detach()


def cell_loss(tw, anchor, rho=0.45, sites=QSITES):
    """L_cell = mean_i relu(|u_i - c0_i| - rho)^2 over the given sites,
    u = fold(W)/s (differentiable through the fold graph). Zero inside
    |u-c0| <= rho; grows before the cell boundary at 0.5."""
    total = 0.0
    n = 0
    for k in sites:
        a = anchor[k]
        u = tw[k].float() / a["scale"].to(tw[k].device)
        d = (u - a["c0"].to(u.device).float()).abs()
        pen = torch.relu(d - rho) ** 2
        total = total + pen.sum()
        n += pen.numel()
    return total / n


@torch.no_grad()
def drift_metrics(tw, anchor, tw0=None):
    """H_Q (code flip rate), D_Q (normalized dequantized drift under
    FROZEN scales), and per-site rates; D_FP if tw0 (anchor folds) given."""
    flips = tot = 0
    dq_num = dq_den = 0.0
    dfp_num = dfp_den = 0.0
    per = {}
    for k in QSITES:
        a = anchor[k]
        s = a["scale"].to(tw[k].device)
        c = codes(tw[k], s)
        c0 = a["c0"].to(c.device)
        f = (c != c0)
        per[k] = dict(flip=float(f.float().mean()),
                      n=int(f.numel()))
        flips += int(f.sum())
        tot += f.numel()
        qw, qw0 = s * c.float(), s * c0.float()
        dq_num += float((qw - qw0).pow(2).sum())
        dq_den += float(qw0.pow(2).sum())
        if tw0 is not None:
            dfp_num += float((tw[k].float() - tw0[k].float()).pow(2).sum())
            dfp_den += float(tw0[k].float().pow(2).sum())
    out = dict(H_Q=flips / tot, D_Q=dq_num / max(dq_den, 1e-12),
               per_site=per, n_total=tot, n_flipped=flips)
    if tw0 is not None:
        out["D_FP"] = dfp_num / max(dfp_den, 1e-12)
    return out


def save_anchor(anchor, path):
    torch.save({k: {kk: vv for kk, vv in v.items()}
                for k, v in anchor.items()}, path)


def load_anchor(path, device="cpu"):
    a = torch.load(path, map_location=device, weights_only=True)
    for k in QSITES:
        assert k in a and "scale" in a[k] and "c0" in a[k]
    return a
