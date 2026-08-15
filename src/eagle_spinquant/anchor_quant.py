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


# ---- Phase D: hard flip-budget projection (contract §2 site policy) --------
#
# Enforced EXACTLY on the 7 uncoupled AR sites + W_rec: the W_rec e-half
# writes W_e (the shared e-fold) and the W_rec h-half writes W_h via the
# RECURRENT fold. W_first is measured-only (its folds move implicitly
# through W_e/W_h; preregistered limitation, reported).
HB_SITES = ("q", "k", "v", "o", "gate", "up", "down", "W_rec")
HB_MARGIN = 0.49


def _had_matrix(model):
    """Explicit matrix of the down-proj input FWHT (orthogonal; NOT
    assumed symmetric — the inverse uses H^T). Cached on the model."""
    H = getattr(model, "_hb_had_mat", None)
    if H is None:
        sb.add_spinquant_to_syspath()
        from utils import hadamard_utils
        n = model.Wdown.shape[1]
        eye = torch.eye(n, device=model.dev, dtype=torch.float32)
        H = hadamard_utils.matmul_hadU_cuda(eye, model.had_K,
                                            model.had_KK)
        err = (H @ H.t() - eye).abs().max()
        assert float(err) < 1e-4, f"FWHT matrix not orthogonal: {err}"
        model._hb_had_mat = H
    return H


@torch.no_grad()
def _write_masters(model, newW, R):
    """Write projected FOLDED weights back to the fp32 masters through
    the exact inverse folds (fp64). newW values are fp64 folded tensors
    for HB_SITES."""
    from .exact_quantized_rotation_forward import NH, HD
    dev = model.dev
    D = model.D
    Rt = R.t()
    R2 = (model.r2_rot.R(torch.float64).to(dev)
          if model.r2_rot is not None else model.R2_64.to(dev))
    gl = model.gl64.detach().to(dev).double()
    model.Wq.data.copy_((newW["q"] @ Rt).float())
    model.Wk.data.copy_((newW["k"] @ Rt).float())
    V1 = torch.einsum("ab,hbc->hac", R2.t(),
                      newW["v"].reshape(NH, HD, D))
    model.Wv.data.copy_((V1.reshape(NH * HD, D) @ Rt).float())
    O1 = torch.einsum("ohb,bc->ohc",
                      newW["o"].reshape(D, NH, HD), R2)
    model.Wo.data.copy_((R @ O1.reshape(D, NH * HD)).float())
    model.Wgate.data.copy_(
        ((newW["gate"] @ Rt) / gl.unsqueeze(0)).float())
    model.Wup.data.copy_(
        ((newW["up"] @ Rt) / gl.unsqueeze(0)).float())
    H = _had_matrix(model).double()
    model.Wdown.data.copy_((R @ (newW["down"] @ H.t())).float())
    a = model.alpha_exact
    model.W_e.data.copy_((newW["W_rec"][:, :D] * a).float())
    model.W_h.data.copy_((newW["W_rec"][:, D:] @ Rt).float())


@torch.no_grad()
def hard_budget_project(model, anchor, budget_frac, grads,
                        margin=HB_MARGIN):
    """Project the masters so that at most K = floor(budget_frac *
    N_total) code flips (vs the frozen anchor c0) survive across
    HB_SITES; all other weights are pulled into the anchor cell
    interior |u - c0| <= margin.

    utility(flip) = -g * (Q_new - Q_anchor): the first-order decrease of
    the task loss from KEEPING the flip (g = dL/d folded weight via the
    STE identity, captured from the current batch). Kept = deterministic
    top-K by utility (stable sort; ties resolved by site order then flat
    index). Post-conditions (hard gates): every post-projection flip is
    in the kept set, and the flip count is <= K.
    """
    assert model.train_draft_core, "hard budget needs trainable core"
    dev = model.dev
    tw = model.transformed_weights(exact=False)
    R = model.rot.R().detach().to(dev).double()
    u_by, flip_by, keep_by = {}, {}, {}
    util_parts, idx_parts = [], []
    n_total = 0
    for site in HB_SITES:
        s = anchor[site]["scale"].to(dev).float()
        c0 = anchor[site]["c0"].to(dev).float()
        u = tw[site].float() / s
        cn = torch.clamp(torch.round(u), -(MAXQ + 1), MAXQ)
        flip = cn != c0
        n_total += u.numel()
        g = grads[site].float()
        util = -(g * (s * (cn - c0)))
        idx = flip.reshape(-1).nonzero(as_tuple=True)[0]
        util_parts.append(util.reshape(-1)[idx])
        idx_parts.append((site, idx))
        u_by[site], flip_by[site] = u, flip
    K = int(budget_frac * n_total)
    cat = (torch.cat(util_parts) if util_parts
           else torch.zeros(0, device=dev))
    n_flips = int(cat.numel())
    # keep only flips the first-order model calls HELPFUL (utility > 0),
    # at most K of them — a negative-utility flip is never worth its
    # budget slot (documented deviation-free refinement of top-K)
    k_eff = min(K, int((cat > 0).sum()))
    order = torch.argsort(cat, descending=True, stable=True)
    keep_mask = torch.zeros(n_flips, dtype=torch.bool, device=dev)
    keep_mask[order[:k_eff]] = True
    off = 0
    for site, idx in idx_parts:
        keep_by[site] = idx[keep_mask[off:off + idx.numel()]]
        off += idx.numel()
    kept = k_eff
    # clamp EVERY non-kept weight into the cell interior — also the
    # unflipped boundary-huggers, so fp16 refold noise (~4e-3 in u
    # units, vs the 0.01 slack before the 0.5 rounding boundary) cannot
    # mint new flips after the master round-trip. The down inverse goes
    # through the explicit fp32 FWHT matrix whose round-trip can rarely
    # exceed the slack (observed ~1 weight in 4.5e7), so escapees are
    # pulled to the CELL CENTER and re-written (converges in one pass).
    km_by = {}
    for site in HB_SITES:
        km = torch.zeros(u_by[site].numel(), dtype=torch.bool,
                         device=dev)
        km[keep_by[site]] = True
        km_by[site] = km.reshape(u_by[site].shape)
    per_site = {}
    # retry with a TIGHTENING margin: centering an escapee perturbs its
    # whole down-proj row through the FWHT round-trip (~5e-3 in u
    # units), which can push OTHER boundary-huggers out of their cells —
    # re-clamping everything deeper into the cell each attempt makes the
    # slack outgrow the perturbation, so the loop converges
    for attempt in range(8):
        m_eff = max(margin - 0.04 * attempt, 0.25)
        newW = {}
        for site in HB_SITES:
            s = anchor[site]["scale"].to(dev).float()
            c0 = anchor[site]["c0"].to(dev).float()
            u = u_by[site]
            u_cell = torch.clamp(u, c0 - m_eff, c0 + m_eff)
            u_new = torch.where(km_by[site], u, u_cell)
            newW[site] = (s * u_new).double()
        _write_masters(model, newW, R)
        tw2 = model.transformed_weights(exact=False)
        post = lost = escaped = 0
        for site in HB_SITES:
            s = anchor[site]["scale"].to(dev).float()
            c0 = anchor[site]["c0"].to(dev)
            f2 = (codes(tw2[site], s) != c0).reshape(-1)
            nk = f2 & ~km_by[site].reshape(-1)
            ne = int(nk.sum())
            if ne:
                # pull escapees to the cell center for the next pass
                u_by[site].reshape(-1)[nk] = \
                    c0.float().reshape(-1)[nk]
                escaped += ne
            per_site[site] = dict(kept=int(keep_by[site].numel()),
                                  post=int(f2.sum()))
            post += int(f2.sum())
            lost += int(keep_by[site].numel()) - int(f2.sum())
        if escaped == 0:
            break
    assert escaped == 0, f"HB gate: {escaped} escapees after retries"
    assert post <= kept, f"HB gate: post {post} > kept {kept}"
    return dict(n_total=n_total, budget_K=K, n_flips_pre=n_flips,
                kept=kept, post=post, boundary_lost=lost,
                per_site=per_site)


def save_anchor(anchor, path):
    torch.save({k: {kk: vv for kk, vv in v.items()}
                for k, v in anchor.items()}, path)


def load_anchor(path, device="cpu"):
    a = torch.load(path, map_location=device, weights_only=True)
    for k in QSITES:
        assert k in a and "scale" in a[k] and "c0" in a[k]
    return a
