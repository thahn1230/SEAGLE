"""P3: exact embedding-scaling reparameterization for the concat projection.

    e' = alpha * e          (draft-only embedding table E' = alpha*E)
    W_e' = W_e / alpha      (e-slice of BOTH first and recurrent projections)

FP16/FP32 function is unchanged (Gate E); under a SHARED per-token
asymmetric A4 scale over concat([e', h]) the effective ranges of the two
slices are brought closer, and under W4 per-output-channel weight quant the
e-columns no longer dominate the row absmax.

Alpha is calibrated OFFLINE on captured projection inputs (calibration
prompts disjoint from the MT-bench eval set), optimizing projection-OUTPUT
reconstruction error after BOTH W4 weight quant and shared A4 input quant —
never on MT-bench acceptance length.
"""
from __future__ import annotations

import json
import math

import torch

from . import fake_w4a4_draft as fq


@torch.no_grad()
def capture_projection_inputs(adapter, model, ids_list, tree, max_rows=8192,
                              max_steps=48):
    """Run greedy EAGLE generation on calibration prompts with the (FP16)
    adapter installed and capture fc inputs z=[e|h] for first and recurrent
    selects separately. Returns dict(first=(N,2D) fp32 cpu, recurrent=...)."""
    store = {"first": [], "recurrent": []}
    split = adapter.split
    orig_fwd = split.forward

    def hooked(z):
        sel = split.select
        if sel in store and sum(t.shape[0] for t in store[sel]) < max_rows:
            store[sel].append(
                z.detach().reshape(-1, z.shape[-1]).float().cpu())
        return orig_fwd(z)

    split.forward = hooked
    try:
        for ids in ids_list:
            gen = model.ea_generate(ids, temperature=0.0,
                                    max_steps=max_steps,
                                    tree_choices=tree)
            for _ in gen:
                pass
            if all(sum(t.shape[0] for t in store[k]) >= max_rows
                   for k in store):
                break
    finally:
        split.forward = orig_fwd
    return {k: torch.cat(v)[:max_rows] for k, v in store.items() if v}


@torch.no_grad()
def _shared_a4_fake_quant(x, bits=4):
    """Per-token asymmetric fake quant with ONE shared scale over the last
    dim (the ordinary concat policy)."""
    qmax = 2 ** bits - 1
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    scale = (mx - mn).clamp(min=1e-8) / qmax
    q = ((x - mn) / scale).round().clamp(0, qmax)
    return q * scale + mn


@torch.no_grad()
def alpha_objective(W_first, W_rec, bias, Z_first, Z_rec, alpha,
                    w_bits=4, a_bits=4):
    """Reconstruction error of the reparameterized quantized projection vs
    the FP reference, for one alpha. Returns per-path metrics dict."""
    out = {}
    D = W_first.shape[1] // 2
    for name, W, Z in (("first", W_first, Z_first),
                       ("recurrent", W_rec, Z_rec)):
        if Z is None:
            continue
        Wf = W.float().clone()
        y_ref = Z @ Wf.t() + bias.float()
        Wp = Wf.clone()
        Wp[:, :D] = Wp[:, :D] / alpha
        Zp = Z.clone()
        Zp[:, :D] = Zp[:, :D] * alpha
        Wq = fq._weight_fake_quant(Wp, w_bits).float()
        Zq = _shared_a4_fake_quant(Zp, a_bits)
        y = Zq @ Wq.t() + bias.float()
        err = y - y_ref
        nmse = float((err ** 2).sum() / (y_ref ** 2).sum().clamp(min=1e-12))
        rel_l2 = float(err.norm() / y_ref.norm().clamp(min=1e-12))
        cos = float(torch.nn.functional.cosine_similarity(
            y.flatten(), y_ref.flatten(), dim=0))
        chan_max = float(err.abs().amax(dim=0).max())
        # saturation proxy: fraction of e-slice values at the shared-grid rails
        qmax = 2 ** a_bits - 1
        mn = Zp.min(dim=-1, keepdim=True).values
        mx = Zp.max(dim=-1, keepdim=True).values
        scale = (mx - mn).clamp(min=1e-8) / qmax
        codes = ((Zp - mn) / scale).round().clamp(0, qmax)
        e_codes = codes[:, :D]
        e_sat = float(((e_codes == 0) | (e_codes == qmax)).float().mean())
        w_absmax_ratio = float(Wp[:, :D].abs().max()
                               / Wp[:, D:].abs().max().clamp(min=1e-12))
        out[name] = dict(nmse=nmse, rel_l2=rel_l2, cos=cos,
                         chan_max=chan_max, e_rail_frac=e_sat,
                         w_e_over_h_absmax=w_absmax_ratio)
    return out


@torch.no_grad()
def sweep_alpha(W_first, W_rec, bias, Z_first, Z_rec, k_min=-8.0, k_max=8.0,
                k_step=0.25, w_bits=4, a_bits=4):
    """Deterministic alpha grid sweep. Objective = sum of projection-output
    NMSE over first+recurrent (weight AND activation quant in the loop).
    Returns (best_alpha, rows)."""
    rows = []
    best = (None, math.inf)
    k = k_min
    while k <= k_max + 1e-9:
        alpha = 2.0 ** k
        m = alpha_objective(W_first, W_rec, bias, Z_first, Z_rec, alpha,
                            w_bits=w_bits, a_bits=a_bits)
        obj = sum(v["nmse"] for v in m.values())
        row = dict(k=round(k, 2), alpha=alpha, objective_nmse_sum=obj)
        for path, v in m.items():
            for kk, vv in v.items():
                row[f"{path}_{kk}"] = round(vv, 6)
        rows.append(row)
        if obj < best[1]:
            best = (alpha, obj)
        k += k_step
    return best[0], rows
