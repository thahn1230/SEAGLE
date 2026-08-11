"""FIDI §6/§14: full-interface weight audit for DFlash draft quantization.

§6  W_c audit: fc.weight stock vs R1_T-folded (interfaces.fold_wc), full
    matrix + the five per-source blocks (target layers 1/8/15/22/29),
    per-row/col arrays + source-RMS imbalance.
§14 draft weight audit: q/k/v/o/gate/up/down for all 5 draft layers under
    "stock" and (with --draft-rot-ckpt) "vsq_deployed" — the gamma-fused
    R1_D/R2_D-rotated views exactly as vsq_draft_rot.RotQuantDraft
    freeze_for_eval builds them, minus quantization; k/v emit separate
    ctx (hidden_norm gamma) and noise (input_ln gamma) rows.

Per (weight, view): global wstats, per-row/col rms/absmax/p99/kurtosis,
SVD spectrum summaries, and W4 error under BOTH deployed policies:
w4_mse = SpinQuant WeightQuantizer (per-channel sym, MSE clip — the
target-side/DKVA policy) and w4_rtn = rc.rtn_sym_perchannel (the
RotQuantDraft deployed policy). Norm gammas get a stats-only table.

Writes tables/draft_weight_stats.csv, tables/draft_weight_quant_stats.csv,
tables/wc_weight_stats.csv, tables/norm_gamma_stats.csv and
raw/weights/fidi_weight_arrays.npz (axis arrays + 64x64 mean-pooled |W|
grids for 3D figures).
"""
import argparse
import csv
import os

import numpy as np
import torch

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils
from . import spinquant_target as sq
from . import interfaces
from .rc import rtn_sym_perchannel
from .dkva_weights import wstats, axis_stats, quant_report

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]
D = 4096


def axis_full(wcpu, wdev, dim):
    """dkva axis_stats (rms/absmax/p99, CPU numpy percentile) + kurtosis."""
    ax = axis_stats(wcpu, dim)
    ctr = wdev - wdev.mean(dim, keepdim=True)
    var = ctr.pow(2).mean(dim)
    ax["kurt"] = (ctr.pow(4).mean(dim) / (var ** 2 + 1e-30)).cpu()
    return ax


def svd_stats(wdev):
    s = torch.linalg.svdvals(wdev)
    e2 = s.double().pow(2)
    tot = e2.sum()
    p = e2 / tot
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    k = min(64, s.numel() - 1)
    out = {"spectral_norm": s[0].item(),
           "stable_rank": (tot / e2[0]).item(),
           "effective_rank": ent.exp().item(),
           "sv_decay_ratio_10": (s[9] / s[0]).item() if s.numel() >= 10
           else float("nan"),
           "sv_tail_energy_frac": (e2[k:].sum() / tot).item()}
    del s, e2
    return out


def w4_mse_policy(wf):
    """dkva w4_mse rebuilt with explicit device (dkva hardcodes .cuda());
    also returns the per-row clip |W|>clip => clipped (MSE shrinks scale)."""
    q = quant_utils.WeightQuantizer()
    q.configure(4, perchannel=True, sym=True, mse=True)
    q.find_params(wf)
    dq = q.quantize(wf)
    clip = q.scale.reshape(-1, 1).float() * float(q.maxq)
    return dq, clip


def quant_rows_for(name, view, wf):
    rows = []
    dq, clip = w4_mse_policy(wf)
    r = quant_report(wf, dq)
    r["w4_sat_ratio"] = (wf.abs() > clip).float().mean().item()
    rows.append({"name": name, "view": view, "policy": "w4_mse",
                 "scale_max_over_median": "", **r})
    del dq, clip
    maxq = 7
    scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq
    dq = rtn_sym_perchannel(wf, 4)
    r = quant_report(wf, dq)
    codes = torch.round(dq / scale)
    r["w4_sat_ratio"] = (codes.abs() >= maxq).float().mean().item()
    s = scale.flatten()
    rows.append({"name": name, "view": view, "policy": "w4_rtn",
                 "scale_max_over_median": (s.max() / s.median()).item(),
                 **r})
    del dq, codes, scale
    return rows


def grid64(wdev):
    a = torch.nn.functional.adaptive_avg_pool2d(
        wdev.abs().unsqueeze(0), (64, 64))
    return a.squeeze(0).cpu().numpy().astype(np.float16)


def write_csv(path, rws, lead):
    keys = sorted({k for r in rws for k in r})
    keys = [k for k in lead if k in keys] + \
        [k for k in keys if k not in lead]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rws)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--draft-rot-ckpt", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    od, dev = args.run_dir, args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel

    os.makedirs(f"{od}/tables", exist_ok=True)
    os.makedirs(f"{od}/raw/weights", exist_ok=True)

    draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sq.load_rbin(args.rbin)["R1"]
    folded = interfaces.fold_wc(draft, R1)

    stat_rows, wc_rows, qrows, norm_rows = [], [], [], []
    raw = {}
    nviews = 0

    def audit(name, view, w, table, col_arrays=False, row_arrays=False):
        nonlocal nviews
        wdev = w.detach().float().to(dev)
        wcpu = wdev.cpu()
        st = {"name": name, "view": view,
              "shape": "x".join(map(str, w.shape))}
        st.update(wstats(wcpu))
        for pre, dim in (("row", 1), ("col", 0)):
            ax = axis_full(wcpu, wdev, dim)
            for k, v in ax.items():
                st[f"{pre}_{k}_mean"] = float(v.float().mean())
                st[f"{pre}_{k}_max"] = float(v.float().max())
        st.update(svd_stats(wdev))
        table.append(st)
        qr = quant_rows_for(name, view, wdev)
        qrows.extend(qr)
        if col_arrays:
            raw[f"{name}__{view}__col_rms"] = \
                wdev.pow(2).mean(0).sqrt().cpu().numpy().astype(np.float32)
            raw[f"{name}__{view}__col_absmax"] = \
                wdev.abs().amax(0).cpu().numpy().astype(np.float32)
        if row_arrays:
            raw[f"{name}__{view}__row_rms"] = \
                wdev.pow(2).mean(1).sqrt().cpu().numpy().astype(np.float32)
        raw[f"grid__{name}__{view}"] = grid64(wdev)
        print("[fidi_w] %-24s %-12s rms %.4f absmax %.3f kurt %.2f "
              "nmse mse %.5f rtn %.5f" %
              (name, view, st["rms"], st["absmax"], st["kurtosis"],
               qr[0]["w4_nmse"], qr[1]["w4_nmse"]), flush=True)
        nviews += 1
        del wdev, wcpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return st

    # §6 W_c: stock vs R1_T-folded, full + per-source blocks
    for view, model in (("stock", draft), ("R1T_folded", folded)):
        W = model.fc.weight.data
        audit("fc", view, W, wc_rows, col_arrays=True, row_arrays=True)
        blk_rms = []
        for bi, sid in enumerate(SRC):
            b = audit(f"fc_src{sid}", view, W[:, bi * D:(bi + 1) * D],
                      wc_rows)
            blk_rms.append(b["rms"])
        imb = max(blk_rms) / (min(blk_rms) + 1e-30)
        for r in wc_rows:
            if r["view"] == view:
                r["source_rms_imbalance"] = imb

    # §14 draft layers, stock basis
    for i, layer in enumerate(draft.layers):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            audit(f"l{i}.{nm}", "stock",
                  getattr(layer.self_attn, nm).weight.data, stat_rows)
        for nm in ("gate_proj", "up_proj", "down_proj"):
            audit(f"l{i}.{nm}", "stock",
                  getattr(layer.mlp, nm).weight.data, stat_rows)

    # §14 vsq_deployed basis: RotQuantDraft's gamma-fused rotated views,
    # replicated from its freeze_for_eval WITHOUT the _wq quantization step
    if args.draft_rot_ckpt:
        from .vsq_draft_rot import RotQuantDraft
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)   # eval_al seed => same had4 draw
        ck = torch.load(args.draft_rot_ckpt + ".best", map_location="cpu",
                        weights_only=False)
        R1b = ck["R1_D"].float().to(dev)
        R2b = [t.float().to(dev) if t is not None else None
               for t in ck["R2_D"]]
        rq = RotQuantDraft(folded, w_bits=4, a_bits=4,
                           use_r2=R2b[0] is not None,
                           train_rotations=False, device=dev)
        rq.R1 = lambda: R1b
        rq.R2 = (lambda i: R2b[i]) if R2b[0] is not None else \
            (lambda i: None)
        if R2b[0] is None:
            rq.cfg["use_r2"] = False
        R1d = rq.R1()

        def deployed_views(i):
            R2 = rq.R2(i)
            yield f"l{i}.q_proj", getattr(rq, f"wq_{i}") @ R1d
            yield f"l{i}.k_proj.noise", getattr(rq, f"wk_noise_{i}") @ R1d
            yield f"l{i}.k_proj.ctx", getattr(rq, f"wk_ctx_{i}")
            yield (f"l{i}.v_proj.noise",
                   rq._headwise(getattr(rq, f"wv_noise_{i}"), R2, "out")
                   @ R1d)
            yield (f"l{i}.v_proj.ctx",
                   rq._headwise(getattr(rq, f"wv_ctx_{i}"), R2, "out"))
            yield (f"l{i}.o_proj",
                   R1d.t() @ rq._headwise(getattr(rq, f"wo_{i}"), R2, "in"))
            yield f"l{i}.gate_proj", getattr(rq, f"wg_{i}") @ R1d
            yield f"l{i}.up_proj", getattr(rq, f"wu_{i}") @ R1d
            yield (f"l{i}.down_proj",
                   R1d.t() @ (getattr(rq, f"wd_{i}") @ rq.had4))

        for i in range(rq.n_layers):
            for nm, w in deployed_views(i):
                audit(nm, "vsq_deployed", w, stat_rows,
                      col_arrays=nm.endswith(".ctx"))
                del w
        del rq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del folded

    # norm gammas (stats only)
    norms = {"hidden_norm": draft.hidden_norm.weight,
             "final_norm": draft.norm.weight}
    for i, layer in enumerate(draft.layers):
        norms[f"l{i}.input_ln"] = layer.input_layernorm.weight
        norms[f"l{i}.post_ln"] = layer.post_attention_layernorm.weight
        norms[f"l{i}.q_norm"] = layer.self_attn.q_norm.weight
        norms[f"l{i}.k_norm"] = layer.self_attn.k_norm.weight
    for nm, g in norms.items():
        gf = g.data.float()
        norm_rows.append({"name": nm,
                          "rms": gf.pow(2).mean().sqrt().item(),
                          "absmax": gf.abs().max().item(),
                          "min": gf.min().item(),
                          "max": gf.max().item(),
                          "mean": gf.mean().item(),
                          "frac_below_0.5": (gf < 0.5).float().mean().item(),
                          "frac_above_2": (gf > 2).float().mean().item()})

    write_csv(f"{od}/tables/draft_weight_stats.csv", stat_rows,
              ("name", "view", "shape"))
    write_csv(f"{od}/tables/draft_weight_quant_stats.csv", qrows,
              ("name", "view", "policy"))
    write_csv(f"{od}/tables/wc_weight_stats.csv", wc_rows,
              ("name", "view", "shape"))
    write_csv(f"{od}/tables/norm_gamma_stats.csv", norm_rows, ("name",))
    np.savez_compressed(f"{od}/raw/weights/fidi_weight_arrays.npz", **raw)
    print(f"[fidi_weights] DONE {nviews} weight-views")


if __name__ == "__main__":
    main()
