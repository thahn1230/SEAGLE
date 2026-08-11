"""FIDI figure generation (FIG_A..FIG_E + FIG_INDEX.md).

Reads only artifacts of the FIDI run
(runs/dflash_full_interface_distribution_intervention_20260811_074001):
reservoir activation samples (raw/activations/*.npz), weight arrays
(raw/weights/fidi_weight_arrays.npz, alignment_scatter.npz) and the
summary tables (tables/*.csv).  FIG_B additionally loads the real draft
fc weight (DFlashDraftModel, CPU/bfloat16) and the frozen R1_T rotation
(outputs/rotations/llama31_w4a4kv16_s1/R.bin) to compute W4-RTN
dequantization-error grids for the stock and R1T-folded views.

CPU only.  Saves PDF + PNG (150 dpi) pairs into <run>/figs/.

Usage:  python seagle_port/fidi_figs.py  [--run-dir DIR] [--skip-model]
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/data/hf_cache_thahn1230")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

RUN = ("/home/thahn1230/dflash_workspace/dflash/runs/"
       "dflash_full_interface_distribution_intervention_20260811_074001")
RBIN = "/home/thahn1230/dflash_workspace/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC_LAYERS = [1, 8, 15, 22, 29]
D = 4096
DS = "gsm8k"

# Config display names (verbatim config first, gloss second)
CFG_NAME = {
    "R0": "R0 (FP16)",
    "R1": "R1 (vanilla-SQ)",
    "R2": "R2 (target+draft vanilla-SQ)",
    "R3": "R3 (+R_C)",
}
# Okabe-Ito, fixed assignment per config (never re-ordered)
CFG_COLOR = {"R0": "#0072B2", "R1": "#E69F00", "R3": "#009E73"}

WARN = []


def warn(msg):
    WARN.append(msg)
    print("[fidi_figs][quirk] " + msg, flush=True)


def load_rows(run, cfg, tensor, ds=DS):
    path = f"{run}/raw/activations/{cfg}__{ds}__{tensor}.npz"
    if not os.path.exists(path):
        warn(f"missing tensor file: {path}")
        return None
    r = np.load(path)["rows"].astype(np.float32)
    if r.size == 0:
        warn(f"empty reservoir: {path}")
        return None
    if not np.isfinite(r).all():
        warn(f"non-finite values in {path}")
    return r


def load_channels(run, cfg, ds=DS):
    return np.load(f"{run}/raw/activations/{cfg}__{ds}__channels.npz")


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def csv_val(rows, col, **match):
    hit = [r for r in rows if all(r[k] == v for k, v in match.items())]
    if len(hit) != 1:
        warn(f"csv lookup {match} -> {len(hit)} rows (expected 1)")
        if not hit:
            return np.nan
    return float(hit[0][col])


def sample_rows(mat, n, seed=0):
    if mat.shape[0] < n:
        warn(f"reservoir has only {mat.shape[0]} rows (< {n} requested)")
        n = mat.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(
        mat.shape[0], size=n, replace=False))
    return mat[idx]


def pool_max_abs(mat, factor):
    """|x| max-pooled along channels by `factor`."""
    n, c = mat.shape
    assert c % factor == 0, (c, factor)
    return np.abs(mat).reshape(n, c // factor, factor).max(axis=2)


def add_cbar(fig, ax, im, label):
    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def annotate_sources(ax, scale=1.0, y=0.93, fontsize=7):
    """Vertical lines at source boundaries 4096*i (times `scale`) with
    H1/H8/H15/H22/H29 block labels."""
    for i in range(1, 5):
        ax.axvline(i * D * scale, color="k", ls="--", lw=0.6, alpha=0.7)
    for i, lyr in enumerate(SRC_LAYERS):
        ax.text((i + 0.5) * D * scale, y, f"H{lyr}",
                transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=fontsize,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))


def save(fig, figdir, name):
    fig.savefig(f"{figdir}/{name}.pdf")
    fig.savefig(f"{figdir}/{name}.png", dpi=150)
    plt.close(fig)
    print(f"[fidi_figs] wrote {figdir}/{name}.pdf/.png", flush=True)


# ---------------------------------------------------------------- FIG_A
def fig_a(run, figdir):
    """Target hidden concat B_concat: per-channel absmax + |act| heatmaps,
    R0 vs R2."""
    cfgs = ["R0", "R2"]
    ch = {c: load_channels(run, c)["B_concat__absmax"] for c in cfgs}
    heat, raws = {}, {}
    for c in cfgs:
        raws[c] = load_rows(run, c, "B_concat")
        heat[c] = pool_max_abs(sample_rows(raws[c], 160), 64)  # 160 x 320
    vmax = float(np.quantile(np.concatenate(
        [heat[c].ravel() for c in cfgs]), 0.999))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    ymin = max(1e-3, min(ch[c][ch[c] > 0].min() for c in cfgs) * 0.8)
    ymax = max(ch[c].max() for c in cfgs) * 1.6
    for j, c in enumerate(cfgs):
        ax = axes[0, j]
        ax.plot(np.arange(ch[c].size), ch[c], lw=0.35,
                color=CFG_COLOR.get(c, "#555555"))
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, ch[c].size)
        annotate_sources(ax)
        ax.set_title(f"{CFG_NAME[c]} — B_concat per-channel absmax",
                     fontsize=9)
        ax.set_xlabel("channel index (5 x 4096 source blocks)")
        ax.set_ylabel("per-channel absmax (log)")
    for j, c in enumerate(cfgs):
        ax = axes[1, j]
        im = ax.imshow(heat[c], aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=vmax,
                       extent=(0, raws[c].shape[1], heat[c].shape[0], 0),
                       interpolation="nearest")
        annotate_sources(ax, y=0.97)
        ax.set_title(f"{CFG_NAME[c]} — |activation| "
                     "(160 sampled tokens)", fontsize=9)
        ax.set_xlabel("channel index (max-pooled x64)")
        ax.set_ylabel("sampled token (reservoir row)")
        add_cbar(fig, ax, im,
                 f"|act|, shared scale vmax={vmax:.2f} (99.9th pct of union)")
    fig.suptitle("FIG_A  target hidden concat B_concat (gsm8k): "
                 "R0 (FP16) vs R2 (target+draft vanilla-SQ)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, figdir, "FIG_A_hsrc")


# ---------------------------------------------------------------- FIG_B
def fig_b(run, figdir, skip_model=False):
    """|W_c| grids (stock vs R1T_folded) + W4-RTN dequant-error grids
    from the real fc weight."""
    wz = np.load(f"{run}/raw/weights/fidi_weight_arrays.npz")
    g_stock = wz["grid__fc__stock"].astype(np.float32)
    g_fold = wz["grid__fc__R1T_folded"].astype(np.float32)

    err = {}
    if not skip_model:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from dflash.model import DFlashDraftModel
        print("[fidi_figs] loading draft model (CPU, bf16)...", flush=True)
        draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
        W = draft.fc.weight.data.clone()               # [4096, 20480] bf16
        R1 = torch.load(RBIN, map_location="cpu",
                        weights_only=False)["R1"]      # [4096, 4096]
        Wf = W.to(torch.float64)
        R = R1.to(torch.float64)
        for i in range(5):
            Wf[:, i * D:(i + 1) * D] = Wf[:, i * D:(i + 1) * D] @ R
        W_fold = Wf.to(W.dtype)                        # match interfaces.fold_wc
        del draft, Wf, R

        def rtn_err_grid(w_t):
            w = w_t.float()
            # per-row symmetric 4-bit RTN (rc.rtn_sym_perchannel)
            maxq = 7
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxq
            q = torch.clamp(torch.round(w / scale), -maxq - 1, maxq)
            e = (w - q * scale).abs()
            g = e.reshape(64, w.shape[0] // 64,
                          64, w.shape[1] // 64).mean(dim=(1, 3))
            return g.numpy(), w

        for name, w_t, g_ref in (("stock", W, g_stock),
                                 ("R1T_folded", W_fold, g_fold)):
            err[name], wfl = rtn_err_grid(w_t)
            g_chk = wfl.abs().reshape(64, wfl.shape[0] // 64, 64,
                                      wfl.shape[1] // 64
                                      ).mean(dim=(1, 3)).numpy()
            rel = float(np.abs(g_chk - g_ref).max() / g_ref.max())
            print(f"[fidi_figs] |W| grid check ({name}): "
                  f"max rel diff vs stored = {rel:.4f}", flush=True)
            if rel > 0.02:
                warn(f"recomputed fc {name} |W| grid deviates from stored "
                     f"grid by {rel:.3f} (rbin/model mismatch?)")
            del wfl
    else:
        warn("FIG_B dequant-error panels skipped (--skip-model)")

    nrow = 2 if err else 1
    fig, axes = plt.subplots(nrow, 2, figsize=(11.5, 3.6 * nrow),
                             squeeze=False)
    v_w = float(np.quantile(np.stack([g_stock, g_fold]), 0.999))
    for j, (g, view) in enumerate(((g_stock, "stock"),
                                   (g_fold, "R1T_folded"))):
        ax = axes[0, j]
        im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=0.0, vmax=v_w,
                       extent=(0, 5 * D, D, 0), interpolation="nearest")
        annotate_sources(ax, y=0.96)
        ax.set_title(f"W_c = fc.weight — {view}"
                     + ("  (R1: vanilla-SQ target rotation folded)"
                        if view == "R1T_folded" else "  (R0: FP16 layout)"),
                     fontsize=9)
        ax.set_xlabel("input channel (5 x 4096 source blocks; 64x64 grid)")
        ax.set_ylabel("output channel")
        add_cbar(fig, ax, im,
                 f"mean|W| per block, shared vmax={v_w:.4f} (99.9th pct)")
    if err:
        v_e = float(np.quantile(np.stack([err["stock"],
                                          err["R1T_folded"]]), 0.999))
        for j, view in enumerate(("stock", "R1T_folded")):
            ax = axes[1, j]
            im = ax.imshow(err[view], aspect="auto", cmap="viridis",
                           vmin=0.0, vmax=v_e, extent=(0, 5 * D, D, 0),
                           interpolation="nearest")
            annotate_sources(ax, y=0.96)
            ax.set_title(f"W4-RTN dequant error |W - dq(W)| — {view}",
                         fontsize=9)
            ax.set_xlabel("input channel (5 x 4096 source blocks; "
                          "64x64 grid)")
            ax.set_ylabel("output channel")
            add_cbar(fig, ax, im, "mean|W - dq(W)| per block, shared "
                     f"vmax={v_e:.5f} (99.9th pct)")
    fig.suptitle("FIG_B  W_c (draft fc) stock vs R1T_folded: magnitude "
                 "layout and per-row sym W4-RTN error", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, figdir, "FIG_B_wc")


# ---------------------------------------------------------------- FIG_C
def fig_c(run, figdir):
    """H_t before/after R_C (config R3, gsm8k)."""
    before = load_rows(run, "R3", "C_Ht_fp")
    after = load_rows(run, "R3", "C_Ht_rc_fp")
    hb = pool_max_abs(sample_rows(before, 160), 8)   # 160 x 512
    ha = pool_max_abs(sample_rows(after, 160), 8)
    vmax = float(np.quantile(np.concatenate([hb.ravel(), ha.ravel()]),
                             0.999))
    ch = load_channels(run, "R3")
    am_b = ch["C_Ht_fp__absmax"]
    am_a = ch["C_Ht_rc_fp__absmax"]

    ht = read_csv_rows(f"{run}/tables/ht_stats.csv")
    qp = read_csv_rows(f"{run}/tables/ht_qparam_stats.csv")
    kurt = [csv_val(ht, "kurtosis", config="R3", dataset=DS, tensor=t)
            for t in ("S3_Ht_dep", "S3_Ht_rc_dep")]
    nmse = [csv_val(qp, "nmse_mean", config="R3", dataset=DS, tensor=t)
            for t in ("S3_Ht_dep", "S3_Ht_rc_dep")]

    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 4)
    panels = ((fig.add_subplot(gs[0, 0:2]), hb,
               "(a) R3 (+R_C) — H_t before R_C (C_Ht_fp)"),
              (fig.add_subplot(gs[0, 2:4]), ha,
               "(b) R3 (+R_C) — H_t after R_C (C_Ht_rc_fp)"))
    for ax, h, title in panels:
        im = ax.imshow(h, aspect="auto", cmap="viridis", vmin=0.0,
                       vmax=vmax, extent=(0, D, h.shape[0], 0),
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("channel index (max-pooled x8)")
        ax.set_ylabel("sampled token (reservoir row)")
        add_cbar(fig, ax, im,
                 f"|act|, shared scale vmax={vmax:.2f} (99.9th pct of union)")

    axc = fig.add_subplot(gs[1, 0:2])
    x = np.arange(D)
    axc.plot(x, am_b, lw=0.35, color="#0072B2",
             label="before R_C (C_Ht_fp)")
    axc.plot(x, am_a, lw=0.35, color="#E69F00", alpha=0.85,
             label="after R_C (C_Ht_rc_fp)")
    axc.set_yscale("log")
    axc.set_xlim(0, D)
    axc.set_title("(c) per-channel absmax, before vs after R_C", fontsize=9)
    axc.set_xlabel("channel index")
    axc.set_ylabel("per-channel absmax (log)")
    axc.legend(fontsize=7, loc="upper right")

    labels = ["before R_C\n(S3_Ht_dep)", "after R_C\n(S3_Ht_rc_dep)"]
    for ax, vals, ylab, ttl, fmt in (
            (fig.add_subplot(gs[1, 2]), kurt, "kurtosis",
             "(d1) H_t kurtosis (R3, gsm8k)", "{:.1f}"),
            (fig.add_subplot(gs[1, 3]), nmse, "A4 NMSE (nmse_mean)",
             "(d2) H_t A4-quant NMSE (R3, gsm8k)", "{:.4f}")):
        bars = ax.bar([0, 1], vals, width=0.6,
                      color=["#0072B2", "#E69F00"])
        ax.set_xticks([0, 1], labels, fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=8.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("FIG_C  H_t before/after context rotation R_C — "
                 "config R3 (+R_C), gsm8k", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, figdir, "FIG_C_ht")


# ---------------------------------------------------------------- FIG_D
def fig_d(run, figdir):
    """Stored ctx-K heatmaps R0/R1/R3 for layers 0/2/4 + KV4 NMSE bars."""
    layers = [0, 2, 4]
    cfgs = ["R0", "R1", "R3"]
    kv = read_csv_rows(f"{run}/tables/hypothetical_kv_quant.csv")

    fig, axes = plt.subplots(3, 4, figsize=(15.0, 8.6))
    for i, L in enumerate(layers):
        heat = {}
        for c in cfgs:
            r = load_rows(run, c, f"S4_k_stored_ctx_l{L}")
            heat[c] = pool_max_abs(sample_rows(r, 100), 4)   # 100 x 256
        vmax = float(np.quantile(np.concatenate(
            [heat[c].ravel() for c in cfgs]), 0.999))
        for j, c in enumerate(cfgs):
            ax = axes[i, j]
            im = ax.imshow(heat[c], aspect="auto", cmap="viridis",
                           vmin=0.0, vmax=vmax,
                           extent=(0, 1024, heat[c].shape[0], 0),
                           interpolation="nearest")
            ax.set_title(f"{CFG_NAME[c]} — K stored ctx, layer {L}",
                         fontsize=8.5)
            ax.set_xlabel("channel index (max-pooled x4)", fontsize=8)
            ax.set_ylabel("sampled token", fontsize=8)
            add_cbar(fig, ax, im,
                     f"|K|, shared scale vmax={vmax:.2f} (99.9th pct)")
        axb = axes[i, 3]
        width = 0.25
        for j, c in enumerate(cfgs):
            vals = [csv_val(kv, "nmse_mean", config=c, dataset=DS,
                            tensor=f"KV4_{t}_ctx_l{L}") for t in ("k", "v")]
            bars = axb.bar(np.arange(2) + (j - 1) * width, vals, width,
                           color=CFG_COLOR[c], label=CFG_NAME[c])
            for b, v in zip(bars, vals):
                axb.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                         ha="center", va="bottom", fontsize=5.5)
        axb.set_xticks([0, 1], ["k_ctx", "v_ctx"], fontsize=8)
        axb.set_ylabel("KV4 NMSE (nmse_mean)", fontsize=8)
        axb.set_title(f"KV4 quant NMSE, layer {L} (gsm8k)", fontsize=8.5)
        if i == 0:
            axb.legend(fontsize=6.5)
    fig.suptitle("FIG_D  stored context-K cache and hypothetical KV4 "
                 "quantization NMSE — R0 (FP16) vs R1 (vanilla-SQ) vs "
                 "R3 (+R_C), gsm8k", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, figdir, "FIG_D_kv")


# ---------------------------------------------------------------- FIG_E
def fig_e(run, figdir):
    """Activation channel RMS vs weight column norm, per layer x {K,V}."""
    z = np.load(f"{run}/raw/weights/alignment_scatter.npz")
    fig, axes = plt.subplots(5, 2, figsize=(8.0, 15.5))
    for L in range(5):
        for j, kv in enumerate(("k", "v")):
            ax = axes[L, j]
            act = z[f"scatter_l{L}_{kv}_act"]
            w = z[f"scatter_l{L}_{kv}_w"]
            r = float(np.corrcoef(act, w)[0, 1])
            pos = (act > 0) & (w > 0)
            if pos.sum() < act.size:
                warn(f"FIG_E l{L} {kv}: {act.size - int(pos.sum())} "
                     "non-positive channel values dropped from log axes")
            ax.scatter(act[pos], w[pos], s=3, alpha=0.25, lw=0,
                       color="#0072B2", rasterized=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"layer {L}, {kv.upper()}_proj — "
                         f"Pearson r = {r:.3f}", fontsize=8.5)
            ax.set_xlabel("activation channel RMS (log)", fontsize=8)
            ax.set_ylabel("weight column norm (log)", fontsize=8)
    fig.suptitle("FIG_E  activation-vs-weight column alignment, draft "
                 "ctx K/V projections (per-channel; r computed in raw "
                 "space)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, figdir, "FIG_E_scatter")


INDEX = """# FIDI figures (gsm8k headline panels)

Generated by `seagle_port/fidi_figs.py` from the FIDI run artifacts.
Config names: R0 = FP16 baseline, R1 = vanilla-SQ (target SpinQuant),
R2 = target+draft vanilla-SQ, R3 = +R_C (context rotation).

- **FIG_A_hsrc** (.pdf/.png): Target hidden concat `B_concat` under R0 (FP16)
  vs R2 (target+draft vanilla-SQ) — per-channel absmax profiles (log y, source
  boundaries H1/H8/H15/H22/H29 marked) over per-token |activation| heatmaps on
  a shared color scale. Shows how the rotated pipeline flattens the extreme
  per-source outlier channels of the FP16 concat.

- **FIG_B_wc** (.pdf/.png): Draft `fc` weight W_c as 64x64 mean-|W| grids,
  stock vs R1T_folded, with matching per-row symmetric W4-RTN dequantization
  error grids |W - dq(W)| computed from the real fc weight (shared color scale
  within each row). Shows how folding the target rotation R1_T into W_c
  redistributes column structure and what that does to 4-bit RTN error.

- **FIG_C_ht** (.pdf/.png): H_t before vs after the context rotation R_C in
  config R3 (+R_C): |activation| heatmaps on a shared scale, a per-channel
  absmax overlay (log y), and bars for kurtosis and A4-quant NMSE
  (S3_Ht_dep vs S3_Ht_rc_dep). Shows R_C removing heavy-tailed outlier
  channels (kurtosis 195 -> 2.9) and cutting the A4 quantization NMSE
  ~25x (0.446 -> 0.018).

- **FIG_D_kv** (.pdf/.png): Stored context-K cache heatmaps for layers 0/2/4
  under R0 (FP16), R1 (vanilla-SQ) and R3 (+R_C) on a shared per-layer scale,
  with grouped bars of hypothetical KV4 nmse_mean (k_ctx / v_ctx) per layer.
  Shows the stored KV distributions and that KV4 error is nearly config-
  invariant at this interface.

- **FIG_E_scatter** (.pdf/.png): The activation_vs_weight_column_scatter —
  per-channel activation RMS vs weight column norm for the draft ctx K/V
  projections, 5 layers x {K, V}, log-log, with raw-space Pearson r per panel.
  Shows whether activation outlier channels align with heavy weight columns
  (the premise behind smoothing-based interventions).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--skip-model", action="store_true",
                    help="skip FIG_B dequant panels (no model load)")
    args = ap.parse_args()
    run = args.run_dir
    figdir = f"{run}/figs"
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "figure.dpi": 100})

    fig_a(run, figdir)
    fig_b(run, figdir, skip_model=args.skip_model)
    fig_c(run, figdir)
    fig_d(run, figdir)
    fig_e(run, figdir)

    with open(f"{figdir}/FIG_INDEX.md", "w") as f:
        f.write(INDEX)
    print(f"[fidi_figs] wrote {figdir}/FIG_INDEX.md", flush=True)

    if WARN:
        print("[fidi_figs] data quirks:")
        for w in WARN:
            print("  - " + w)
    else:
        print("[fidi_figs] no data quirks detected.")


if __name__ == "__main__":
    main()
