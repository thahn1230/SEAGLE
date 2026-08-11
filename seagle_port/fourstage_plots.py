"""Four-stage rotation-distribution figures (reference-style 3D maps).

Consumes fourstage_capture outputs + the deployed W_c weights. Style: light
background, 3D perspective, thin vertical structures with visible outlier
spikes; x = channel, y = token/sample (or output channel), z = |value|.
Every BEFORE/AFTER primary pair shares zlim = [0, max(before, after)] and
camera; an `_autoscale` twin is also emitted. zmax values land in
plots/plot_metadata.json. Deterministic: pooled plot rows are evenly-spaced
(seed-free rule fixed before data was seen); a separate WORST_CASE_DIAGNOSTIC
figure shows the max-absmax rows and is labeled as such.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
DSS = ("mtbench", "gsm8k", "humaneval", "sharegpt")
ROWS = 256
META = {}
ELEV, AZIM = 28, -60
SRC_IDS = [1, 8, 15, 22, 29]


def pooled_rows(VR, suffix, rows=ROWS, width=None):
    chunks = []
    for ds in DSS:
        p = f"{VR}/raw_plot_samples/{ds}__{suffix}.npz"
        if os.path.exists(p):
            chunks.append(np.load(p)["rows"].astype(np.float32))
    if not chunks:
        raise FileNotFoundError(suffix)
    X = np.concatenate(chunks)
    idx = np.linspace(0, X.shape[0] - 1, min(rows, X.shape[0])).astype(int)
    return X[idx], idx


def pool_cols(A, out_cols):
    n, d = A.shape
    f = max(1, d // out_cols)
    d2 = (d // f) * f
    return np.abs(A[:, :d2]).reshape(n, d // f, f).max(-1), f


def surf3d(ax, Z, xlab, ylab, zmax, title, xf=1, boundaries=None,
           blabels=None, cmax=None):
    # value-range -> color: PowerNorm with the color range CLIPPED at the
    # pair-level p99.5 (cmax), so sparse tall outliers saturate to RED while
    # the bulk carpet stays blue. The z AXIS keeps the true magnitude
    # (zlim = shared zmax); only the COLOR saturates — stated on the
    # colorbar and in plot_metadata.json. cmax/norm shared across a pair.
    from matplotlib import colors as mcolors
    n, d = Z.shape
    xs = np.arange(d) * xf
    ys = np.arange(n)
    Xg, Yg = np.meshgrid(xs, ys)
    cv = max(cmax if cmax is not None else zmax, 1e-9)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=cv, clip=True)
    surf = ax.plot_surface(Xg, Yg, Z, cmap="coolwarm", norm=norm,
                           rcount=min(n, 128), ccount=min(d, 1024),
                           linewidth=0, antialiased=False)
    cb = plt.colorbar(surf, ax=ax, fraction=0.035, pad=0.08, shrink=0.7)
    cb.set_label("|value| color (red = ≥ pair p99.5, PowerNorm γ=0.5)",
                 fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(xlab, labelpad=8)
    ax.set_ylabel(ylab, labelpad=8)
    ax.set_zlabel("|value|", labelpad=6)
    ax.set_zlim(0, zmax)
    ax.view_init(ELEV, AZIM)
    ax.set_title(title, fontsize=11)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    if boundaries:
        for k, b in enumerate(boundaries):
            ax.plot([b, b], [0, n - 1], [zmax, zmax], color="k",
                    lw=0.7, alpha=0.6)
        if blabels:
            step = boundaries[0]
            for k, lb in enumerate(blabels):
                ax.text(step * (k + 0.5), n * 1.02, zmax, lb,
                        fontsize=8, ha="center")


def pair_fig(VR, name, A, B, tA, tB, xlab, ylab, xf=1, boundaries=None,
             blabels=None):
    zmax = float(max(np.abs(A).max(), np.abs(B).max()))
    cmax = float(max(np.percentile(np.abs(A), 99.5),
                     np.percentile(np.abs(B), 99.5)))
    META[name] = {"zmax": zmax, "color_vmax_p99.5": cmax,
                  "shape_before": list(A.shape),
                  "shape_after": list(B.shape),
                  "color_norm": "PowerNorm(gamma=0.5, vmin=0, "
                                "vmax=pair_p99.5, clip) — z axis unclipped"}
    for mode in ("sameZ", "autoscale"):
        fig = plt.figure(figsize=(13, 5.2), facecolor="white")
        for j, (Z, tt) in enumerate(((A, tA), (B, tB))):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            zm = zmax if mode == "sameZ" else float(np.abs(Z).max())
            surf3d(ax, np.abs(Z), xlab, ylab, zm, tt, xf, boundaries,
                   blabels, cmax=cmax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            if mode == "autoscale" and ext == "pdf":
                continue
            fig.savefig(f"{VR}/plots/{name.replace('sameZ', mode)}.{ext}"
                        if "sameZ" in name else
                        f"{VR}/plots/{name}_{mode}.{ext}", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    VR = args.run_dir
    os.makedirs(f"{VR}/plots", exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white",
                         "axes.facecolor": "white"})

    # ---------------- PAIR A: concat H before/after R1_T
    Abef, _ = pooled_rows(VR, "Hconcat_before_R1")
    Aaft, _ = pooled_rows(VR, "Hconcat_after_R1")
    Pb, f = pool_cols(Abef, 1024)
    Pa, _ = pool_cols(Aaft, 1024)
    lb = [f"H{k+1}(L{SRC_IDS[k]})" for k in range(5)]
    bnd = [4096 * k for k in range(1, 5)]
    pair_fig(VR, "FIG_A_Hconcat_before_after_R1_sameZ", Pb, Pa,
             "Before Target R1_T", "After Target R1_T",
             "Concat channel", "Token / Sample", xf=f,
             boundaries=bnd, blabels=lb)

    # PAIR A 2D panels
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    eps = 1e-6
    for r, (X, tag) in enumerate(((Abef, "before R1_T"),
                                  (Aaft, "after R1_T"))):
        ab = np.abs(X)
        im = axes[r, 0].imshow(np.log10(ab + eps), aspect="auto",
                               cmap="magma")
        axes[r, 0].set_title(f"log10|x| heatmap ({tag})")
        plt.colorbar(im, ax=axes[r, 0], fraction=0.04)
        axes[r, 1].plot(ab.max(0), lw=0.4)
        axes[r, 1].set_title(f"per-channel absmax ({tag})")
        axes[r, 1].set_yscale("log")
        axes[r, 2].plot(np.sqrt((X ** 2).mean(0)), lw=0.4, color="tab:green")
        axes[r, 2].plot(np.percentile(ab, 99.9, axis=0), lw=0.4,
                        color="tab:red", alpha=0.6)
        axes[r, 2].set_title(f"per-channel RMS (green) / p99.9 (red) ({tag})")
        axes[r, 2].set_yscale("log")
        m = X.mean(0)
        v = X.var(0) + 1e-12
        kurt = (((X - m) ** 4).mean(0) / v ** 2)
        axes[r, 3].plot(kurt, lw=0.4, color="tab:purple")
        axes[r, 3].set_title(f"per-channel kurtosis ({tag})")
        axes[r, 3].set_yscale("log")
        for a in axes[r, 1:]:
            for b in bnd:
                a.axvline(b, color="k", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{VR}/plots/FIG_A2_Hconcat_channel_panels.png", dpi=300)
    fig.savefig(f"{VR}/plots/FIG_A2_Hconcat_channel_panels.pdf")
    plt.close(fig)
    # sorted absmax + cumulative energy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for X, tag in ((Abef, "before"), (Aaft, "after")):
        s = np.sort(np.abs(X).max(0))[::-1]
        axes[0].plot(s, lw=0.8, label=tag)
        e = np.sort((X ** 2).sum(0))[::-1]
        axes[1].plot(np.cumsum(e) / e.sum(), lw=0.8, label=tag)
    axes[0].set_yscale("log")
    axes[0].set_title("sorted channel absmax")
    axes[1].set_title("cumulative channel energy")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{VR}/plots/FIG_A3_sorted_energy.png", dpi=300)
    plt.close(fig)

    # ---------------- PAIR B: W_c stock vs folded (+W4)
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("HF_HOME", "/data/hf_cache_thahn1230")
    from dflash.model import DFlashDraftModel
    from . import interfaces as itf
    from . import spinquant_target as sqt
    from .rc import rtn_sym_perchannel
    d0 = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sqt.load_rbin(S1RBIN)["R1"]
    Wc = d0.fc.weight.data.float()
    WcF = itf.fold_wc(d0, R1).fc.weight.data.float()
    del d0
    W4s = rtn_sym_perchannel(Wc, 4)
    W4f = rtn_sym_perchannel(WcF, 4)
    np.savez_compressed(f"{VR}/raw_plot_samples/Wc_stock_sample.npz",
                        rows=Wc[::16, ::16].numpy().astype(np.float16))
    np.savez_compressed(f"{VR}/raw_plot_samples/Wc_R1fold_sample.npz",
                        rows=WcF[::16, ::16].numpy().astype(np.float16))

    def binrms(W, bo=64, bi=256):
        W = W.numpy()
        o, i = W.shape
        W = W[: (o // bo) * bo, : (i // bi) * bi]
        return np.sqrt((W.reshape(o // bo, bo, i // bi, bi) ** 2)
                       .mean((1, 3)))
    csvr = []
    for nm, W in (("stock", Wc), ("R1fold", WcF)):
        B = binrms(W)
        for r in range(B.shape[0]):
            for c in range(B.shape[1]):
                csvr.append({"view": nm, "out_bin": r, "in_bin": c,
                             "bin_rms": float(B[r, c])})
    import csv as _csv
    with open(f"{VR}/tables/wc_bin_rms.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(csvr[0].keys()))
        w.writeheader()
        w.writerows(csvr)
    bndb = [4096 // 256 * k for k in range(1, 5)]
    pair_fig(VR, "FIG_B_Wc_stock_vs_R1fold_sameZ",
             binrms(Wc), binrms(WcF), "W_c stock (bin RMS)",
             "W_c R1_T-folded (bin RMS)", "Input channel bin",
             "Output channel bin", boundaries=bndb, blabels=lb)
    pair_fig(VR, "FIG_B_Wc_stockW4_vs_foldW4_sameZ",
             binrms((W4s - Wc).abs()), binrms((W4f - WcF).abs()),
             "|W4 err| stock", "|W4 err| folded", "Input channel bin",
             "Output channel bin", boundaries=bndb, blabels=lb)
    pair_fig(VR, "FIG_B2_Wc_stride_raw_sameZ",
             Wc[::16, ::16].abs().numpy(), WcF[::16, ::16].abs().numpy(),
             "W_c stock (stride 16)", "W_c folded (stride 16)",
             "Input channel /16", "Output channel /16",
             boundaries=[256 * k for k in range(1, 5)], blabels=lb)
    del Wc, WcF, W4s, W4f

    # ---------------- PAIR C: H_t before/after R_C (+A4 twins)
    Cb, _ = pooled_rows(VR, "Ht_before_RC")
    Ca, _ = pooled_rows(VR, "Ht_after_RC")
    pair_fig(VR, "FIG_C_Ht_before_after_RC_sameZ", Cb, Ca,
             "H_t before R_C (deployed tap)", "H_t after R_C (Ht@R_C)",
             "Hidden channel", "Token / Sample")
    Db, _ = pooled_rows(VR, "Ht_before_RC_A4dq")
    Da, _ = pooled_rows(VR, "Ht_after_RC_A4dq")
    pair_fig(VR, "FIG_C2_Ht_A4_before_after_RC_sameZ", Db, Da,
             "dequant(A4(H_t)) before R_C", "dequant(A4(H_t R_C))",
             "Hidden channel", "Token / Sample")
    # worst-case diagnostic (explicitly labeled)
    wb = np.abs(Cb).max(1)
    order = np.argsort(-wb)[:64]
    pair_fig(VR, "WORST_CASE_DIAGNOSTIC_Ht_RC_sameZ", Cb[order],
             Ca[order], "H_t worst-|x| rows (before R_C)",
             "same rows after R_C", "Hidden channel",
             "Token (worst-case sorted)")

    # ---------------- PAIR D: caches per layer, RC off vs on
    for li in range(5):
        for kind, lab in (("kctx", "context_K"), ("vctx", "context_V"),
                          ("kdr", "draft_K"), ("vdr", "draft_V")):
            try:
                off, _ = pooled_rows(VR, f"L{li}_{kind}_cache_QOFF", 128)
                on, _ = pooled_rows(VR, f"L{li}_{kind}_cache_QON", 128)
            except FileNotFoundError:
                continue
            pair_fig(VR, f"FIG_D_L{li}_{lab}_cache_RCoff_on_sameZ",
                     off, on, f"L{li} {lab} cache, R_C OFF (W4A4)",
                     f"L{li} {lab} cache, R_C ON (W4A4)",
                     "KV channel (head*128+dim)", "Cache position")

    # ---------------- combined 4-stage summary
    fig = plt.figure(figsize=(12, 18), facecolor="white")
    rows = [
        (pool_cols(Abef, 512)[0], pool_cols(Aaft, 512)[0],
         "[H1..H5] BEFORE R1_T", "[H1..H5] AFTER R1_T"),
        None,  # filled below with W_c
        (Cb, Ca, "H_t BEFORE R_C", "H_t AFTER R_C"),
        None,  # filled below with cache
    ]
    d0 = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    Wc = d0.fc.weight.data.float()
    WcF = itf.fold_wc(d0, R1).fc.weight.data.float()
    del d0
    rows[1] = (binrms(Wc), binrms(WcF), "W_c STOCK", "W_c R1_T-FOLDED")
    off, _ = pooled_rows(VR, "L0_kctx_cache_QOFF", 128)
    on, _ = pooled_rows(VR, "L0_kctx_cache_QON", 128)
    rows[3] = (off, on, "ctx K cache (L0) R_C OFF", "R_C ON")
    for r, (A, B, tA, tB) in enumerate(rows):
        A, B = np.asarray(A), np.asarray(B)
        zmax = float(max(np.abs(A).max(), np.abs(B).max()))
        cmax = float(max(np.percentile(np.abs(A), 99.5),
                         np.percentile(np.abs(B), 99.5)))
        for j, (Z, tt) in enumerate(((A, tA), (B, tB))):
            ax = fig.add_subplot(4, 2, r * 2 + j + 1, projection="3d")
            surf3d(ax, np.abs(Z), "Channel",
                   "Token/Out-ch/Pos", zmax, tt, cmax=cmax)
    fig.tight_layout()
    fig.savefig(f"{VR}/plots/FIG_ROTATION_FOUR_STAGE_SUMMARY.png", dpi=300)
    fig.savefig(f"{VR}/plots/FIG_ROTATION_FOUR_STAGE_SUMMARY.pdf")
    plt.close(fig)

    json.dump(META, open(f"{VR}/plots/plot_metadata.json", "w"), indent=1)
    print(f"[plots] DONE {len(glob.glob(VR + '/plots/*.png'))} png, "
          f"zmax metadata for {len(META)} pairs")


if __name__ == "__main__":
    main()
