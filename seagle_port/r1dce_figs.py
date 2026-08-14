"""R1DCE §7/§30 figures.

3D sameZ pair panels follow the validated fourstage_plots conventions
(ELEV/AZIM 28/-60, |value| z-axis, coolwarm + PowerNorm(0.5) with cmax =
pair-level p99.5 so outliers saturate red, column absmax-pooling to 1024
bins, 256 evenly-spaced token rows shared across ALL panels).  Bar charts:
one hue for magnitude, R1_D highlighted, direct value labels, recessive
grid.  Reads captures/ht_sample.npz + tables/*.csv (+ shards for the tau
figure when present); skips gracefully if inputs are missing.
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

ELEV, AZIM = 28, -60
NROWS = 256
CBLUE, CORANGE, CGRAY = "#3b6fb6", "#e07b39", "#9aa5b1"


def pool_cols(A, nbin=1024):
    n, d = A.shape
    k = d // nbin
    return np.abs(A).reshape(n, nbin, k).max(axis=2)


def surf3d(ax, Z, title, zmax, cmax):
    n, d = Z.shape
    X, Y = np.meshgrid(np.arange(d), np.arange(n))
    ax.plot_surface(X, Y, Z, cmap="coolwarm",
                    norm=PowerNorm(0.5, vmin=0, vmax=cmax, clip=True),
                    rcount=min(n, 128), ccount=min(d, 1024),
                    linewidth=0, antialiased=False)
    ax.set_zlim(0, zmax)
    ax.view_init(ELEV, AZIM)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("channel (absmax-pooled)", fontsize=7, labelpad=-4)
    ax.set_ylabel("token", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.05)


def pair_fig(path, panels, suptitle):
    """panels = [(title, Z[n,1024]), ...] — shared zlim + cmax (sameZ)."""
    zmax = max(Z.max() for _, Z in panels)
    cmax = np.percentile(np.concatenate([Z.flatten() for _, Z in panels]),
                         99.5)
    w = 6.5 * len(panels)
    fig = plt.figure(figsize=(w, 5.2), facecolor="white")
    for i, (t, Z) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        surf3d(ax, Z, t, zmax, cmax)
    fig.suptitle(suptitle + f"   (sameZ zmax={zmax:.2f}, "
                 f"color sat p99.5={cmax:.2f})", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {path}.png/.pdf")


def bar_fig(path, names, vals, title, ylab, logy=False, highlight="R1_D",
            baseline=None):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="white")
    cols = [CORANGE if n == highlight else CBLUE for n in names]
    ax.bar(names, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4g}", (i, v), ha="center", va="bottom",
                    fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, color=CGRAY, lw=1, ls="--")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#e5e8ec", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {path}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    fig_d = f"{rd}/figs"
    os.makedirs(fig_d, exist_ok=True)
    import torch
    from .r1dce_ht_stats import candidates

    z = np.load(f"{rd}/captures/ht_sample.npz")
    Ht = torch.tensor(z["rows"], dtype=torch.float32)
    idx = np.linspace(0, Ht.shape[0] - 1, NROWS).astype(int)
    cands = candidates(rd, "cpu")
    X = {}
    for name, R in cands.items():
        Xr = Ht if R is None else Ht @ R.cpu()
        X[name] = pool_cols(Xr[idx].numpy())

    pair_fig(f"{fig_d}/FIG_Ht_noRot_vs_R1D_sameZ",
             [("H_t (no context rotation)", X["none"]),
              ("H_t @ R1_D (draft-R1 extension)", X["R1_D"])],
             "H_t before ctx A4: no rotation vs draft-R1 extension")
    pair_fig(f"{fig_d}/FIG_Ht_R1D_vs_R1T_sameZ",
             [("H_t @ R1_D", X["R1_D"]), ("H_t @ R1_T", X["R1_T_currentRC"])],
             "H_t @ R1_D vs H_t @ R1_T")
    pair_fig(f"{fig_d}/FIG_Ht_R1D_vs_currentRC_sameZ",
             [("H_t @ R1_D", X["R1_D"]),
              ("H_t @ current R_C (== R1_T, Phase-0)", X["R1_T_currentRC"])],
             "H_t @ R1_D vs current deployed R_C")
    pair_fig(f"{fig_d}/FIG_Ht_context_rotation_candidates_sameZ",
             [("no rotation", X["none"]), ("R1_D", X["R1_D"]),
              ("R1_T == current R_C", X["R1_T_currentRC"]),
              ("Hadamard", X["Hadamard"]),
              ("random (s101)", X["Random_s101"])],
             "H_t context-rotation candidates (same tokens)")

    # channel absmax small-multiples (shared log ylim)
    names = list(X.keys())
    nr = (len(names) + 4) // 5
    fig, axes = plt.subplots(nr, 5, figsize=(18, 3 * nr), sharey=True,
                             facecolor="white", squeeze=False)
    ymax = max(x.max() for x in X.values()) * 1.3
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        prof = X[name].max(axis=0)
        ax.plot(prof, lw=0.5,
                color=CORANGE if name == "R1_D" else CBLUE)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, ymax)
        ax.set_title(name, fontsize=9)
        ax.grid(color="#e5e8ec", lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("H_t channel absmax by candidate (1024 pooled bins, "
                 "same tokens, log scale)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{fig_d}/FIG_Ht_all_candidates_channel_absmax.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote FIG_Ht_all_candidates_channel_absmax")

    # bar charts from tables
    def read_csv(p):
        return list(csv.DictReader(open(p)))
    hs = read_csv(f"{rd}/tables/ht_candidate_stats.csv")
    bar_fig(f"{fig_d}/FIG_Ht_all_candidates_kurtosis",
            [r["candidate"] for r in hs],
            [float(r["kurtosis"]) for r in hs],
            "H_t kurtosis by context-rotation candidate (Gaussian=3)",
            "kurtosis (log)", logy=True, baseline=3.0)
    ha = read_csv(f"{rd}/tables/ht_a4_candidate_stats.csv")
    for p in ("FIG_Ht_candidate_A4_NMSE", "FIG_Ht_all_candidates_A4_NMSE"):
        bar_fig(f"{fig_d}/{p}",
                [r["candidate"] for r in ha],
                [float(r["a4_nmse_mean"]) for r in ha],
                "H_t A4 NMSE (mean per-token) by candidate",
                "A4 NMSE (log)", logy=True)
    kv = read_csv(f"{rd}/tables/ctx_kv_aw_decomposition.csv")
    for proj in ("K", "V"):
        agg, order = {}, []
        for r in kv:
            if r["proj"] == proj:
                agg.setdefault(r["candidate"], []).append(
                    float(r["AW_nmse"]))
                if r["candidate"] not in order:
                    order.append(r["candidate"])
        bar_fig(f"{fig_d}/FIG_ctx{proj}_candidate_NMSE",
                order, [float(np.mean(agg[n])) for n in order],
                f"ctx {proj} projection W4A4 NMSE (mean over 5 layers)",
                "A+W NMSE (log)", logy=True)

    # validation tau (only when shards exist)
    shards = sorted(glob.glob(f"{rd}/shards/al__*__w4a4__gsm8kvalid.csv"))
    if shards:
        tags, taus = [], []
        for s in shards:
            tag = os.path.basename(s).split("__")[1]
            tot = cyc = 0
            for row in csv.DictReader(open(s)):
                ts = [int(x) for x in row["taus"].split(";") if x]
                tot += sum(ts)
                cyc += len(ts)
            if cyc:
                tags.append(tag)
                taus.append(tot / cyc)
        if tags:
            bar_fig(f"{fig_d}/FIG_validation_tau_by_context_rotation",
                    tags, taus,
                    "Validation AL (gsm8kvalid n60, cycle-pooled tau)",
                    "tau", highlight="C1_r1d")


if __name__ == "__main__":
    main()
