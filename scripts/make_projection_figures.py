#!/usr/bin/env python
"""Paper-quality 3D projection-layer figures (PNG+PDF, 300dpi) from the
extracted npz tensors. Blue->red (coolwarm), z=|value|, channel
boundary between embedding-side and hidden-side annotated.

Max-abs pooling is used to keep surfaces tractable while PRESERVING
outlier structure (documented in the metadata json)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

NR = sys.argv[1]
ART = os.path.join(NR, "artifacts")
OUT = os.path.join(NR, "plots")
os.makedirs(OUT, exist_ok=True)

METHODS = ["SEAGLE_PTQ", "SEAGLE_QAT", "SEAGLE_RT"]
FIGN = {"SEAGLE_PTQ": "FIG1", "SEAGLE_QAT": "FIG2", "SEAGLE_RT": "FIG3"}
D = 4096


def pool_max(x, ry, rx):
    """max-|.| pooling to (ceil(H/ry), ceil(W/rx))."""
    H, W = x.shape
    ph = (-H) % ry
    pw = (-W) % rx
    a = np.abs(x)
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw)))
    a = a.reshape((H + ph) // ry, ry, (W + pw) // rx, rx)
    return a.max(axis=(1, 3))


def surface(ax, z, ylabel, boundary_col, zmax=None):
    H, W = z.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    norm = plt.Normalize(0, zmax if zmax else z.max())
    ax.plot_surface(X, Y, z, facecolors=cm.coolwarm(norm(z)),
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, rasterized=True)
    if zmax:
        ax.set_zlim(0, zmax)
    ax.set_xlabel("channel (pooled)", fontsize=8, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=6)
    ax.set_zlabel("|value|", fontsize=8, labelpad=4)
    ax.tick_params(labelsize=7)
    # embedding|hidden boundary wall
    zm = zmax if zmax else z.max()
    ys = np.array([[0, 0], [H - 1, H - 1]])
    xs = np.full_like(ys, boundary_col, dtype=float)
    zs = np.array([[0, zm], [0, zm]], dtype=float)
    ax.plot_surface(xs, ys, zs, color="k", alpha=0.18, shade=False)
    ax.view_init(elev=28, azim=-60)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


meta = {"pooling": {}, "zscales": {}}
data = {m: np.load(os.path.join(ART, f"{m}_projection.npz"))
        for m in METHODS}

count = 0
for m in METHODS:
    d = data[m]
    for path in ("first", "recurrent"):
        for kind, ylab, ry, rx in (
                ("act", "token index", 1, 16),
                ("w", "output channel (pooled)", 32, 16)):
            key_o = (f"act_{path}_original" if kind == "act"
                     else f"w_{'first' if path=='first' else 'rec'}_original")
            key_a = key_o.replace("original", "after")
            zo = pool_max(d[key_o], ry, rx)
            za = pool_max(d[key_a], ry, rx)
            bcol = D // rx
            zmax = max(zo.max(), za.max())
            meta["pooling"][f"{m}_{path}_{kind}"] = dict(
                ry=ry, rx=rx, mode="max-abs")
            meta["zscales"][f"{m}_{path}_{kind}"] = float(zmax)
            for tag, z in (("original", zo), ("after", za)):
                for sc, zm in (("samez", zmax), ("autoscale", None)):
                    fig = plt.figure(figsize=(7, 5))
                    ax = fig.add_subplot(111, projection="3d")
                    surface(ax, z, ylab, bcol, zm)
                    kd = "activation" if kind == "act" else "weight"
                    ax.set_title(
                        f"{m.replace('_', '-')} · {path} projection · "
                        f"{kd} ({tag})\n"
                        "black wall = embedding|hidden boundary "
                        f"(ch {D})", fontsize=9)
                    save(fig, f"{FIGN[m]}_{m}_{path}_{kd}_{tag}_{sc}")
                    count += 2

# FIG4/FIG5: side-by-side comparisons (first path)
for fign, kind, keyf, ry, rx, ylab in (
        ("FIG4", "activation", "act_first_original", 1, 16,
         "token index"),
        ("FIG5", "weight", "w_first_original", 32, 16,
         "output channel (pooled)")):
    zs = {m: pool_max(data[m][keyf], ry, rx) for m in METHODS}
    zmax = max(z.max() for z in zs.values())
    fig = plt.figure(figsize=(16, 5))
    for i, m in enumerate(METHODS):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        surface(ax, zs[m], ylab, D // rx, zmax)
        ax.set_title(m.replace("_", "-"), fontsize=10)
    fig.suptitle(f"First projection {kind} (original basis), "
                 "same z-scale", fontsize=11)
    save(fig, f"{fign}_all_methods_first_{kind}_original_samez")
    count += 2

# FIG6: summary stat bars
stats = {}
for m in METHODS:
    j = json.load(open(os.path.join(ART, f"{m}_manifest.json")))
    stats[m] = j["stats"]
keys = ["act_first_original", "act_first_after",
        "w_first_original", "w_first_after"]
metrics = ["rms", "absmax", "kurtosis", "top01pct_channel_energy"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
xs = np.arange(len(keys))
wd = 0.25
for ai, met in enumerate(metrics):
    ax = axes[ai]
    for mi, m in enumerate(METHODS):
        ax.bar(xs + mi * wd, [stats[m][k][met] for k in keys], wd,
               label=m.replace("_", "-"))
    ax.set_xticks(xs + wd)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys],
                       fontsize=7)
    ax.set_title(met, fontsize=10)
    if met in ("absmax", "kurtosis"):
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("First-projection tensor statistics by method",
             fontsize=11)
save(fig, "FIG6_projection_stats_by_method")
count += 2

json.dump(meta, open(os.path.join(NR, "artifacts",
                                  "plot_metadata.json"), "w"),
          indent=1)
print(f"[fig] wrote {count} files -> {OUT}")
