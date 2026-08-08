"""DKVA §7-11, 14-15, 18, 22, 25-26: all figures.

Reads reservoir NPZ + channel arrays + qparam NPZ + weight arrays and
produces the headline figures (FIG-1..8), DFLASH_WC_WEIGHT_3D,
DFLASH_CTX_VS_DRAFT_ACT_3D_layer{0..4}, plus histograms/ECDF/channel plots.

Conventions (§22): deterministic seed-0 sampling of 256 token rows;
before/after pairs share identical camera + axis limits (primary), with
`_autoscale` variants; PNG(300dpi)+PDF; perspective 3D + top-down heatmap.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

SEED = 0
NROWS = 256
SRC = [1, 8, 15, 22, 29]
DPI = 300


def load_rows(rd, cfg, tensor, n=NROWS, ds_order=("mtbench", "gsm8k",
                                                  "sharegpt", "humaneval")):
    rows = []
    for ds in ds_order:
        p = f"{rd}/raw/activations/{cfg}__{ds}__{tensor}.npz"
        if os.path.exists(p):
            rows.append(np.load(p)["rows"])
    if not rows:
        return None
    x = np.concatenate(rows).astype(np.float32)
    rng = np.random.default_rng(SEED)
    if len(x) > n:
        x = x[rng.choice(len(x), n, replace=False)]
    return x


def surf3d(ax, A, cmap="viridis", zlim=None, stride=1):
    T, D = A.shape
    xs = np.arange(0, D, stride)
    X, Y = np.meshgrid(xs, np.arange(T))
    Z = np.abs(A[:, ::stride])
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False,
                    rcount=min(T, 128), ccount=min(len(xs), 512))
    if zlim:
        ax.set_zlim(0, zlim)
    ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("channel", fontsize=9)
    ax.set_ylabel("token", fontsize=9)
    ax.set_zlabel("|activation|", fontsize=9)


def save(fig, rd, sub, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{rd}/plots/{sub}/{name}.{ext}", dpi=DPI,
                    bbox_inches="tight")
    plt.close(fig)


def pair3d(rd, sub, name, A, B, la, lb, boundaries=None, stride=8):
    zmax = max(np.abs(A).max(), np.abs(B).max())
    for mode, zl in (("", zmax), ("_autoscale", None)):
        fig = plt.figure(figsize=(13, 5))
        for i, (M, lab) in enumerate(((A, la), (B, lb))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            surf3d(ax, M, zlim=zl if mode == "" else None, stride=stride)
            ax.set_title(lab, fontsize=10)
            if boundaries:
                for b in boundaries[1:-1]:
                    ax.plot([b, b], [0, M.shape[0]], [0, 0], "r-", lw=0.5)
        save(fig, rd, sub, f"{name}{mode}")
    # top-down heatmaps (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    vmax = np.log10(zmax + 1e-6)
    for ax, M, lab in ((axes[0], A, la), (axes[1], B, lb)):
        im = ax.imshow(np.log10(np.abs(M) + 1e-6), aspect="auto",
                       cmap="magma", vmin=-4, vmax=vmax)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("channel"); ax.set_ylabel("token")
        if boundaries:
            for b in boundaries[1:-1]:
                ax.axvline(b, color="w", lw=0.5)
    fig.colorbar(im, ax=axes, label="log10|x|")
    save(fig, rd, "heatmaps", f"{name}_heatmap")


def single3d(rd, sub, name, A, label, boundaries=None, stride=8):
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf3d(ax, A, stride=stride)
    ax.set_title(label, fontsize=10)
    if boundaries:
        for b in boundaries[1:-1]:
            ax.plot([b, b], [0, A.shape[0]], [0, 0], "r-", lw=0.5)
    save(fig, rd, sub, name)


def hist_suite(rd, tensors, name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for lab, x in tensors:
        a = np.abs(x.flatten())
        axes[0, 0].hist(a, bins=200, alpha=0.5, label=lab)
        axes[0, 1].hist(a, bins=200, alpha=0.5, label=lab, log=True)
        axes[0, 2].hist(np.log10(a + 1e-8), bins=200, alpha=0.5, label=lab)
        s = np.sort(a)
        axes[1, 0].plot(s, np.linspace(0, 1, len(s)), label=lab)
        axes[1, 1].loglog(s, 1 - np.linspace(0, 1, len(s), endpoint=False),
                          label=lab)
        axes[1, 2].hist(np.abs(x).max(1), bins=60, alpha=0.5, label=lab)
    titles = ["|x| hist", "|x| hist (log-y)", "log10|x| hist", "ECDF |x|",
              "CCDF |x| (log-log)", "per-token absmax"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
    save(fig, rd, "histograms", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    for sub in ("activations_3d", "weights_3d", "heatmaps", "histograms",
                "ecdf", "channels", "qparams", "projection_error",
                "attention"):
        os.makedirs(f"{rd}/plots/{sub}", exist_ok=True)

    # ---------- FIG-1/2: concat before/after R_T ----------
    cat_before = load_rows(rd, "C0", "B_concat")
    cat_after = load_rows(rd, "C2", "B_concat")
    bnd = [i * 4096 for i in range(6)]
    if cat_before is not None and cat_after is not None:
        pair3d(rd, "activations_3d", "FIG1_2_concat_before_after_RT",
               cat_before, cat_after,
               "5-source concat BEFORE R_T (C0)",
               "5-source concat AFTER R_T (C2)", boundaries=bnd, stride=32)
        single3d(rd, "activations_3d", "concat_before_RT_3d", cat_before,
                 "concat H_1|H_8|H_15|H_22|H_29 before R_T",
                 boundaries=bnd, stride=32)
        single3d(rd, "activations_3d", "concat_after_RT_3d", cat_after,
                 "concat after R_T", boundaries=bnd, stride=32)

    # per-source 3D (A tensors)
    for l in SRC:
        a0 = load_rows(rd, "C0", f"A_src{l}")
        a2 = load_rows(rd, "C2", f"A_src{l}")
        if a0 is not None and a2 is not None:
            pair3d(rd, "activations_3d", f"A_src{l}_before_after_RT",
                   a0, a2, f"H_{l} before R_T", f"H_{l} after R_T")

    # ---------- FIG-3/4: H_t before/after R_C ----------
    Ht = load_rows(rd, "C2", "C_Ht")
    Htr = load_rows(rd, "C3", "C_Ht_rot")
    Ht_q = load_rows(rd, "C2", "C_Ht_q")
    if Ht is not None and Htr is not None:
        pair3d(rd, "activations_3d", "FIG3_4_Ht_before_after_RC",
               Ht, Htr, "H_t BEFORE R_C", "H_t AFTER R_C (= R_T reuse)")
    Zt = load_rows(rd, "C2", "C_Zt")
    if Zt is not None:
        single3d(rd, "activations_3d", "Zt_before_rmsnorm", Zt,
                 "Z_t = W_c output before RMSNorm")

    # A4-dequant pair (recompute deployed quantizer on the same rows)
    from .dkva_capture import act_quant_detail
    import torch
    if Ht is not None and Htr is not None:
        dq_b = act_quant_detail(torch.from_numpy(Ht))["dequant"].numpy()
        dq_a = act_quant_detail(torch.from_numpy(Htr))["dequant"].numpy()
        pair3d(rd, "activations_3d", "FIGC_D_Ht_A4dequant_before_after",
               dq_b, dq_a, "Q_A4(H_t) dequant", "Q_A4(H_t R_C) dequant")

    # ---------- FIG-5: per-channel comparison ----------
    ch2 = np.load(glob.glob(f"{rd}/raw/activations/C2__*__channels.npz")[0])
    ch3 = np.load(glob.glob(f"{rd}/raw/activations/C3__*__channels.npz")[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, met in zip(axes, ("rms", "absmax", "absmax")):
        a = ch2[f"C_Ht__{met}"]
        b = ch3[f"C_Ht_rot__{met}"]
        if met == "absmax" and ax is axes[2]:
            ax.semilogy(np.sort(a)[::-1], label="H_t")
            ax.semilogy(np.sort(b)[::-1], label="H_t R_C")
            ax.set_title("sorted channel absmax")
        else:
            ax.plot(a, lw=0.4, label="H_t")
            ax.plot(b, lw=0.4, alpha=0.7, label="H_t R_C")
            ax.set_title(f"per-channel {met}")
        ax.set_xlabel("channel"); ax.legend(fontsize=8)
    save(fig, rd, "channels", "FIG5_Ht_channel_before_after_RC")

    # ---------- FIG-6: qparams ----------
    q2 = {os.path.basename(p).split("__", 2)[2][:-4]: np.load(p)
          for p in glob.glob(f"{rd}/raw/qparams/C2__mtbench__C_Ht.npz")}
    q3 = {os.path.basename(p).split("__", 2)[2][:-4]: np.load(p)
          for p in glob.glob(f"{rd}/raw/qparams/C3__mtbench__C_Ht_rot.npz")}
    if q2 and q3:
        a, b = q2["C_Ht"], q3["C_Ht_rot"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(a["scale"], bins=80, alpha=0.6, label="H_t")
        axes[0].hist(b["scale"], bins=80, alpha=0.6, label="H_t R_C")
        axes[0].set_title("A4 per-token scale")
        axes[1].hist(a["zero"], bins=16, alpha=0.6, label="H_t")
        axes[1].hist(b["zero"], bins=16, alpha=0.6, label="H_t R_C")
        axes[1].set_title("A4 per-token zero-point")
        axes[2].hist(a["tok_nmse"], bins=80, alpha=0.6, label="H_t",
                     log=True)
        axes[2].hist(b["tok_nmse"], bins=80, alpha=0.6, label="H_t R_C",
                     log=True)
        axes[2].set_title("per-token A4 NMSE (log-y)")
        for ax in axes:
            ax.legend(fontsize=8)
        save(fig, rd, "qparams", "FIG6_Ht_qparams_before_after_RC")

    # ---------- FIG-7: K/V projection NMSE ----------
    aw = f"{rd}/tables/kv_projection_error.csv"
    if os.path.exists(aw):
        import csv as _csv
        rows = list(_csv.DictReader(open(aw)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for pi, pname in enumerate(("K", "V")):
            ax = axes[pi]
            for br, lab in (("ctx_RCoff", "ctx R_C OFF"),
                            ("ctx_RC1", "ctx R_C=R_T"),
                            ("draft", "draft-side")):
                ys = [float(r["AW_nmse"]) for r in rows
                      if r["proj"] == pname and r["branch"] == br]
                if ys:
                    ax.plot(range(len(ys)), ys, marker="o", label=lab)
            ax.set_yscale("log")
            ax.set_xlabel("draft layer"); ax.set_ylabel("A+W NMSE")
            ax.set_title(f"{pname}_proj output error")
            ax.legend(fontsize=8)
        save(fig, rd, "projection_error", "FIG7_kv_projection_nmse")

    # ---------- FIG-8 / §26: W_c weight 3D ----------
    wz = np.load(f"{rd}/raw/weights/weights_arrays.npz")
    def wgrid(name):
        return wz[f"{name}__grid"].astype(np.float32)
    for pairname, na, nb, la, lb in (
        ("DFLASH_WC_WEIGHT_3D_stock_vs_folded", "Wc_stock", "Wc_foldedRT",
         "W_c stock |W|", "W_c R_T-folded |W|"),
        ("DFLASH_WC_WEIGHT_3D_w4", "Wc_stock", "Wc_foldedRT",
         "W_c stock W4-dequant", "W_c folded W4-dequant")):
        A = wgrid(na) if "w4" not in pairname else \
            wz[f"{na}__grid_q"].astype(np.float32)
        B = wgrid(nb) if "w4" not in pairname else \
            wz[f"{nb}__grid_q"].astype(np.float32)
        zmax = max(A.max(), B.max())
        fig = plt.figure(figsize=(14, 5.5))
        for i, (M, lab) in enumerate(((A, la), (B, lb))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            T, D = M.shape
            X, Y = np.meshgrid(np.arange(D), np.arange(T))
            ax.plot_surface(X, Y, M, cmap="viridis", linewidth=0,
                            rcount=128, ccount=512)
            ax.set_zlim(0, zmax)
            ax.view_init(elev=30, azim=-55)
            ax.set_title(lab + "\n(H1|H8|H15|H22|H29 blocks, downsampled)",
                         fontsize=9)
            ax.set_xlabel("input ch (ds)"); ax.set_ylabel("output ch (ds)")
        save(fig, rd, "weights_3d", pairname)
    # K/V ctx views
    for li in range(5):
        for nm in ("k_proj", "v_proj"):
            A = wz[f"l{li}_{nm}_ctxview_RCoff__grid"].astype(np.float32)
            B = wz[f"l{li}_{nm}_ctxview_RC1__grid"].astype(np.float32)
            zmax = max(A.max(), B.max())
            fig = plt.figure(figsize=(12, 5))
            for i, (M, lab) in enumerate(((A, f"W_{nm} original"),
                                          (B, f"W_{nm} @ R_C view"))):
                ax = fig.add_subplot(1, 2, i + 1, projection="3d")
                T, D = M.shape
                X, Y = np.meshgrid(np.arange(D), np.arange(T))
                ax.plot_surface(X, Y, M, cmap="viridis", linewidth=0,
                                rcount=96, ccount=256)
                ax.set_zlim(0, zmax)
                ax.view_init(elev=30, azim=-55)
                ax.set_title(f"layer{li} {lab}", fontsize=9)
            save(fig, rd, "weights_3d", f"kv_view_l{li}_{nm}")

    # ---------- §26 ctx vs draft side-by-side ----------
    for li in range(5):
        Hd = load_rows(rd, "C2", f"D_draft_k_l{li}")
        if Ht is None or Hd is None:
            continue
        n = min(len(Ht), len(Hd), NROWS)
        M = np.concatenate([Ht[:n], Hd[:n]], axis=1)  # [n, 8192]
        fig = plt.figure(figsize=(10, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        surf3d(ax, M, stride=16)
        ax.plot([4096, 4096], [0, n], [0, 0], "r-", lw=1.0)
        ax.text(1500, n * 1.02, 0, "Target-context input", fontsize=9)
        ax.text(5200, n * 1.02, 0, "Draft-side input", fontsize=9)
        ax.set_title(f"layer {li}: ctx H_t (ch 0-4095) vs draft H_d "
                     f"(ch 4096-8191); independent per-token qparams",
                     fontsize=9)
        save(fig, rd, "activations_3d",
             f"DFLASH_CTX_VS_DRAFT_ACT_3D_layer{li}")

    # ---------- §10 histogram suites ----------
    suites = [("Ht_vs_HtRC", [("H_t", Ht), ("H_t R_C", Htr)])]
    for li in (0, 4):
        Hd = load_rows(rd, "C2", f"D_draft_k_l{li}")
        if Hd is not None:
            suites.append((f"Ht_vs_Hd_l{li}",
                           [("H_t", Ht), (f"H_d l{li}", Hd)]))
    for name, tensors in suites:
        tensors = [(l, x) for l, x in tensors if x is not None]
        if len(tensors) >= 2:
            hist_suite(rd, tensors, name)

    print("[dkva_plots] DONE")


if __name__ == "__main__":
    main()
