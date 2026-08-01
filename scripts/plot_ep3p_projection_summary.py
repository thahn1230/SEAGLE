#!/usr/bin/env python
"""Four publication-oriented EP3-P summary figures (spec §12).

All panels are rendered from the measured plot_data NPZs produced by
plot_ep3p_projection_3d.py — no schematic or synthetic values.
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

D = 4096
DPI = 300


def zload(rd, name):
    return np.load(os.path.join(rd, "plot_data", name + ".npz"))


def hm(ax, Z, title, norm=None, xlab="input channel", boundary=D,
       cbar_label=None, fig=None, center0=False, extent=None,
       ylab="token index"):
    if norm is None:
        if center0:
            M = float(np.abs(Z).max())
            norm = colors.Normalize(-M, M)
        else:
            norm = colors.Normalize(float(Z.min()), float(Z.max()))
    im = ax.imshow(Z, cmap="coolwarm", norm=norm, aspect="auto",
                   origin="lower",
                   extent=extent or [0, 2 * D, 0, Z.shape[0]])
    if boundary:
        ax.axvline(boundary, color="k", lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlab, fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cbar_label or "abs value", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    man = json.load(open(os.path.join(
        rd, "metadata", "collection_manifest.json")))
    summ = json.load(open(os.path.join(
        rd, "tables", "ep3p_viz_summary.json")))
    m_first, m_rec = man["m_first"], man["m_rec"]
    sdir = os.path.join(rd, "summaries")

    def path_summary(tag, m, fname):
        Xb = zload(rd, f"{tag}_projection_input_before_3d")["Z"]
        Xa = zload(rd, f"{tag}_projection_input_after_3d")["Z"]
        wtag = "first" if tag == "first" else "recurrent"
        Wb = zload(rd, "projection_weight_original_3d")["Z"]
        Wa = zload(rd, f"projection_weight_{wtag}_migrated_3d")["Z"]
        Q = zload(rd, f"{wtag}_quantized_projection_error_3d")
        Gn, Gep = Q["Z_left"], Q["Z_right"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        anorm = colors.Normalize(0, float(max(Xb.max(), Xa.max())))
        wnorm = colors.Normalize(0, float(max(Wb.max(), Wa.max())))
        qnorm = colors.Normalize(0, float(max(Gn.max(), Gep.max())))
        hm(axes[0, 0], Xb, "projection input BEFORE (abs)", anorm,
           fig=fig)
        hm(axes[0, 1], Xa,
           f"projection input AFTER (e x {m:.2f})", anorm, fig=fig)
        hm(axes[0, 2], Gn, "W4A4 output error, naive", qnorm,
           xlab="output channel", boundary=None, fig=fig,
           extent=[0, D, 0, Gn.shape[0]])
        hm(axes[1, 0], Wb, "projection weight BEFORE (abs)", wnorm,
           fig=fig)
        hm(axes[1, 1], Wa,
           f"projection weight AFTER (W_e / {m:.2f})", wnorm,
           fig=fig)
        hm(axes[1, 2], Gep, "W4A4 output error, EP3-P", qnorm,
           xlab="output channel", boundary=None, fig=fig,
           extent=[0, D, 0, Gep.shape[0]])
        qk = summ[f"quant_{wtag}"]
        fig.suptitle(
            f"EP3-P {tag} path (m = {m:.2f}): one complete "
            "projection, embedding|hidden boundary at channel "
            f"{D}; output NMSE naive "
            f"{qk['output_nmse_naive']:.4f} -> EP3-P "
            f"{qk['output_nmse_ep3p']:.4f}", fontsize=12)
        for ax in axes[:, :2].ravel():
            ax.set_ylabel("token", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(sdir, fname), dpi=DPI,
                    bbox_inches="tight")
        fig.savefig(os.path.join(sdir, fname.replace(".png", ".pdf")),
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[summary] {fname}")

    path_summary("first", m_first, "ep3p_first_path_summary.png")
    path_summary("recurrent_all", m_rec,
                 "ep3p_recurrent_path_summary.png")

    # C: first vs recurrent
    Xf = zload(rd, "first_projection_input_before_3d")["Z"]
    Xr = zload(rd, "recurrent_all_projection_input_before_3d")["Z"]
    Wf = zload(rd, "projection_weight_first_migrated_3d")["Z"]
    Wr = zload(rd, "projection_weight_recurrent_migrated_3d")["Z"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    hnorm = colors.Normalize(0, float(max(Xf.max(), Xr.max())))
    wnorm = colors.Normalize(0, float(max(Wf.max(), Wr.max())))
    hm(axes[0, 0], Xf, "first-path input distribution (abs)", hnorm,
       fig=fig)
    hm(axes[0, 1], Xr, "recurrent-path input distribution (abs)",
       hnorm, fig=fig)
    hm(axes[1, 0], Wf,
       f"first migrated weight (W_e / {m_first:.2f})", wnorm,
       fig=fig)
    hm(axes[1, 1], Wr,
       f"recurrent migrated weight (W_e / {m_rec:.2f})", wnorm,
       fig=fig)
    qf, qr = summ["quant_first"], summ["quant_recurrent"]
    fig.suptitle(
        f"EP3-P first vs recurrent: beta_first={man['beta_first']} "
        f"(m={m_first:.2f}), beta_recurrent={man['beta_rec']} "
        f"(m={m_rec:.2f}) | W4A4 output NMSE first "
        f"{qf['output_nmse_naive']:.4f}->"
        f"{qf['output_nmse_ep3p']:.4f}, recurrent "
        f"{qr['output_nmse_naive']:.4f}->"
        f"{qr['output_nmse_ep3p']:.4f}", fontsize=11)
    for ax in axes.ravel():
        ax.set_ylabel("token / output ch", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(
        sdir, "ep3p_first_vs_recurrent_summary.png"), dpi=DPI,
        bbox_inches="tight")
    plt.close(fig)
    print("[summary] ep3p_first_vs_recurrent_summary.png")

    # D: mechanism
    Xb = zload(rd, "first_projection_input_before_3d")["Z"]
    Xa = zload(rd, "first_projection_input_after_3d")["Z"]
    Wb = zload(rd, "projection_weight_original_3d")["Z"]
    Wa = Wf
    Ediff = zload(rd, "first_projection_input_difference_3d")["Z"]
    Yerr = zload(rd, "first_projection_output_error_3d")["Z"]
    Q = zload(rd, "first_quantized_projection_error_3d")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    anorm = colors.Normalize(0, float(max(Xb.max(), Xa.max())))
    wnorm2 = colors.Normalize(0, float(max(Wb.max(), Wa.max())))
    hm(axes[0, 0], Xb, "1. input before: embedding region tiny",
       anorm, fig=fig, cbar_label="abs activation")
    hm(axes[0, 1], Xa,
       f"2. input after: embedding region x m_first={m_first:.1f}",
       anorm, fig=fig, cbar_label="abs activation")
    hm(axes[0, 2], Ediff,
       "3. input difference: hidden region exactly 0", fig=fig,
       center0=True, cbar_label="abs(after)-abs(before)")
    hm(axes[1, 0], Wa - Wb,
       f"4. weight difference: embedding side / m_first="
       f"{m_first:.1f}, hidden side 0", fig=fig, center0=True,
       cbar_label="abs(after)-abs(before)",
       ylab="output-channel block")
    fp_max = summ["fp_first"]["max_abs_error"]
    hm(axes[1, 1], Yerr,
       f"5. FP output error: max {fp_max:.1e} (function preserved)",
       xlab="output channel", boundary=None, fig=fig,
       extent=[0, D, 0, Yerr.shape[0]], cbar_label="abs FP error")
    qn = colors.Normalize(0, float(max(Q["Z_left"].max(),
                                       Q["Z_right"].max())))
    hm(axes[1, 2], Q["Z_left"] - Q["Z_right"],
       "6. W4A4 error reduction: naive - EP3-P (red = EP3-P "
       "better)", xlab="output channel", boundary=None, fig=fig,
       center0=True, extent=[0, D, 0, Q["Z_left"].shape[0]],
       cbar_label="error(naive)-error(EP3-P)")
    fig.suptitle(
        "EP3-P mechanism (all panels measured): migrate embedding "
        "scale into the activation, divide it out of the weight — "
        "FP output unchanged, W4A4 reconstruction error drops "
        f"({summ['quant_first']['output_nmse_naive']:.4f} -> "
        f"{summ['quant_first']['output_nmse_ep3p']:.4f})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(sdir, "ep3p_mechanism_summary.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[summary] ep3p_mechanism_summary.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
