#!/usr/bin/env python
"""EP3-P projection visualization: 3D surfaces + heatmaps + log
companions + statistics tables (study spec §5-§11, §13).

Everything is plotted from the measured tensors in
plot_data/ep3p_tensors.pt (see collect_ep3p_projection_tensors.py).
The projection is always shown as ONE complete input [N, 2D] or ONE
complete weight matrix [out, 2D] with the embedding/hidden boundary
drawn at input channel D=4096.

Grouped mode (default, recorded in visualization_manifest.json):
  activations: channel groups of 16 (mean abs; --act-agg maxabs),
               every collected token retained
  weights:     16 x 16 input x output blocks (mean abs; --w-agg
               rmsabs/maxabs)
Full-resolution statistics are computed on UNgrouped tensors and land
in tables/. Exact plotted matrices land in plot_data/<figure>.npz with
<figure>_metadata.json.
"""
import argparse, json, math, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

D = 4096
DPI = 300
EPS = 1e-6
CAM = dict(elev=27, azim=-58)


# ---------------------------------------------------------------- util
def group_rows(A, g, agg):
    """A: [N, C] abs-values -> [N, C//g]."""
    N, C = A.shape
    v = A.reshape(N, C // g, g)
    if agg == "meanabs":
        return v.mean(-1)
    if agg == "maxabs":
        return v.max(-1)
    if agg == "rmsabs":
        return np.sqrt((v ** 2).mean(-1))
    raise ValueError(agg)


def group2d(W, gi, go, agg):
    """W: [O, I] abs-values -> [O//go, I//gi]."""
    O, I = W.shape
    v = W.reshape(O // go, go, I // gi, gi).transpose(0, 2, 1, 3) \
        .reshape(O // go, I // gi, -1)
    if agg == "meanabs":
        return v.mean(-1)
    if agg == "maxabs":
        return v.max(-1)
    if agg == "rmsabs":
        return np.sqrt((v ** 2).mean(-1))
    raise ValueError(agg)


class Fig:
    """Shared saver: PNG(300dpi)+PDF, NPZ + metadata JSON."""

    def __init__(self, rd, meta_common):
        self.rd = rd
        self.mc = meta_common
        self.n3d = 0
        self.nheat = 0

    def _save(self, fig, sub, name, pdf=True):
        p = os.path.join(self.rd, sub, name)
        if os.path.exists(p + ".png") and os.environ.get(
                "EP3P_SKIP_EXISTING") == "1":
            plt.close(fig)
            return
        fig.savefig(p + ".png", dpi=DPI, bbox_inches="tight")
        if pdf:
            try:
                fig.savefig(p + ".pdf", bbox_inches="tight")
            except Exception:
                pass
        plt.close(fig)

    def _data(self, name, arrays, meta):
        pd = os.path.join(self.rd, "plot_data")
        np.savez_compressed(os.path.join(pd, name + ".npz"), **arrays)
        json.dump({**self.mc, **meta}, open(os.path.join(
            pd, name + "_metadata.json"), "w"), indent=1)

    def surf(self, sub, name, Z, xlab, ylab, zlab, xvals=None,
             boundary=None, norm=None, title=None, ylines=None,
             region_labels=("Embedding input region",
                            "Hidden-feature input region"),
             meta=None, log_companion=False, cmap="coolwarm"):
        """Z: [ny, nx]; xvals: true channel coordinate per column."""
        Z = np.asarray(Z, dtype=float)
        xs = np.asarray(xvals if xvals is not None
                        else np.arange(Z.shape[1]), dtype=float)
        ys = np.arange(Z.shape[0], dtype=float)
        X, Y = np.meshgrid(xs, ys)
        if norm is None:
            norm = colors.Normalize(vmin=float(Z.min()),
                                    vmax=float(Z.max()))
        # rendering stride only (full matrix goes to NPZ unreduced)
        rs = max(1, Z.shape[0] // 160)
        cs = max(1, Z.shape[1] // 320)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        s = ax.plot_surface(X, Y, Z, facecolors=cm.get_cmap(cmap)(
            norm(Z)), rstride=rs, cstride=cs, linewidth=0,
            antialiased=False, shade=False)
        s.set_rasterized(True)
        if boundary is not None:
            zl = ax.get_zlim()
            bx = np.full((2, 2), float(boundary))
            by = np.array([[ys[0], ys[-1]], [ys[0], ys[-1]]])
            bz = np.array([[zl[0], zl[0]], [zl[1], zl[1]]])
            ax.plot_surface(bx, by, bz, color="k", alpha=0.18)
            ax.text(boundary * 0.45, ys[-1], zl[1],
                    region_labels[0], fontsize=9)
            ax.text(boundary * 1.45, ys[-1], zl[1],
                    region_labels[1], fontsize=9)
        m = cm.ScalarMappable(cmap=cm.get_cmap(cmap), norm=norm)
        m.set_array(Z)
        fig.colorbar(m, ax=ax, shrink=0.55, pad=0.09)
        ax.set_xlabel(xlab, fontsize=9, labelpad=14)
        ax.set_ylabel(ylab, fontsize=9, labelpad=10)
        ax.set_zlabel(zlab, fontsize=9, labelpad=10)
        ax.set_title(title or name, fontsize=11)
        ax.view_init(**CAM)
        if ylines:
            for yv, lab in ylines:
                ax.plot(xs, np.full_like(xs, yv),
                        np.full_like(xs, ax.get_zlim()[0]), "k--",
                        lw=0.8)
        self._save(fig, sub, name, pdf=False)
        self.n3d += 1
        md = dict(figure=name, scale="linear",
                  norm=[float(norm.vmin), float(norm.vmax)],
                  render_row_stride=rs, render_col_stride=cs,
                  render_note="stride affects rendering only; NPZ "
                              "matrix is unreduced")
        md.update(meta or {})
        self._data(name, dict(Z=Z, x=xs, y=ys), md)
        if log_companion:
            Zl = np.log10(np.abs(Z) + EPS)
            self.surf("log_scale", name + "_log10", Zl, xlab, ylab,
                      f"log10(abs + {EPS:g}) [LOG SCALE COMPANION]",
                      xvals=xs, boundary=boundary,
                      title=(title or name) + " — log10 companion",
                      ylines=ylines, region_labels=region_labels,
                      meta={**(meta or {}), "scale": "log10"},
                      log_companion=False)
            self.n3d -= 1   # companions not counted as primary 3D
        print(f"[fig] {sub}/{name}")

    def pair(self, sub, name, Za, Zb, la, lb, xlab, ylab, zlab,
             xvals=None, boundary=None, title=None, meta=None):
        """Two surfaces side by side, ONE shared normalization."""
        Za, Zb = np.asarray(Za, float), np.asarray(Zb, float)
        vmax = float(max(Za.max(), Zb.max()))
        vmin = float(min(Za.min(), Zb.min()))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        xs = np.asarray(xvals if xvals is not None
                        else np.arange(Za.shape[1]), dtype=float)
        fig = plt.figure(figsize=(18, 8))
        for i, (Z, lab) in enumerate(((Za, la), (Zb, lb))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ys = np.arange(Z.shape[0], dtype=float)
            X, Y = np.meshgrid(xs, ys)
            rs = max(1, Z.shape[0] // 160)
            cs = max(1, Z.shape[1] // 320)
            s = ax.plot_surface(X, Y, Z, facecolors=cm.coolwarm(
                norm(Z)), rstride=rs, cstride=cs, linewidth=0,
                antialiased=False, shade=False)
            s.set_rasterized(True)
            ax.set_zlim(vmin, vmax)
            if boundary is not None:
                bx = np.full((2, 2), float(boundary))
                by = np.array([[ys[0], ys[-1]], [ys[0], ys[-1]]])
                bz = np.array([[vmin, vmin], [vmax, vmax]])
                ax.plot_surface(bx, by, bz, color="k", alpha=0.18)
            ax.set_xlabel(xlab, fontsize=8, labelpad=12)
            ax.set_ylabel(ylab, fontsize=8, labelpad=8)
            ax.set_zlabel(zlab, fontsize=8, labelpad=8)
            ax.set_title(lab, fontsize=11)
            if boundary is not None:
                ax.text(boundary * 0.4, ys[-1], vmax,
                        "Embedding-side input", fontsize=8)
                ax.text(boundary * 1.4, ys[-1], vmax,
                        "Hidden-side input", fontsize=8)
            ax.view_init(**CAM)
        m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
        m.set_array(np.concatenate([Za.ravel(), Zb.ravel()]))
        fig.colorbar(m, ax=fig.axes, shrink=0.5, pad=0.06,
                     label=zlab)
        fig.suptitle(title or name.replace("_", " "), fontsize=12)
        self._save(fig, sub, name, pdf=False)
        self.n3d += 1
        self._data(name, dict(Z_left=Za, Z_right=Zb, x=xs),
                   dict(figure=name, scale="linear", left=la,
                        right=lb,
                        norm=[vmin, vmax], **(meta or {})))
        print(f"[fig] {sub}/{name}")

    def heat(self, sub, name, Z, xlab, ylab, zlab, xvals=None,
             boundary=None, norm=None, title=None, ylines=None,
             center0=False, meta=None):
        Z = np.asarray(Z, float)
        xs = np.asarray(xvals if xvals is not None
                        else np.arange(Z.shape[1]), dtype=float)
        if norm is None:
            if center0:
                M = float(np.abs(Z).max())
                norm = colors.Normalize(vmin=-M, vmax=M)
            else:
                norm = colors.Normalize(vmin=float(Z.min()),
                                        vmax=float(Z.max()))
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(Z, cmap="coolwarm", norm=norm, aspect="auto",
                       origin="lower",
                       extent=[xs[0], xs[-1], 0, Z.shape[0]])
        fig.colorbar(im, ax=ax, label=zlab)
        if boundary is not None:
            ax.axvline(boundary, color="k", lw=1.4)
            bb = dict(facecolor="white", alpha=0.75,
                      edgecolor="none", pad=1.5)
            ax.text(boundary * 0.4, Z.shape[0] * 0.94,
                    "Embedding input region", fontsize=9,
                    ha="center", bbox=bb)
            ax.text(boundary * 1.5, Z.shape[0] * 0.94,
                    "Hidden-feature input region", fontsize=9,
                    ha="center", bbox=bb)
        if ylines:
            for yv, lab in ylines:
                ax.axhline(yv, color="k", ls="--", lw=0.8)
                ax.text(xs[-1] * 1.005, yv, lab, fontsize=7,
                        va="bottom")
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title or name, fontsize=11)
        self._save(fig, sub, name)
        self.nheat += 1
        self._data(name, dict(Z=Z, x=xs),
                   dict(figure=name, scale="linear",
                        norm=[float(norm.vmin), float(norm.vmax)],
                        **(meta or {})))
        print(f"[fig] {sub}/{name}")


# ------------------------------------------------------------- stats
def region_stats(A, prefix):
    """A: [N, 2D] abs already taken? No — raw values; abs here."""
    out = {}
    for reg, sl in (("embedding", slice(0, D)),
                    ("hidden", slice(D, 2 * D))):
        v = np.abs(A[:, sl]).astype(np.float64)
        out[f"{prefix}_{reg}_absmax"] = float(v.max())
        out[f"{prefix}_{reg}_rms"] = float(np.sqrt((v ** 2).mean()))
        out[f"{prefix}_{reg}_p99"] = float(np.percentile(v, 99))
        out[f"{prefix}_{reg}_p999"] = float(np.percentile(v, 99.9))
    out[f"{prefix}_e_over_h_rms"] = (out[f"{prefix}_embedding_rms"]
                                     / out[f"{prefix}_hidden_rms"])
    out[f"{prefix}_e_over_h_p999"] = (out[f"{prefix}_embedding_p999"]
                                      / out[f"{prefix}_hidden_p999"])
    return out


def nmse(y, ref):
    y = y.double(); ref = ref.double()
    return float(((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-30))


def act_code_stats(X, dev):
    """Per-token asym 4-bit code statistics (16 levels)."""
    x = X.float()
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    s = (mx - mn).clamp_min(1e-12) / 15.0
    code = torch.round((x - mn) / s).clamp(0, 15)
    zero_code = torch.round((0 - mn) / s).clamp(0, 15)
    sat = ((code == 0) | (code == 15)).float().mean()
    zrate = (code == zero_code).float().mean()
    util = [torch.unique(code[i]).numel() / 16.0
            for i in range(0, code.shape[0],
                           max(1, code.shape[0] // 64))]
    return dict(saturation_rate=float(sat),
                zero_code_rate=float(zrate),
                level_utilization=float(sum(util) / len(util)))


def w_code_stats(Wq_scaledback, W):
    """Symmetric per-out-channel 4-bit codes recovered from dequant."""
    s = Wq_scaledback.abs().amax(dim=1, keepdim=True).clamp_min(
        1e-12) / 7.0
    code = torch.round(Wq_scaledback / s).clamp(-8, 7)
    sat = ((code == 7) | (code == -8)).float().mean()
    zrate = (code == 0).float().mean()
    util = [torch.unique(code[i]).numel() / 16.0
            for i in range(0, code.shape[0], 64)]
    return dict(saturation_rate=float(sat),
                zero_code_rate=float(zrate),
                level_utilization=float(sum(util) / len(util)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--act-group", type=int, default=16)
    ap.add_argument("--act-agg", default="meanabs",
                    choices=["meanabs", "maxabs"])
    ap.add_argument("--w-group", type=int, default=16)
    ap.add_argument("--w-agg", default="meanabs",
                    choices=["meanabs", "rmsabs", "maxabs"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rd = args.run_dir
    dev = args.device if torch.cuda.is_available() else "cpu"
    blob = torch.load(os.path.join(rd, "plot_data", "ep3p_tensors.pt"),
                      map_location="cpu", weights_only=False)
    man = json.load(open(os.path.join(rd, "metadata",
                                      "collection_manifest.json")))
    m_first, m_rec = man["m_first"], man["m_rec"]
    ga, gw, aa, wa = (args.act_group, args.w_group, args.act_agg,
                      args.w_agg)
    json.dump(dict(act_group=ga, act_agg=aa, w_group_in=gw,
                   w_group_out=gw, w_agg=wa,
                   token_policy="every collected token retained",
                   note="grouped mode; full-resolution stats in "
                        "tables/; raw tensors in plot_data/"
                        "ep3p_tensors.pt"),
              open(os.path.join(rd, "metadata",
                                "visualization_manifest.json"), "w"),
              indent=1)
    F = Fig(rd, dict(checkpoint_sha256=man["anchor_sha256"],
                     calibration_manifest=man["calibration"],
                     beta_first=man["beta_first"],
                     beta_rec=man["beta_rec"], m_first=m_first,
                     m_rec=m_rec, grouping=dict(act=ga, w=gw),
                     aggregation=dict(act=aa, w=wa)))
    xs_act = np.arange(2 * D // ga) * ga + ga / 2
    xs_w = np.arange(2 * D // gw) * gw + gw / 2
    xs_out = np.arange(D // ga) * ga + ga / 2

    stats_rows = {"first": {}, "recurrent": {}}
    wstats = {}
    qstats = {}

    # ---------------- activations: first + rec depths + rec all
    def act_block(tag, sub, X_raw, m, depth=None, ylines=None):
        """X_raw torch [N,2D] raw; m = migration factor for this path."""
        Xb = X_raw.float().numpy()
        Xa = Xb.copy()
        Xa[:, :D] *= m
        Gb = group_rows(np.abs(Xb), ga, aa)
        Ga = group_rows(np.abs(Xa), ga, aa)
        vmax = float(max(Gb.max(), Ga.max()))
        norm = colors.Normalize(vmin=0.0, vmax=vmax)
        meta = dict(source_tensor=f"X_{tag}", path=tag,
                    original_shape=list(Xb.shape),
                    plotted_shape=list(Gb.shape),
                    migration_factor=m, recurrent_depth=depth)
        F.surf(sub, f"{tag}_projection_input_before_3d", Gb,
               "projection input channel", "token index",
               f"abs activation ({aa}, group {ga})", xvals=xs_act,
               boundary=D, norm=norm, ylines=ylines,
               title=f"{tag}: complete projection input BEFORE "
                     "migration", meta=meta, log_companion=True)
        F.surf(sub, f"{tag}_projection_input_after_3d", Ga,
               "projection input channel", "token index",
               f"abs activation ({aa}, group {ga})", xvals=xs_act,
               boundary=D, norm=norm, ylines=ylines,
               title=f"{tag}: complete projection input AFTER "
                     f"migration (e x {m:.2f})", meta=meta,
               log_companion=True)
        F.pair(sub, f"{tag}_projection_input_before_after_3d",
               Gb, Ga, "before migration",
               f"after migration (e x {m:.2f})",
               "projection input channel", "token index",
               f"abs activation ({aa}, group {ga})", xvals=xs_act,
               boundary=D,
               title=f"{tag}: before vs after (single normalization)",
               meta=meta)
        F.surf(sub, f"{tag}_projection_input_difference_3d",
               Ga - Gb, "projection input channel", "token index",
               "abs(after) - abs(before)", xvals=xs_act, boundary=D,
               ylines=ylines,
               title=f"{tag}: migration difference "
                     "(hidden region must be 0)", meta=meta)
        if tag == "first":
            F.surf(sub, f"{tag}_projection_input_ratio_3d",
                   Ga / (Gb + EPS), "projection input channel",
                   "token index", "abs(after)/(abs(before)+eps)",
                   xvals=xs_act, boundary=D,
                   title=f"{tag}: migration ratio (embedding ~"
                         f"{m:.1f}x, hidden ~1x)", meta=meta)
        F.heat("heatmaps", f"{tag}_projection_input_before_heatmap",
               Gb, "projection input channel", "token index",
               f"abs activation ({aa}, group {ga})", xvals=xs_act,
               boundary=D, norm=norm, ylines=ylines,
               title=f"{tag}: input before migration", meta=meta)
        F.heat("heatmaps", f"{tag}_projection_input_after_heatmap",
               Ga, "projection input channel", "token index",
               f"abs activation ({aa}, group {ga})", xvals=xs_act,
               boundary=D, norm=norm, ylines=ylines,
               title=f"{tag}: input after migration", meta=meta)
        if tag == "first":
            F.heat("heatmaps",
                   f"{tag}_projection_input_difference_heatmap",
                   Ga - Gb, "projection input channel", "token index",
                   "abs(after) - abs(before)", xvals=xs_act,
                   boundary=D, center0=True,
                   title=f"{tag}: difference heatmap", meta=meta)
        return Xb, Xa

    Xf = blob["X_first_raw"]
    Xb_first, Xa_first = act_block("first", "activations", Xf,
                                   m_first)
    stats_rows["first"].update(region_stats(Xb_first, "before"))
    stats_rows["first"].update(region_stats(Xa_first, "after"))

    rec_blocks = []
    for k in (1, 2, 3, 4):
        Xk = blob[f"X_rec{k}_raw"]
        if Xk.shape[0] == 0:
            continue
        act_block(f"recurrent_depth{k}", "activations", Xk, m_rec,
                  depth=k)
        rec_blocks.append((k, Xk))
    X_rec_all = torch.cat([x for _, x in rec_blocks])
    ylines, off = [], 0
    for k, x in rec_blocks:
        off += x.shape[0]
        ylines.append((off, f"depth {k} end"))
    Xb_rec, Xa_rec = act_block("recurrent_all", "activations",
                               X_rec_all, m_rec,
                               depth="1-4 concatenated (token axis, "
                                     "depth order)", ylines=ylines)
    stats_rows["recurrent"].update(region_stats(Xb_rec, "before"))
    stats_rows["recurrent"].update(region_stats(Xa_rec, "after"))

    # ---------------- weights: one complete matrix, three variants
    W0 = blob["W_before"].float()               # [4096, 8192] out,in
    Wf = W0.clone(); Wf[:, :D] /= m_first
    Wr = W0.clone(); Wr[:, :D] /= m_rec
    G0 = group2d(np.abs(W0.numpy()), gw, gw, wa)
    Gf = group2d(np.abs(Wf.numpy()), gw, gw, wa)
    Gr = group2d(np.abs(Wr.numpy()), gw, gw, wa)
    wnorm = colors.Normalize(vmin=0.0, vmax=float(max(
        G0.max(), Gf.max(), Gr.max())))
    wmeta = dict(source_tensor="W_before (unmigrated fold)",
                 original_shape=list(W0.shape),
                 plotted_shape=list(G0.shape),
                 orientation="[out_channel, in_channel]; plotted "
                             "x=input channel (dim 1), y=output "
                             "channel (dim 0)")
    wx = "inner/input channel"
    wy = "outer/output channel"
    wz = f"abs weight ({wa}, {gw}x{gw} blocks)"
    for name, G, t in (
            ("projection_weight_original_3d", G0,
             "complete projection weight, BEFORE migration"),
            ("projection_weight_first_migrated_3d", Gf,
             f"first-path migrated (W_e / {m_first:.2f})"),
            ("projection_weight_recurrent_migrated_3d", Gr,
             f"recurrent-path migrated (W_e / {m_rec:.2f})")):
        F.surf("weights", name, G, wx, wy, wz, xvals=xs_w, boundary=D,
               norm=wnorm, title=t,
               region_labels=("Embedding-side input",
                              "Hidden-side input"),
               meta={**wmeta, "figure_variant": name},
               log_companion=True)
    F.pair("weights", "projection_weight_original_vs_first_3d", G0,
           Gf, "original", f"first migrated (/ {m_first:.2f})", wx,
           wy, wz, xvals=xs_w, boundary=D, meta=wmeta,
           title="complete projection weight: original vs first-path "
                 f"migration (W_e / {m_first:.2f})")
    F.pair("weights", "projection_weight_original_vs_recurrent_3d",
           G0, Gr, "original", f"recurrent migrated (/ {m_rec:.2f})",
           wx, wy, wz, xvals=xs_w, boundary=D, meta=wmeta,
           title="complete projection weight: original vs recurrent-"
                 f"path migration (W_e / {m_rec:.2f})")
    F.pair("weights", "projection_weight_first_vs_recurrent_3d", Gf,
           Gr, f"first migration (W_e / {m_first:.2f})",
           f"recurrent migration (W_e / {m_rec:.2f})", wx, wy, wz,
           xvals=xs_w, boundary=D, meta=wmeta,
           title="complete projection weight under the two pathwise "
                 "migration factors (single normalization)")
    F.surf("weights", "projection_weight_first_difference_3d",
           Gf - G0, wx, wy, "abs(first) - abs(original)",
           xvals=xs_w, boundary=D,
           title="first migration difference (hidden-side must be 0)",
           meta=wmeta)
    F.surf("weights", "projection_weight_recurrent_difference_3d",
           Gr - G0, wx, wy, "abs(recurrent) - abs(original)",
           xvals=xs_w, boundary=D,
           title="recurrent migration difference (hidden-side must "
                 "be 0)", meta=wmeta)
    F.surf("weights", "projection_weight_first_ratio_3d",
           Gf / (G0 + EPS), wx, wy, "abs(first)/(abs(original)+eps)",
           xvals=xs_w, boundary=D,
           title=f"first ratio (embedding-side ~1/{m_first:.1f}, "
                 "hidden-side ~1)", meta=wmeta)
    F.surf("weights", "projection_weight_recurrent_ratio_3d",
           Gr / (G0 + EPS), wx, wy,
           "abs(recurrent)/(abs(original)+eps)", xvals=xs_w,
           boundary=D,
           title=f"recurrent ratio (embedding-side ~1/{m_rec:.1f}, "
                 "hidden-side ~1)", meta=wmeta)
    for name, G, c0 in (
            ("projection_weight_original_heatmap", G0, False),
            ("projection_weight_first_migrated_heatmap", Gf, False),
            ("projection_weight_recurrent_migrated_heatmap", Gr,
             False),
            ("projection_weight_first_difference_heatmap", Gf - G0,
             True),
            ("projection_weight_recurrent_difference_heatmap",
             Gr - G0, True)):
        F.heat("heatmaps", name, G, wx, wy, wz, xvals=xs_w,
               boundary=D, norm=None if c0 else wnorm, center0=c0,
               title=name, meta=wmeta)
    for W, pfx in ((W0, "before"), (Wf, "first_after"),
                   (Wr, "recurrent_after")):
        wstats.update(region_stats(W.numpy(), pfx))

    # ---------------- FP function preservation + quantized effect
    bias = blob.get("bias")
    b_t = (bias.float().to(dev) if bias is not None else None)

    def out_figs(tag, Xb, Xa, Wafter, m):
        Xb_t = torch.from_numpy(Xb).float().to(dev)
        Xa_t = torch.from_numpy(Xa).float().to(dev)
        W0_t = W0.to(dev)
        Wa_t = Wafter.to(dev)
        # repository bias handling: bias is NOT migrated (identical on
        # both sides) and is added after the matmul, matching the
        # deployed FakeW4A4Linear (bias kept fp16, unquantized)
        Yb = Xb_t @ W0_t.t()
        Ya = Xa_t @ Wa_t.t()
        if b_t is not None:
            Yb = Yb + b_t
            Ya = Ya + b_t
        err = (Ya - Yb).abs()
        fp = dict(max_abs_error=float(err.max()),
                  mean_abs_error=float(err.mean()),
                  relative_error=float(err.max()
                                       / Yb.abs().max().clamp_min(
                                           1e-30)),
                  nmse=nmse(Ya, Yb),
                  cosine=float(torch.nn.functional.cosine_similarity(
                      Ya.flatten(), Yb.flatten(), dim=0)))
        Gyb = group_rows(Yb.abs().cpu().numpy(), ga, aa)
        Gya = group_rows(Ya.abs().cpu().numpy(), ga, aa)
        onorm = colors.Normalize(0.0, float(max(Gyb.max(),
                                                Gya.max())))
        ometa = dict(path=tag, migration_factor=m,
                     original_shape=list(Yb.shape),
                     plotted_shape=list(Gyb.shape),
                     mode="FP-equivalence (NO fake quantization)")
        F.surf("outputs", f"{tag}_projection_output_abs_before_3d",
               Gyb, "output channel", "token index",
               f"abs output ({aa}, group {ga})", xvals=xs_out,
               norm=onorm, title=f"{tag}: FP output, original "
               "geometry", meta=ometa)
        F.surf("outputs", f"{tag}_projection_output_abs_after_3d",
               Gya, "output channel", "token index",
               f"abs output ({aa}, group {ga})", xvals=xs_out,
               norm=onorm, title=f"{tag}: FP output, migrated "
               "geometry", meta=ometa)
        Ge = group_rows(err.cpu().numpy(), ga, "maxabs")
        F.surf("outputs", f"{tag}_projection_output_error_3d", Ge,
               "output channel", "token index",
               "abs FP error (max in group)", xvals=xs_out,
               title=f"{tag}: FP-equivalence error "
                     f"(max {fp['max_abs_error']:.2e})", meta=ometa)
        # quantized-effect (explicitly labeled, never mixed with FP)
        from eagle_spinquant import fake_w4a4_draft as fq
        aq = fq._act_quantizer(4)
        aq.find_params(Xb_t.half()); Xbq = aq(Xb_t.half()).float()
        aq.free()
        aq = fq._act_quantizer(4)
        aq.find_params(Xa_t.half()); Xaq = aq(Xa_t.half()).float()
        aq.free()
        W0q = fq._weight_fake_quant(W0_t.half(), 4).float()
        Waq = fq._weight_fake_quant(Wa_t.half(), 4).float()
        Yref = Yb
        Ynaive = Xbq @ W0q.t()
        Yep3p = Xaq @ Waq.t()
        if b_t is not None:
            Ynaive = Ynaive + b_t
            Yep3p = Yep3p + b_t
        En = (Ynaive - Yref).abs()
        Ee = (Yep3p - Yref).abs()
        q = dict(
            A4_nmse_before=nmse(Xbq, Xb_t),
            A4_nmse_after=nmse(Xaq, Xa_t),
            W4_nmse_before=nmse(W0q, W0_t),
            W4_nmse_after=nmse(Waq, Wa_t),
            output_nmse_naive=nmse(Ynaive, Yref),
            output_nmse_ep3p=nmse(Yep3p, Yref),
            act_codes_before=act_code_stats(Xb_t.cpu(), dev),
            act_codes_after=act_code_stats(Xa_t.cpu(), dev),
            w_codes_before=w_code_stats(W0q.cpu(), W0),
            w_codes_after=w_code_stats(Waq.cpu(), Wafter))
        Gn = group_rows(En.cpu().numpy(), ga, "meanabs")
        Gep = group_rows(Ee.cpu().numpy(), ga, "meanabs")
        qmeta = dict(path=tag, migration_factor=m,
                     mode="quantized-error comparison (W4A4 fake "
                          "quant) — separate from FP-equivalence")
        F.pair("quantized_effect",
               f"{tag}_quantized_projection_error_3d", Gn, Gep,
               f"naive W4A4 (NMSE {q['output_nmse_naive']:.4f})",
               f"EP3-P W4A4 (NMSE {q['output_nmse_ep3p']:.4f})",
               "output channel", "token index",
               "abs(Y_q - Y_ref) (mean in group)", xvals=xs_out,
               title=f"{tag}: quantized projection error, naive vs "
                     "EP3-P (single normalization)", meta=qmeta)
        Mx = float(max(Gn.max(), Gep.max()))
        qn = colors.Normalize(0.0, Mx)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        for axi, (Gz, lab) in zip(axes, ((Gn, "naive W4A4"),
                                         (Gep, "EP3-P W4A4"))):
            im = axi.imshow(Gz, cmap="coolwarm", norm=qn,
                            aspect="auto", origin="lower",
                            extent=[0, D, 0, Gz.shape[0]])
            axi.set_title(f"{lab} abs error vs FP reference")
            axi.set_xlabel("output channel")
        axes[0].set_ylabel("token index")
        fig.colorbar(im, ax=axes, shrink=0.8,
                     label="abs(Y_q - Y_ref)")
        F._save(fig, "quantized_effect",
                f"{tag}_naive_vs_ep3p_error_heatmap")
        F.nheat += 1
        F._data(f"{tag}_naive_vs_ep3p_error_heatmap",
                dict(Z_naive=Gn, Z_ep3p=Gep),
                dict(figure=f"{tag}_naive_vs_ep3p_error_heatmap",
                     norm=[0.0, Mx], **qmeta))
        print(f"[fig] quantized_effect/{tag}_naive_vs_ep3p_error_"
              f"heatmap")
        return fp, q, (Gyb, Gya, Gn, Gep)

    fp_first, q_first, panels_first = out_figs(
        "first", Xb_first, Xa_first, Wf, m_first)
    fp_rec, q_rec, panels_rec = out_figs(
        "recurrent", Xb_rec, Xa_rec, Wr, m_rec)

    # ---------------- tables
    import csv as _csv

    def wcsv(name, d):
        p = os.path.join(rd, "tables", name)
        with open(p, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in d.items():
                w.writerow([k, json.dumps(v) if isinstance(v, dict)
                            else v])
        print(f"[table] {name}")

    wcsv("ep3p_first_statistics.csv",
         {**stats_rows["first"], **{f"fp_{k}": v
                                    for k, v in fp_first.items()}})
    wcsv("ep3p_recurrent_statistics.csv",
         {**stats_rows["recurrent"], **{f"fp_{k}": v
                                        for k, v in fp_rec.items()}})
    wcsv("ep3p_weight_statistics.csv", wstats)
    wcsv("ep3p_quantized_error_statistics.csv",
         {**{f"first_{k}": v for k, v in q_first.items()},
          **{f"recurrent_{k}": v for k, v in q_rec.items()}})
    summary = dict(
        fp_first=fp_first, fp_recurrent=fp_rec,
        quant_first={k: v for k, v in q_first.items()
                     if not isinstance(v, dict)},
        quant_recurrent={k: v for k, v in q_rec.items()
                         if not isinstance(v, dict)},
        n_3d=F.n3d, n_heatmaps=F.nheat,
        act_rms_ratio_first=dict(
            before=stats_rows["first"]["before_e_over_h_rms"],
            after=stats_rows["first"]["after_e_over_h_rms"]),
        act_rms_ratio_recurrent=dict(
            before=stats_rows["recurrent"]["before_e_over_h_rms"],
            after=stats_rows["recurrent"]["after_e_over_h_rms"]),
        w_rms_ratio=dict(
            before=wstats["before_e_over_h_rms"],
            first_after=wstats["first_after_e_over_h_rms"],
            recurrent_after=wstats["recurrent_after_e_over_h_rms"]))
    json.dump(summary, open(os.path.join(
        rd, "tables", "ep3p_viz_summary.json"), "w"), indent=1)
    print(f"[plot] done: {F.n3d} primary 3D, {F.nheat} heatmaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
