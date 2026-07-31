#!/usr/bin/env python
"""16 required 3D figures + companion 2D heatmaps (study §26).

coolwarm colormap, one global normalization per figure, color bar,
labeled axes, >=250 DPI. Every surface's matrix saved as NPZ (and the
heatmap doubles as the 2D companion). Missing inputs skip gracefully so
the script can rerun as the study fills in.
"""
import argparse, csv, glob, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

DPI = 260
D = 4096


def surf(rd, name, Z, xlab, ylab, zlab, xt=None, yt=None, title=None,
         xvals=None, yvals=None):
    """Z: 2D array (ny, nx). Draw 3D surface + companion heatmap."""
    pdir = os.path.join(rd, "plots")
    os.makedirs(os.path.join(pdir, "data"), exist_ok=True)
    Zm = np.ma.masked_invalid(np.asarray(Z, dtype=float))
    if Zm.count() == 0:
        print(f"[3d] {name}: no data, skip")
        return
    norm = colors.Normalize(vmin=float(Zm.min()), vmax=float(Zm.max()))
    xs = np.asarray(xvals if xvals is not None
                    else np.arange(Zm.shape[1]), dtype=float)
    ys = np.asarray(yvals if yvals is not None
                    else np.arange(Zm.shape[0]), dtype=float)
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    Zp = np.where(np.isfinite(Z), Z, np.nanmin(Zm))
    s = ax.plot_surface(X, Y, Zp, facecolors=cm.coolwarm(norm(Zp)),
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False)
    s.set_rasterized(True)
    m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    m.set_array(Zm)
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label=zlab)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_zlabel(zlab)
    if xt:
        ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=6)
    if yt:
        ax.set_yticks(ys); ax.set_yticklabels(yt, fontsize=6)
    ax.set_title(title or name)
    ax.view_init(elev=28, azim=-60)
    fig.savefig(os.path.join(pdir, name + ".png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    # companion heatmap
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(Zm, cmap="coolwarm", norm=norm, aspect="auto",
                   origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    fig.colorbar(im, ax=ax, label=zlab)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if xt:
        ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=6,
                                              rotation=60)
    if yt:
        ax.set_yticks(ys); ax.set_yticklabels(yt, fontsize=6)
    ax.set_title((title or name) + " (heatmap)")
    fig.savefig(os.path.join(pdir, name + "_heatmap.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    np.savez(os.path.join(pdir, "data", name + ".npz"),
             Z=np.asarray(Z, dtype=float), x=xs, y=ys)
    print(f"[3d] {name}")


def beta_grid(entries, key):
    """entries: list of dicts with beta_first/beta_rec/<key> ->
    (Z, bf_vals, br_vals) with NaN holes."""
    bf = sorted({round(e["beta_first"], 4) for e in entries})
    br = sorted({round(e["beta_rec"], 4) for e in entries})
    Z = np.full((len(br), len(bf)), np.nan)
    for e in entries:
        if e.get(key) is None:
            continue
        Z[br.index(round(e["beta_rec"], 4)),
          bf.index(round(e["beta_first"], 4))] = e[key]
    return Z, bf, br


def tau_of(path):
    taus = [t for r in csv.DictReader(open(path))
            for t in json.loads(r["acceptance_list"])]
    return sum(taus) / max(len(taus), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    tb = os.path.join(rd, "tables")
    nd = {}
    for tgt in ("fp16", "int4"):
        p = os.path.join(tb, f"path_distributions_{tgt}.npz")
        if os.path.exists(p):
            nd[tgt] = np.load(p)

    # 1-3. hidden-path channel stats: rows = target x path, e||h concat
    for stat in ("absmax", "rms", "p999"):
        rows, yt = [], []
        for tgt in nd:
            for path in ("first", "rec"):
                rows.append(np.concatenate(
                    [nd[tgt][f"{path}_e_{stat}_g64"],
                     nd[tgt][f"{path}_h_{stat}_g64"]]))
                yt.append(f"{tgt}:{path}")
        if rows:
            surf(rd, f"hidden_path_channel_{stat}_3d",
                 np.stack(rows), "64-channel group "
                 "(0-63 e-slice, 64-127 h-slice)", "target:path",
                 stat, yt=yt,
                 title=f"projection-input channel {stat} "
                       "(group=64)")

    # 4-5. embedding vs hidden range per target
    for tgt, fname in (("fp16", "embedding_hidden_range_3d_"
                        "fp16_target"),
                       ("int4", "embedding_hidden_range_3d_"
                        "w4a4_target")):
        if tgt not in nd:
            continue
        rows, yt = [], []
        for path in ("first", "rec"):
            for sl in ("e", "h"):
                rows.append(nd[tgt][f"{path}_{sl}_absmax_g64"])
                yt.append(f"{path}_{sl}")
        surf(rd, fname, np.stack(rows), "64-channel group",
             "path_slice", "absmax", yt=yt,
             title=f"{tgt} target: embedding vs hidden channel range")

    # 6-7. projection weight blocks (W_first rows 0-63, W_rec 64-127)
    for stat in ("rms", "absmax"):
        for tgt in nd:
            k1, k2 = f"W_first_block_{stat}", f"W_rec_block_{stat}"
            if k1 in nd[tgt]:
                Z = np.concatenate([nd[tgt][k1], nd[tgt][k2]], axis=0)
                surf(rd, f"projection_weight_blocks_{stat}_3d"
                     + ("" if tgt == "fp16" else "_int4"),
                     Z, "column block (0-63 e, 64-127 h)",
                     "row block (0-63 W_first, 64-127 W_rec)", stat,
                     title=f"{tgt} fold: 64x64 weight-block {stat}")
            break   # spec asks one figure; fp16 fold is primary

    # 8. GQAT weight-ratio trajectory
    rows, yt = [], []
    steps = None
    for var in ("T0", "T1"):
        p = os.path.join(tb, f"gqat_rebalancing_{var}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for s in (0, 1, 2):
            tr = sorted([r for r in d["rows"]
                         if r["name"] == f"GQAT_s{s}"],
                        key=lambda r: r["step"])
            if tr:
                rows.append([r["rms_ratio"] for r in tr])
                yt.append(f"{var}_s{s}")
                steps = [r["step"] for r in tr]
    if rows:
        n = min(len(r) for r in rows)
        surf(rd, "generic_qat_weight_ratio_trajectory_3d",
             np.stack([r[:n] for r in rows]), "training step",
             "variant_seed", "W_e/W_h RMS ratio",
             xvals=steps[:n], yt=yt,
             title="does generic QAT learn P3-like rebalancing?")

    # 9. method projection error (from beta grids + legacy alphas)
    meth_rows, yt = [], []
    for tgt in ("fp16", "int4"):
        sp = os.path.join(tb, f"p3exp_search_{tgt}.json")
        selp = os.path.join(tb, f"ep3_selection_{tgt}.json")
        if not os.path.exists(sp):
            continue
        sr = json.load(open(sp))
        sel = json.load(open(selp)) if os.path.exists(selp) else {}
        la = 45.254834 if tgt == "fp16" else 32.0
        lb = math.log(la) / math.log(D)

        def near(grid, b):
            return min(grid, key=lambda e: abs(e["beta"] - b))
        g1, g2 = (sr["paths"]["first"]["grid"],
                  sr["paths"]["rec"]["grid"])
        cases = dict(
            NPTQ=(0.0, 0.0), LP3=(lb, lb),
            EP3G=((sel.get("ep3g") or {}).get("beta", lb),) * 2)
        pb = sel.get("primary_rule3") or sel.get("primary_by_al")
        if pb:
            cases["EP3P"] = (pb["beta_first"], pb["beta_rec"])
        for mname, (b1, b2) in cases.items():
            e1, e2 = near(g1, b1), near(g2, b2)
            meth_rows.append([e1["nmse_e"], e1["nmse_h"], e1["nmse"],
                              e2["nmse_e"], e2["nmse_h"], e2["nmse"]])
            yt.append(f"{tgt}:{mname}")
    if meth_rows:
        surf(rd, "method_projection_error_3d", np.stack(meth_rows),
             "error slice", "target:method", "NMSE",
             xt=["first_e", "first_h", "first_tot",
                 "rec_e", "rec_h", "rec_tot"],
             title="projection NMSE by method and path slice")

    # 10. method acceptance surface (MT-Bench tau per target x method)
    shards = glob.glob(os.path.join(rd, "shards",
                                    "al__MTX_*__mtbench.csv"))
    by = {}
    for s in shards:
        base = os.path.basename(s)[4:-4]
        tag, tgt, _ = base.split("__")
        by.setdefault(tag.replace("MTX_", ""), {})[tgt] = tau_of(s)
    if by:
        meths = sorted(by)
        Z = np.full((2, len(meths)), np.nan)
        for j, mname in enumerate(meths):
            for i, tgt in enumerate(("fp16", "int4")):
                if tgt in by[mname]:
                    Z[i, j] = by[mname][tgt]
        surf(rd, "method_acceptance_3d", Z, "method", "target",
             "MT-Bench tau", xt=meths, yt=["fp16", "int4"],
             title="method acceptance surface")

    # 11-16. beta-pair surfaces (fp16 target primary; int4 companion
    # via *_int4 names when present)
    for tgt in ("fp16", "int4"):
        sp = os.path.join(tb, f"p3exp_search_{tgt}.json")
        selp = os.path.join(tb, f"ep3_selection_{tgt}.json")
        if not os.path.exists(sp):
            continue
        sr = json.load(open(sp))
        sel = json.load(open(selp)) if os.path.exists(selp) else {}
        sfx = "" if tgt == "fp16" else "_int4"
        Z, bf, br = beta_grid(sr["pairs25"], "nmse_sum")
        surf(rd, f"beta_pair_projection_nmse_3d{sfx}", Z,
             "beta_first", "beta_recurrent", "projection NMSE",
             xvals=bf, yvals=br,
             title=f"{tgt}: beta-pair projection NMSE")
        cal = sel.get("top9_calib", [])
        if cal:
            Z, bf, br = beta_grid(cal, "calib_tau")
            surf(rd, f"beta_pair_acceptance_length_3d{sfx}", Z,
                 "beta_first", "beta_recurrent",
                 "held-out calib AL", xvals=bf, yvals=br,
                 title=f"{tgt}: beta-pair acceptance (20-prompt "
                       "calib; unevaluated cells masked)")
        rc = sel.get("top3_rcal", [])
        if rc:
            for key, fname, lab in (
                    ("RCAL", f"beta_pair_rcal_3d{sfx}", "RCAL"),
                    ("SAL", f"beta_pair_sal_3d{sfx}", "SAL"),
                    ("AFS", f"beta_pair_afs_3d{sfx}", "AFS"),
                    ("dalr", f"beta_pair_delta_al_minus_rcal_3d{sfx}",
                     "AL_q - RCAL")):
                ent = []
                for r in rc:
                    e = dict(r)
                    e["SAL"] = r["AL_q"] - r["RCAL"]
                    e["dalr"] = e["SAL"]
                    ent.append(e)
                Z, bf, br = beta_grid(ent, key)
                surf(rd, fname, Z, "beta_first", "beta_recurrent",
                     lab, xvals=bf, yvals=br,
                     title=f"{tgt}: beta-pair {lab} (top-3 evaluated "
                           "cells; others masked)")
    print("[3d] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
