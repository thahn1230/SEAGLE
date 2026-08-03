#!/usr/bin/env python
"""Sweet-spot surfaces + 1-D beta overlays (spec §17).

Error surfaces are separable sums J_first(bf) + J_rec(br) from the
measured 1-D grids (the calibration objective is per-path separable —
stated on the figure). tau / RCAL surfaces show the actually-evaluated
(bf, br) points (masked elsewhere; never interpolated).
1-D overlays: identity vs block-diagonal vs cross vs full per path.
"""
import argparse, glob, importlib.util, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "vz", os.path.join(ROOT, "scripts", "plot_ep3p_projection_3d.py"))
vz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vz)


def grid_of(rows):
    return {round(r["beta"], 4): r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    F = vz.Fig(rd, dict(study="rep3p_beta_surfaces"))
    s1f = grid_of(json.load(open(os.path.join(
        rd, "candidates", "s1_int4_first.json"))))
    s1r = grid_of(json.load(open(os.path.join(
        rd, "candidates", "s1_int4_rec.json"))))
    s3f = json.load(open(os.path.join(rd, "candidates",
                                      "s3_int4_first.json")))
    s3r = json.load(open(os.path.join(rd, "candidates",
                                      "s3_int4_rec.json")))

    def surface(name, gf, gr, ttl):
        bf = sorted(gf)
        br = sorted(gr)
        Z = np.zeros((len(br), len(bf)))
        for i, b2 in enumerate(br):
            for j, b1 in enumerate(bf):
                Z[i, j] = gf[b1]["j_out"] + gr[b2]["j_out"]
        from matplotlib import cm
        X, Y = np.meshgrid(np.array(bf), np.array(br))
        norm = colors.Normalize(float(Z.min()), float(Z.max()))
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=cm.coolwarm(norm(Z)),
                        rstride=1, cstride=1, linewidth=0,
                        shade=False)
        m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
        m.set_array(Z)
        fig.colorbar(m, ax=ax, shrink=0.55,
                     label="W4A4 projection-output NMSE")
        ax.set_xlabel("beta_first"); ax.set_ylabel("beta_recurrent")
        ax.set_zlabel("output NMSE (separable sum)")
        ax.set_title(ttl + " (separable J_first+J_rec, measured "
                     "grids)")
        ax.view_init(elev=28, azim=-60)
        fig.savefig(os.path.join(rd, "plots", name + ".png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(Z, cmap="coolwarm", norm=norm, aspect="auto",
                       origin="lower",
                       extent=[bf[0], bf[-1], br[0], br[-1]])
        fig.colorbar(im, ax=ax, label="output NMSE")
        ax.set_xlabel("beta_first"); ax.set_ylabel("beta_recurrent")
        ax.set_title(ttl)
        fig.savefig(os.path.join(rd, "heatmaps",
                                 name + "_heatmap.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(os.path.join(rd, "plot_data",
                                         name + ".npz"),
                            Z=Z, bf=np.array(bf), br=np.array(br))
        F.n3d += 1; F.nheat += 1
        print(f"[beta] {name}")
    surface("identity_rotation_beta_error_surface", s1f, s1r,
            "identity rotation: coupled beta sweet spot")
    gf = grid_of(s3f["cross"]["grid"])
    gr = grid_of(s3r["cross"]["grid"])
    surface("rotated_beta_error_surface", gf, gr,
            "cross-rotated: beta error surface")

    # tau / RCAL surfaces: evaluated points only
    c9 = json.load(open(os.path.join(rd, "candidates",
                                     "s5_calib9.json")))
    s5 = json.load(open(os.path.join(rd, "candidates",
                                     "s5_mtbench.json")))
    met_p = os.path.join(rd, "tables", "rcal_metrics.json")
    met = json.load(open(met_p)) if os.path.exists(met_p) else {}
    pts_tau = [(r["first"]["beta"], r["rec"]["beta"],
                r["calib_tau"]) for r in c9]
    pts_tau_id = [(0.40, 0.45,
                   s5["mtbench"].get("MTX_EP3P"))]
    for name, pts, ttl in (
            ("rotated_beta_tau_surface", pts_tau,
             "rotated: held-out calib tau at evaluated pairs"),
            ("identity_rotation_beta_tau_surface", pts_tau_id,
             "identity: tau at the deployed pair")):
        bf = sorted({p[0] for p in pts})
        br = sorted({p[1] for p in pts})
        Z = np.full((len(br), len(bf)), np.nan)
        for a, b, v in pts:
            if v is not None:
                Z[br.index(b), bf.index(a)] = v
        F.surf("plots", name, Z, "beta_first", "beta_recurrent",
               "tau", xvals=np.array(bf),
               title=ttl + " (unevaluated cells masked)")
    rc_id = met.get("RC2_EP3P__mtbench", {}).get("RCAL")
    rc_rot = met.get("RC2_REP3P__mtbench", {}).get("RCAL")
    best = s5["best"]
    for name, pts, ttl in (
            ("identity_rotation_beta_rcal_surface",
             [(0.40, 0.45, rc_id)], "identity: RCAL"),
            ("rotated_beta_rcal_surface",
             [(best["first"]["beta"], best["rec"]["beta"],
               rc_rot)], "rotated: RCAL")):
        Z = np.array([[p[2] if p[2] else np.nan for p in pts]])
        F.surf("plots", name, Z, "beta_first", "beta_recurrent",
               "RCAL", xvals=np.array([p[0] for p in pts]),
               title=ttl + " (evaluated deployment points only)")

    # 1-D overlays per path
    for path, s1g, s3d in (("first", s1f, s3f), ("rec", s1r, s3r)):
        for metric, lab in (("a4_nmse", "activation A4 NMSE"),
                            ("w4_nmse", "weight W4 NMSE"),
                            ("j_out", "projection-output NMSE")):
            fig, ax = plt.subplots(figsize=(8, 5))
            rows = []
            xs = sorted(s1g)
            ax.plot(xs, [s1g[b][metric] for b in xs], "k-o", ms=3,
                    label="identity")
            rows += [("identity", b, s1g[b][metric]) for b in xs]
            for fam, c in (("dual", "tab:orange"),
                           ("cross", "tab:blue"),
                           ("full", "tab:red")):
                if fam not in s3d:
                    continue
                g = grid_of(s3d[fam]["grid"])
                xs2 = sorted(g)
                ax.plot(xs2, [g[b][metric] for b in xs2], "-o",
                        ms=3, color=c, label=fam)
                rows += [(fam, b, g[b][metric]) for b in xs2]
            ax.set_yscale("log")
            ax.set_xlabel(f"beta_{path}")
            ax.set_ylabel(lab + " (log)")
            ax.set_title(f"{path}: {lab} vs beta — does rotation "
                         "widen or lower the sweet spot?")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            name = f"beta_curve_{path}_{metric}"
            fig.savefig(os.path.join(rd, "plots", name + ".png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            import csv
            with open(os.path.join(rd, "plot_data",
                                   name + ".csv"), "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["family", "beta", metric])
                w.writerows(rows)
            print(f"[beta] {name}")
    print(f"[beta] surfaces done ({F.n3d} 3D)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
