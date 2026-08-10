#!/usr/bin/env python
"""Publication figures for the GS R1/R2 factorial study (spec section 22).

Each figure is written as PDF + PNG + a CSV of its source data under
<run>/figures/. Palette: validated categorical order (dataviz reference
palette slots 1-4) — A0 blue #2a78d6, A1 orange #eb6834, A2 aqua
#1baf7a, A3 yellow #eda100; controls use slots 5+ / gray. Sequential
uses one hue; no dual axes; thin marks; direct labels only where they
disambiguate.

Inputs (all produced earlier in the pipeline):
  tables/al_by_dataset.csv          arm,dataset,tau,n_prompts,n_cycles
  stats/bootstrap_pairs_<ds>.json   paired deltas + CIs
  geometry/r2_mechanism.json        depth overlap, eigenangles, proxies
  rotations/*.pt                    training logs (val curves)
  rcal/rcal_metrics.json            AL_q/RCAL/SAL/LAL per arm
"""
import argparse, csv, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

C = {"A0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a", "A3": "#eda100",
     "A4": "#e87ba4", "RAND": "#8a8a85", "BASE": "#2a78d6"}
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.dpi": 150})


def save(fig, figdir, name, rows, header):
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(figdir, name + ".png"))
    with open(os.path.join(figdir, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    plt.close(fig)
    print(f"[fig] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", default="A0,A1,A2,A3")
    args = ap.parse_args()
    rd = args.run_dir
    figdir = os.path.join(rd, "figures")
    os.makedirs(figdir, exist_ok=True)
    arms = args.arms.split(",")

    al = {}
    with open(os.path.join(rd, "tables", "al_by_dataset.csv")) as f:
        for r in csv.DictReader(f):
            al[(r["arm"], r["dataset"])] = float(r["tau"])

    # ---- fig 1: 2x2 AL by dataset ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    W = 0.8 / len(arms)
    rows = []
    for i, a in enumerate(arms):
        xs = [j + i * W for j in range(len(DS))]
        ys = [al.get((a, d), float("nan")) for d in DS]
        ax.bar(xs, ys, width=W * 0.94, color=C.get(a, "#8a8a85"),
               label=a, zorder=3)
        rows += [[a, d, y] for d, y in zip(DS, ys)]
    ax.set_xticks([j + W * (len(arms) - 1) / 2 for j in range(len(DS))])
    ax.set_xticklabels(DS)
    lo = min(v for v in al.values()) - 0.15
    ax.set_ylim(max(lo, 1.0), None)
    ax.set_ylabel("official micro-tau AL")
    ax.legend(ncol=len(arms), frameon=False, loc="upper left",
              fontsize=8)
    ax.set_title("W4A4 EAGLE-1 acceptance length: R1/R2 draft-aware 2x2 "
                 "(GS, T4)", fontsize=9)
    save(fig, figdir, "fig1_al_2x2_by_dataset", rows,
         ["arm", "dataset", "tau"])

    # ---- fig 2: four-dataset mean ----------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    rows = []
    for i, a in enumerate(arms):
        m = sum(al.get((a, d), float("nan")) for d in DS) / len(DS)
        ax.bar([i], [m], color=C.get(a, "#8a8a85"), width=0.7, zorder=3)
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
        rows.append([a, m])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms)
    ax.set_ylim(min(r[1] for r in rows) - 0.12, None)
    ax.set_ylabel("mean4 tau (descriptive)")
    ax.set_title("Four-dataset arithmetic mean", fontsize=9)
    save(fig, figdir, "fig2_mean4", rows, ["arm", "mean4_tau"])

    # ---- fig 3: per-depth overlap ----------------------------------------
    mech_p = os.path.join(rd, "geometry", "r2_mechanism.json")
    if os.path.exists(mech_p):
        mech = json.load(open(mech_p))
        name_map = {"BASE": "A0"}
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        rows = []
        for nm, e in mech.items():
            a = name_map.get(nm, nm.split("_")[0])
            if a not in arms or "depth_overlap" not in e:
                continue
            ks = [1, 2, 3, 4]
            ax.plot(ks, e["depth_overlap"], "-o", ms=4, lw=2,
                    color=C.get(a, "#8a8a85"), label=a)
            rows += [[a, k, v] for k, v in zip(ks, e["depth_overlap"])]
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("recurrent depth k")
        ax.set_ylabel("full-vocab overlap alpha_k (held-out)")
        ax.legend(frameon=False, fontsize=8)
        save(fig, figdir, "fig3_depth_overlap", rows,
             ["arm", "depth", "overlap"])

        # ---- fig 4: R2_D eigenangle distribution -------------------------
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        rows = []
        for nm, e in mech.items():
            full = e.get("r2_eigenangles_full")
            if not full:
                continue
            a = name_map.get(nm, nm.split("_")[0])
            ax.hist(full, bins=32, histtype="step", lw=2,
                    color=C.get(a, "#8a8a85"), label=nm)
            rows += [[nm, t] for t in full]
        ax.set_xlabel("relative eigenangle of R2_B^T R2_D (rad)")
        ax.set_ylabel("count (of 128)")
        ax.legend(frameon=False, fontsize=7)
        save(fig, figdir, "fig4_r2_eigenangles", rows,
             ["arm", "theta_rad"])

        # ---- fig 5: quant proxies vs AL ----------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
        rows = []
        prox = [("v W4 NMSE", lambda e: e["weights"]["v"]["w4_nmse"]),
                ("o W4 NMSE", lambda e: e["weights"]["o"]["w4_nmse"]),
                ("o_in absmax", lambda e:
                 e["activations"]["o_in"]["absmax"])]
        for ax_i, (lbl, fn) in zip(axes, prox):
            for nm, e in mech.items():
                a = name_map.get(nm, nm.split("_")[0])
                if a not in arms:
                    continue
                x, y = fn(e), al.get((a, "mtbench"))
                if y is None:
                    continue
                ax_i.scatter([x], [y], s=45, color=C.get(a, "#8a8a85"),
                             zorder=3)
                ax_i.annotate(a, (x, y), textcoords="offset points",
                              xytext=(4, 3), fontsize=7)
                rows.append([a, lbl, x, y])
            ax_i.set_xlabel(lbl)
        axes[0].set_ylabel("mtbench tau")
        fig.suptitle("Quantization proxies vs acceptance", fontsize=9)
        save(fig, figdir, "fig5_proxies_vs_al", rows,
             ["arm", "proxy", "value", "mtbench_tau"])

    # ---- fig 6: training/val curves --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    rows = []
    for ck in sorted(glob.glob(os.path.join(rd, "rotations",
                                            "RD_GS_A*_s*.pt"))):
        if ck.endswith(".final.pt"):      # last-step twin, not the arm
            continue
        nm = os.path.basename(ck)[:-3]
        a = nm.split("_")[2]
        d = torch.load(ck, map_location="cpu", weights_only=False)
        vs = [(l["step"], l["val_loss"]) for l in d["log"]
              if "val_loss" in l and l["val_loss"] is not None]
        es = [(l["step"], l["val_expected_tau"]) for l in d["log"]
              if "val_expected_tau" in l]
        if vs:
            axes[0].plot(*zip(*vs), lw=1.5, color=C.get(a, "#8a8a85"),
                         alpha=0.8)
        if es:
            axes[1].plot(*zip(*es), lw=1.5, color=C.get(a, "#8a8a85"),
                         alpha=0.8, label=nm)
        rows += [[nm, s, v, ""] for s, v in vs]
        rows += [[nm, s, "", e] for s, e in es]
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("held-out val LK loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("held-out val expected tau")
    axes[1].legend(frameon=False, fontsize=6)
    save(fig, figdir, "fig6_training_curves", rows,
         ["ckpt", "step", "val_loss", "val_expected_tau"])

    # ---- fig 7: seed variability -----------------------------------------
    shard_tau = {}
    for p in glob.glob(os.path.join(rd, "shards", "al__A*__int4__*.csv")):
        base = os.path.basename(p)[4:-4]
        tag, _tgt, ds = base.split("__")[0], None, base.split("__")[-1]
        taus = []
        with open(p) as f:
            for r in csv.DictReader(f):
                taus += json.loads(r["acceptance_list"])
        if taus:
            shard_tau[(tag, ds)] = sum(taus) / len(taus)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    rows = []
    xi = 0
    xticks, xlabels = [], []
    for a in [x for x in arms if x != "A0"]:
        for d in DS:
            pts = [(t, v) for (t, ds_), v in shard_tau.items()
                   if ds_ == d and t.startswith(a + "_s")]
            for t, v in pts:
                ax.scatter([xi], [v], s=30, color=C.get(a, "#8a8a85"),
                           zorder=3)
                rows.append([a, d, t, v])
            base_v = shard_tau.get(("A0_BASE", d))
            if base_v is not None:
                ax.plot([xi - 0.3, xi + 0.3], [base_v, base_v], lw=1.5,
                        color=C["A0"], zorder=2)
            xticks.append(xi)
            xlabels.append(f"{a}\n{d[:4]}")
            xi += 1
        xi += 0.6
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=6.5)
    ax.set_ylabel("micro-tau")
    ax.set_title("Seed variability (dots = seeds; blue line = A0)",
                 fontsize=9)
    save(fig, figdir, "fig7_seed_variability", rows,
         ["arm", "dataset", "tag", "tau"])

    # ---- fig 8: RCAL decomposition ---------------------------------------
    rcal_p = os.path.join(rd, "tables", "rcal_metrics.json")
    if os.path.exists(rcal_p):
        rc = json.load(open(rcal_p))
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        rows = []
        keys = [k for k in rc if any(k.startswith(a) for a in arms)]
        for i, k in enumerate(sorted(keys)):
            m = rc[k]
            a = k.split("_")[0]
            ax.bar([i], [m["RCAL"]], width=0.62,
                   color=C.get(a, "#8a8a85"), zorder=3,
                   label="RCAL" if i == 0 else None)
            ax.bar([i], [m["SAL"]], width=0.62, bottom=[m["RCAL"]],
                   color=C.get(a, "#8a8a85"), alpha=0.45, zorder=3,
                   hatch="//", edgecolor="white", linewidth=1,
                   label="SAL (verifier-specific)" if i == 0 else None)
            rows.append([k, m["AL_q"], m["AL_0"], m["RCAL"], m["SAL"],
                        m["LAL"], m["AFS"]])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(sorted(keys), fontsize=7)
        ax.set_ylabel("accepted proposal tokens / cycle")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("AL_q = RCAL + SAL (T4, mtbench-80)", fontsize=9)
        save(fig, figdir, "fig8_rcal_decomposition", rows,
             ["arm", "AL_q", "AL_0", "RCAL", "SAL", "LAL", "AFS"])

    print(f"[fig] all -> {figdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
