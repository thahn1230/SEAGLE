#!/usr/bin/env python
"""LRGF study figures (spec §24): 15 conventional + 4 3D (coolwarm,
heatmap+NPZ companions). Defensive: skips figures whose inputs are
absent; every figure saves its plotted data (CSV/NPZ).
"""
import argparse, csv, glob, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPI = 300


def j(rd, *p):
    fp = os.path.join(rd, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def save(fig, rd, name, rows=None, header=None, Z=None, extra=None):
    os.makedirs(os.path.join(rd, "plots"), exist_ok=True)
    os.makedirs(os.path.join(rd, "plot_data"), exist_ok=True)
    fig.savefig(os.path.join(rd, "plots", name + ".png"), dpi=DPI,
                bbox_inches="tight")
    fig.savefig(os.path.join(rd, "plots", name + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)
    if rows is not None:
        with open(os.path.join(rd, "plot_data", name + ".csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    if Z is not None:
        np.savez_compressed(os.path.join(rd, "plot_data",
                                         name + ".npz"),
                            Z=Z, **(extra or {}))
    print(f"[fig] {name}")


def surf3d(rd, name, Z, xlab, ylab, zlab, xt=None, yt=None,
           title=None, xvals=None, yvals=None):
    Z = np.ma.masked_invalid(np.asarray(Z, float))
    if Z.count() == 0:
        print(f"[fig] {name}: no data")
        return
    norm = colors.Normalize(float(Z.min()), float(Z.max()))
    xs = np.asarray(xvals if xvals is not None
                    else np.arange(Z.shape[1]), float)
    ys = np.asarray(yvals if yvals is not None
                    else np.arange(Z.shape[0]), float)
    X, Y = np.meshgrid(xs, ys)
    Zp = np.where(np.isfinite(Z), Z, float(Z.min()))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zp, facecolors=cm.coolwarm(norm(Zp)),
                    rstride=1, cstride=1, linewidth=0, shade=False)
    m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    m.set_array(Z)
    fig.colorbar(m, ax=ax, shrink=0.55, label=zlab)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_zlabel(zlab)
    if xt:
        ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=6,
                                              rotation=45)
    if yt:
        ax.set_yticks(ys); ax.set_yticklabels(yt, fontsize=7)
    ax.set_title(title or name)
    ax.view_init(elev=28, azim=-60)
    save(fig, rd, name, Z=np.asarray(Z, float),
         extra=dict(x=xs, y=ys))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, cmap="coolwarm", norm=norm, aspect="auto",
                   origin="lower")
    fig.colorbar(im, ax=ax, label=zlab)
    if xt:
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt, fontsize=6, rotation=60)
    if yt:
        ax.set_yticks(range(len(yt)))
        ax.set_yticklabels(yt, fontsize=7)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title((title or name) + " (heatmap)")
    save(fig, rd, name + "_heatmap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir

    # 1. oracle gains
    ca = j(rd, "tables", "component_audit.json")
    if ca:
        gains = [(k.replace("_gain_vs_FULL", ""), v["delta"],
                  v["ci"][0], v["ci"][1],
                  v.get("significant", False))
                 for k, v in ca["paired"].items()
                 if k.endswith("gain_vs_FULL")]
        gains.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(9, 5))
        ys = range(len(gains))
        cols = ["tab:red" if g[4] else "tab:blue" for g in gains]
        ax.barh(list(ys), [g[1] for g in gains], color=cols)
        for i, g in enumerate(gains):
            ax.plot([g[2], g[3]], [i, i], "k-", lw=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([g[0] for g in gains], fontsize=8)
        ax.axvline(0.05, ls="--", c="k", lw=0.8)
        ax.set_xlabel("restore-to-FP16 tau gain vs FULL "
                      "(red = significant & >=0.05)")
        ax.set_title("post-EP3-P oracle: where is the remaining "
                     "bottleneck?")
        save(fig, rd, "post_ep3p_component_oracle_gain",
             gains, ["component", "gain", "lo", "hi", "sig"])

    # 2-3. error flow + depth amplification
    ef = j(rd, "tables", "error_flow.json")
    if ef:
        locs = [k for k in next(iter(ef.values()))
                if not k.startswith("rec")]
        fig, ax = plt.subplots(figsize=(10, 5))
        rows = []
        for name, d in ef.items():
            xs = [i for i, k in enumerate(locs) if k in d]
            ax.plot(xs, [d[k]["nmse"] for k in locs if k in d],
                    marker="o", ms=4, label=name)
            rows += [(name, k, d[k]["nmse"]) for k in locs
                     if k in d]
        ax.set_xticks(range(len(locs)))
        ax.set_xticklabels(locs, rotation=45, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("activation NMSE vs FP16 (log)")
        ax.set_title("quantization error flow through the draft")
        ax.legend(fontsize=8)
        save(fig, rd, "quantization_error_flow_by_layer", rows,
             ["config", "location", "nmse"])
        fig, ax = plt.subplots(figsize=(8, 5))
        rows = []
        for name, d in ef.items():
            ks = [k for k in d if k.startswith("rec")
                  and k.endswith("_h")]
            dep = sorted(int(k[3]) for k in ks)
            ax.plot(dep, [d[f"rec{k}_h"]["nmse"] for k in dep],
                    marker="o", label=name)
            rows += [(name, k, d[f"rec{k}_h"]["nmse"])
                     for k in dep]
        ax.set_yscale("log")
        ax.set_xlabel("recurrent depth")
        ax.set_ylabel("hidden NMSE (log)")
        ax.set_title("error amplification by recurrent depth")
        ax.legend(fontsize=8)
        save(fig, rd, "error_amplification_by_recurrent_depth",
             rows, ["config", "depth", "nmse"])

    # 4. training curves
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for p in sorted(glob.glob(os.path.join(rd, "rotations",
                                           "*_s0_sh.json"))):
        d = json.load(open(p))
        h = d.get("hist", [])
        if not h:
            continue
        ax.plot([r["step"] for r in h],
                [r["loss"] for r in h], label=d["tag"], lw=1)
        rows += [(d["tag"], r["step"], r["loss"]) for r in h]
    ax.set_xlabel("step"); ax.set_ylabel("objective")
    ax.set_title("learned-rotation training curves (seed 0)")
    ax.legend(fontsize=7)
    save(fig, rd, "learned_rotation_training_curves", rows,
         ["tag", "step", "loss"])

    # 5-6. selection + objective scatter
    mech = j(rd, "tables", "lrgf_mechanism.json")
    if mech:
        ok = [r for r in mech["arms"]
              if r.get("heldout_tau") is not None]
        fig, ax = plt.subplots(figsize=(8, 6))
        rows = []
        for r in ok:
            x = r.get("val_rcal")
            y = r["heldout_tau"]
            c = dict(nmse="tab:gray", eagle="tab:blue",
                     acc="tab:red", rc="tab:orange",
                     hybrid="tab:green").get(r["objective"], "k")
            if x is not None:
                ax.scatter(x, y, color=c)
                ax.annotate(r["tag"], (x, y), fontsize=6,
                            textcoords="offset points",
                            xytext=(4, 2))
            rows.append((r["tag"], r["objective"], x, y))
        for k, v in (mech.get("references") or {}).items():
            ax.axhline(v, ls="--", lw=0.8, c="k")
            ax.text(ax.get_xlim()[0], v, k, fontsize=6,
                    va="bottom")
        ax.set_xlabel("validation RCAL (held-out captures)")
        ax.set_ylabel("held-out calib tau")
        ax.set_title("rotation objectives: AL vs RCAL "
                     "(colors = objective)")
        save(fig, rd, "rotation_objective_al_rcal_scatter", rows,
             ["tag", "objective", "val_rcal", "heldout_tau"])
        sel_r = mech.get("selected_max_rcal")
        rows2 = [("max_RCAL_rule", json.dumps(sel_r)),
                 ("max_heldout_tau", json.dumps(
                     mech.get("selected_max_heldout_tau")))]
        nm = [r for r in ok if r["objective"] == "nmse"]
        if nm:
            rows2.append(("min_NMSE_counterfactual",
                          json.dumps(nm[0])))
        fig, ax = plt.subplots(figsize=(8, 4))
        labs = [r[0] for r in rows2]
        vals = []
        for _, js in rows2:
            d = json.loads(js) or {}
            vals.append(d.get("heldout_tau") or 0)
        ax.bar(labs, vals, color=["tab:green", "tab:blue",
                                  "tab:gray"][: len(labs)])
        ax.set_ylabel("held-out tau")
        ax.set_title("selection rule vs proxy counterfactual")
        save(fig, rd, "nmse_selected_vs_rcal_selected", rows2,
             ["rule", "selected"])

    # 7-9. granularity figures
    gg = j(rd, "tables", "granularity_grid.json")
    if gg:
        first = [r for r in gg if r["path"] == "first"]
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r["granularity"] for r in first]
        ax.bar(names, [r["j_out"] for r in first],
               color="tab:blue")
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.set_ylabel("W4A4 projection-output NMSE")
        ax.set_title("granularity vs local error (first path; "
                     "structural, untrained)")
        save(fig, rd, "rotation_granularity_rcal",
             [(r["granularity"], r["j_out"], r["params"],
               r["flops_per_token"]) for r in first],
             ["granularity", "j_out", "params", "flops"])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter([r["flops_per_token"] for r in first],
                   [r["j_out"] for r in first])
        for r in first:
            ax.annotate(r["granularity"],
                        (r["flops_per_token"], r["j_out"]),
                        fontsize=6, textcoords="offset points",
                        xytext=(4, 2))
        ax.set_xscale("symlog")
        ax.set_xlabel("transform adds per token")
        ax.set_ylabel("j_out")
        ax.set_title("granularity latency proxy vs local error")
        save(fig, rd, "rotation_granularity_latency",
             [(r["granularity"], r["flops_per_token"], r["j_out"])
              for r in first],
             ["granularity", "flops", "j_out"])
        fig, ax = plt.subplots(figsize=(8, 5))
        pts = sorted([(r["flops_per_token"], r["j_out"],
                       r["granularity"]) for r in first])
        pareto = []
        best = 1e9
        for f, jo, g in pts:
            if jo < best:
                pareto.append((f, jo, g))
                best = jo
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                "r-o")
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=12, alpha=0.5)
        ax.set_xscale("symlog"); ax.set_ylabel("j_out")
        ax.set_xlabel("adds/token")
        ax.set_title("granularity Pareto frontier")
        save(fig, rd, "rotation_granularity_pareto", pareto,
             ["flops", "j_out", "granularity"])

    # 10. shared vs pathwise
    held = j(rd, "tables", "learned_rot_heldout.json") or {}
    sh = {k: v for k, v in held.items() if k.endswith("_sh")}
    pw = {k: v for k, v in held.items() if k.endswith("_pw")}
    if sh and pw:
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = []
        for k, v in sorted(pw.items()):
            base = k[:-3] + "_sh"
            if base in sh:
                rows.append((k[3:-3], sh[base], v))
        x = range(len(rows))
        ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4,
               label="shared Q", color="tab:blue")
        ax.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4,
               label="pathwise Q", color="tab:red")
        ax.set_xticks(list(x))
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        ax.set_ylabel("held-out tau"); ax.legend()
        ax.set_title("shared vs pathwise learned rotation")
        save(fig, rd, "shared_vs_pathwise_rotation", rows,
             ["arm", "shared", "pathwise"])

    # 11. layer rotation ablation (from component audit A-family)
    if ca:
        arms = [(k, v) for k, v in ca["tau"].items()
                if k.startswith("A")]
        arms.sort(key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([a[0] for a in arms], [a[1] for a in arms],
                color="tab:blue")
        ax.set_xlabel("tau (quantize-one arm)")
        ax.set_title("layer sensitivity (quantize-one family)")
        save(fig, rd, "layer_rotation_ablation", arms,
             ["arm", "tau"])

    # 12. foldability diagram (table render)
    fa = j(rd, "tables", "foldability_audit.json")
    if fa and "classification" in fa:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        rows = [(k, v["cls"]) for k, v in
                fa["classification"].items()]
        colmap = dict(F0="#4a90d9", F1="#7fc97f", F2="#fdb462",
                      F3="#fb8072")
        for i, (k, c) in enumerate(rows):
            base = c.split("_")[0]
            ax.barh(i, 1, color=colmap.get(base, "#cccccc"))
            ax.text(0.01, i, f"{k}  [{c}]", va="center",
                    fontsize=9)
        ax.set_title("foldability classification (F0 offline / F1 "
                     "dup-params / F2 fusable / F3 online)")
        save(fig, rd, "foldability_classification_diagram", rows,
             ["transform", "class"])

    # 13-14. kernel latency + memory traffic
    fb = j(rd, "tables", "folding_benchmark.json")
    kb = (fb or {}).get("kernel_bench") or j(rd, "tables",
                                             "kernel_bench.json")
    if kb:
        rows = [(k, v.get("fused_ms"), v.get("explicit_ms"))
                for k, v in kb.items() if isinstance(v, dict)
                and "fused_ms" in v]
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = range(len(rows))
            ax.bar([i - 0.2 for i in x],
                   [r[2] or 0 for r in rows], 0.4,
                   label="explicit (unfused torch)",
                   color="tab:red")
            ax.bar([i + 0.2 for i in x],
                   [r[1] or 0 for r in rows], 0.4,
                   label="fused Triton", color="tab:blue")
            ax.set_xticks(list(x))
            ax.set_xticklabels([r[0] for r in rows], fontsize=8)
            ax.set_ylabel("ms per call"); ax.set_yscale("log")
            ax.legend()
            ax.set_title("explicit vs fused transform+A4 latency")
            save(fig, rd, "explicit_vs_fused_kernel_latency",
                 rows, ["kernel", "fused_ms", "explicit_ms"])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            tr = [(k, v.get("bytes_rw_est"))
                  for k, v in kb.items() if isinstance(v, dict)
                  and v.get("bytes_rw_est")]
            if tr:
                ax.bar([t[0] for t in tr],
                       [t[1] / 1e6 for t in tr],
                       color="tab:blue")
                ax.set_ylabel("MB moved per call (est)")
                ax.set_title("transform kernel memory traffic")
                save(fig, rd, "transform_kernel_memory_traffic",
                     tr, ["kernel", "bytes"])

    # 15. AL/RCAL/latency pareto over deployed arms
    met = j(rd, "tables", "rcal_metrics.json") or {}
    if held:
        fig, ax = plt.subplots(figsize=(8, 6))
        rows = []
        for k, v in held.items():
            mk = f"V{k}__c4"
            rc = met.get(mk, {}).get("RCAL")
            if rc:
                ax.scatter(v, rc, c="tab:blue")
                ax.annotate(k, (v, rc), fontsize=6,
                            textcoords="offset points",
                            xytext=(4, 2))
                rows.append((k, v, rc))
        ax.set_xlabel("held-out tau")
        ax.set_ylabel("validation RCAL")
        ax.set_title("deployed arms: AL vs RCAL (validation)")
        save(fig, rd, "al_rcal_latency_pareto", rows,
             ["arm", "tau", "rcal"])

    # ---- 3D A: granularity x objective -> heldout tau
    if held:
        objs = ["nmse", "eagle", "acc", "rc", "hybrid"]
        params = ["cayley32", "givens32", "householder32"]
        Z = np.full((len(objs), len(params)), np.nan)
        for i, o in enumerate(objs):
            for jx, pm in enumerate(params):
                k = f"LR_{o}_{pm}_s0_sh"
                if k in held:
                    Z[i, jx] = held[k]
        surf3d(rd, "granularity_objective_3d", Z,
               "parameterization", "objective", "held-out tau",
               xt=params, yt=objs,
               title="objective x parameterization -> held-out tau")
    # ---- 3D B: block size x tokens -> kernel latency
    if kb:
        toks = sorted({v["n_tokens"] for v in kb.values()
                       if isinstance(v, dict)
                       and "n_tokens" in v})
        blocks = sorted({v["block"] for v in kb.values()
                         if isinstance(v, dict) and "block" in v})
        if toks and blocks:
            Z = np.full((len(toks), len(blocks)), np.nan)
            for v in kb.values():
                if isinstance(v, dict) and "block" in v:
                    Z[toks.index(v["n_tokens"]),
                      blocks.index(v["block"])] = v["fused_ms"]
            surf3d(rd, "granularity_latency_3d", Z, "block size",
                   "token count", "fused kernel ms",
                   xt=[str(b) for b in blocks],
                   yt=[str(t) for t in toks])
    # ---- 3D C: beta x beta -> RCAL (evaluated points from REP3P)
    rep = os.path.join(ROOT, open(os.path.join(
        ROOT, "runs", "REP3P_RUN_DIR")).read().strip())
    met2 = j(rep, "tables", "rcal_metrics.json") or {}
    pts = []
    for tag, bf, br in (("RC2_EP3P", 0.40, 0.45),
                        ("RC2_REP3P", 0.38, 0.44),
                        ("RC2_FIXEDBETA", 0.40, 0.45)):
        v = met2.get(f"{tag}__mtbench", {}).get("RCAL")
        if v:
            pts.append((bf, br, v))
    if pts:
        bfv = sorted({p[0] for p in pts})
        brv = sorted({p[1] for p in pts})
        Z = np.full((len(brv), len(bfv)), np.nan)
        for bf, br, v in pts:
            Z[brv.index(br), bfv.index(bf)] = v
        surf3d(rd, "beta_rotation_rcal_3d", Z, "beta_first",
               "beta_recurrent", "RCAL",
               xt=[str(b) for b in bfv],
               yt=[str(b) for b in brv],
               title="beta pair -> RCAL (evaluated deployments)")
    # ---- 3D D: layer/depth x channel-group error
    if ef and "ep3p" in ef:
        locs = [k for k in ef["ep3p"]]
        Z = np.array([[ef[c][k]["nmse"] if k in ef[c] else np.nan
                       for k in locs]
                      for c in ef])
        surf3d(rd, "layer_depth_error_3d", Z, "trace location",
               "config", "NMSE", xt=locs, yt=list(ef),
               title="layer/depth error surface")
    print("[fig] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
