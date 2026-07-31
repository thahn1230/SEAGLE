#!/usr/bin/env python
"""12 conventional RCAL figures (study §25). PNG+PDF, CSV data.

Inputs: tables/rcal_metrics.json, stats/rcal_bootstrap_mtbench.json,
tables/tree_seq_fidelity_*.json, tables/target_ppl.json (optional).
Method tags follow the RC_<METHOD>_<Tx> convention (mtbench captures).
Figures land in plots/; each has a same-stem CSV in plots/data/.
"""
import argparse, csv, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 250


def method_label(tag):
    return tag.replace("RC_", "").replace("__mtbench", "")


def save(fig, pdir, name, rows, header):
    fig.savefig(os.path.join(pdir, name + ".png"), dpi=DPI,
                bbox_inches="tight")
    fig.savefig(os.path.join(pdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    os.makedirs(os.path.join(pdir, "data"), exist_ok=True)
    with open(os.path.join(pdir, "data", name + ".csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[plot] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    pdir = os.path.join(rd, "plots")
    os.makedirs(pdir, exist_ok=True)
    met = json.load(open(os.path.join(rd, "tables",
                                      "rcal_metrics.json")))
    mt = {k: v for k, v in met.items()
          if k.startswith("RC_") and k.endswith("__mtbench")}
    labs = sorted(mt, key=lambda k: mt[k]["RCAL"])
    L = [method_label(k) for k in labs]

    # 1. AL vs RCAL scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [mt[k]["AL_q"] for k in labs]
    ys = [mt[k]["RCAL"] for k in labs]
    ax.scatter(xs, ys, c="tab:blue")
    for x, y, l in zip(xs, ys, L):
        ax.annotate(l, (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 3))
    lim = [0, max(xs + ys) * 1.08 + 0.1]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x (SAL = 0)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("deployed AL_q"); ax.set_ylabel("RCAL")
    ax.legend(); ax.set_title("AL_q vs RCAL (gap below diagonal = SAL)")
    save(fig, pdir, "al_vs_rcal_scatter",
         list(zip(L, xs, ys)), ["method", "AL_q", "RCAL"])

    # 2/3. stacked decompositions
    for name, top, tl in (("acceptance_decomposition_stacked", "SAL",
                           "AL_q = RCAL + SAL"),
                          ("reference_decomposition_stacked", "LAL",
                           "AL_0 = RCAL + LAL")):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        rc = [mt[k]["RCAL"] for k in labs]
        tp = [mt[k][top] for k in labs]
        ax.bar(L, rc, label="RCAL", color="tab:blue")
        ax.bar(L, tp, bottom=rc, label=top, color="tab:red")
        ax.set_ylabel("accepted proposal tokens / cycle")
        ax.set_title(tl)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend()
        save(fig, pdir, name, list(zip(L, rc, tp)),
             ["method", "RCAL", top])

    # 4/5. precision-recall + AFS
    fig, ax = plt.subplots(figsize=(6, 5))
    pa = [mt[k]["P_A"] for k in labs]
    ra = [mt[k]["R_A"] for k in labs]
    ax.scatter(ra, pa, c="tab:blue")
    for x, y, l in zip(ra, pa, L):
        ax.annotate(l, (x, y), fontsize=7,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("acceptance recall R_A")
    ax.set_ylabel("acceptance precision P_A")
    ax.set_title("acceptance precision vs recall")
    save(fig, pdir, "acceptance_precision_recall",
         list(zip(L, ra, pa)), ["method", "R_A", "P_A"])

    fig, ax = plt.subplots(figsize=(8, 4))
    af = [mt[k]["AFS"] for k in labs]
    ax.bar(L, af, color="tab:blue")
    for th in (0.90, 0.95, 0.98):
        ax.axhline(th, ls="--", lw=0.7, c="tab:red")
    ax.set_ylabel("AFS"); ax.set_ylim(0, 1.05)
    ax.set_title("acceptance fidelity score "
                 "(pre-registered thresholds 0.90/0.95/0.98)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    save(fig, pdir, "acceptance_fidelity_score",
         list(zip(L, af)), ["method", "AFS"])

    # 6/7. depth survival + spurious/lost
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for k in labs:
        ds = mt[k]["depth_survival"]
        dd = sorted(int(x) for x in ds)
        ax.plot(dd, [ds[str(x)]["S_RC"] for x in dd], marker="o",
                ms=3, label=method_label(k))
        rows += [(method_label(k), x, ds[str(x)]["S_q"],
                  ds[str(x)]["S_0"], ds[str(x)]["S_RC"],
                  ds[str(x)]["spurious"], ds[str(x)]["lost"])
                 for x in dd]
    ax.set_xlabel("depth d"); ax.set_ylabel("S_RC(d)")
    ax.set_title("reference-consistent depth survival")
    ax.legend(fontsize=6)
    save(fig, pdir, "rcal_depth_survival", rows,
         ["method", "depth", "S_q", "S_0", "S_RC", "spurious", "lost"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for k in labs:
        ds = mt[k]["depth_survival"]
        dd = sorted(int(x) for x in ds)
        axes[0].plot(dd, [ds[str(x)]["spurious"] for x in dd],
                     marker="o", ms=3, label=method_label(k))
        axes[1].plot(dd, [ds[str(x)]["lost"] for x in dd],
                     marker="o", ms=3, label=method_label(k))
    axes[0].set_title("spurious acceptance by depth (S_q - S_RC)")
    axes[1].set_title("lost acceptance by depth (S_0 - S_RC)")
    for a in axes:
        a.set_xlabel("depth d")
    axes[0].legend(fontsize=6)
    save(fig, pdir, "spurious_lost_depth_survival", rows,
         ["method", "depth", "S_q", "S_0", "S_RC", "spurious", "lost"])

    # 8. first branch divergence depth
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for k in labs:
        h = mt[k].get("first_divergence_depth_hist", {})
        if not h:
            continue
        dd = sorted(int(x) for x in h)
        tot = sum(h.values())
        ax.plot(dd, [h[str(x)] / tot for x in dd], marker="o", ms=3,
                label=method_label(k))
        rows += [(method_label(k), x, h[str(x)]) for x in dd]
    ax.set_xlabel("first divergence depth")
    ax.set_ylabel("fraction of divergent cycles")
    ax.set_title("first branch-divergence depth")
    ax.legend(fontsize=6)
    save(fig, pdir, "first_branch_divergence_depth", rows,
         ["method", "depth", "count"])

    # 9/11. tree-vs-seq + token agreement vs RCAL
    ts = {}
    for p in glob.glob(os.path.join(rd, "tables",
                                    "tree_seq_fidelity_*.json")):
        d = json.load(open(p))
        ts[d["tag"]] = d
    if ts:
        ks = sorted(ts)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tv = [ts[k]["tq_tree_vs_seq_agreement"] for k in ks]
        sv = [ts[k]["tq_seq_vs_t0_seq_agreement"] for k in ks]
        x = range(len(ks))
        ax.bar([i - 0.2 for i in x], tv, 0.4,
               label="Tq tree-vs-seq (shape drift)", color="tab:blue")
        ax.bar([i + 0.2 for i in x], sv, 0.4,
               label="Tq-seq vs T0-seq (quant drift)", color="tab:red")
        ax.set_xticks(list(x))
        ax.set_xticklabels([method_label(k) for k in ks], fontsize=7,
                           rotation=30)
        ax.set_ylabel("token agreement"); ax.legend(fontsize=7)
        ax.set_title("tree-vs-sequential verifier fidelity")
        save(fig, pdir, "tree_vs_sequential_rcal",
             list(zip(ks, tv, sv)),
             ["tag", "tree_vs_seq", "seq_vs_seq"])

        fig, ax = plt.subplots(figsize=(6, 5))
        rows = []
        for k in ks:
            mk = f"{k}__mtbench"
            if mk in mt:
                ag = ts[k]["t0_seq_agreement_with_deployed_traj"]
                ax.scatter(ag, mt[mk]["RCAL"], c="tab:blue")
                ax.annotate(method_label(k), (ag, mt[mk]["RCAL"]),
                            fontsize=7, textcoords="offset points",
                            xytext=(4, 3))
                rows.append((method_label(k), ag, mt[mk]["RCAL"]))
        ax.set_xlabel("T0 sequential token agreement with deployed "
                      "trajectory")
        ax.set_ylabel("RCAL")
        ax.set_title("target token agreement vs RCAL")
        save(fig, pdir, "target_token_agreement_vs_rcal", rows,
             ["method", "t0_agreement", "RCAL"])

    # 10. target PPL vs RCAL (per-target PPL; methods share targets)
    pp = os.path.join(rd, "tables", "target_ppl.json")
    if os.path.exists(pp):
        ppl = json.load(open(pp))
        fig, ax = plt.subplots(figsize=(6, 5))
        rows = []
        for k in labs:
            tgt = "fp16" if k.endswith("_T0__mtbench") or \
                "F16__" in k else "int4"
            x = ppl.get(tgt)
            if x:
                ax.scatter(x, mt[k]["RCAL"], c="tab:blue")
                ax.annotate(method_label(k), (x, mt[k]["RCAL"]),
                            fontsize=7, textcoords="offset points",
                            xytext=(4, 3))
                rows.append((method_label(k), x, mt[k]["RCAL"]))
        ax.set_xlabel("target wikitext-2 PPL")
        ax.set_ylabel("RCAL")
        ax.set_title("target PPL vs RCAL")
        save(fig, pdir, "target_ppl_vs_rcal", rows,
             ["method", "target_ppl", "RCAL"])

    # 12. delta AL vs delta RCAL from bootstrap pairs
    bp = os.path.join(rd, "stats", "rcal_bootstrap_mtbench.json")
    if os.path.exists(bp):
        boot = json.load(open(bp))["pairs"]
        fig, ax = plt.subplots(figsize=(7, 6))
        rows = []
        for pair, v in boot.items():
            if v.get("status") == "missing":
                continue
            x, y = v["delta_AL_q"], v["delta_RCAL"]
            xe = [[x - v["ci_delta_AL_q"][0]],
                  [v["ci_delta_AL_q"][1] - x]]
            ye = [[y - v["ci_delta_RCAL"][0]],
                  [v["ci_delta_RCAL"][1] - y]]
            col = "tab:red" if v["deceptive_al_gain"] else "tab:blue"
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=col,
                        ms=4, capsize=2)
            ax.annotate(pair.replace("RC_", ""), (x, y), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
            rows.append((pair, x, y, *v["ci_delta_AL_q"],
                         *v["ci_delta_RCAL"],
                         v["deceptive_al_gain"]))
        ax.axhline(0, lw=0.6, c="k"); ax.axvline(0, lw=0.6, c="k")
        ax.plot([-2, 4], [-2, 4], "k--", lw=0.6)
        ax.set_xlabel("delta AL_q (B - A)")
        ax.set_ylabel("delta RCAL (B - A)")
        ax.set_title("paired deltas: AL vs RCAL "
                     "(red = deceptive AL gain)")
        save(fig, pdir, "delta_al_vs_delta_rcal", rows,
             ["pair", "dAL_q", "dRCAL", "dAL_lo", "dAL_hi",
              "dRCAL_lo", "dRCAL_hi", "deceptive"])
    print("[plot] rcal figures done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
