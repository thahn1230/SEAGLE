#!/usr/bin/env python
"""VSQ study figures.

Reads tables/shards/stats under the VSQ run dir (RD) plus two tables from the
FIDI run dir, and writes FIG_A..FIG_H as PDF+PNG (150 dpi) into RD/figs,
together with a FIG_INDEX.md caption file.

CPU-only, matplotlib-only, standalone (no repo imports). Does not modify any
existing file; it only creates RD/figs/* outputs.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
RD = "/home/thahn1230/dflash_workspace/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
FIDI = "/home/thahn1230/dflash_workspace/dflash/runs/dflash_full_interface_distribution_intervention_20260811_074001"
FIGS = os.path.join(RD, "figs")
os.makedirs(FIGS, exist_ok=True)

# ----------------------------------------------------------------------------
# Style: colorblind-safe palette (validated categorical hue order, light mode)
# ----------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
GREY = "#8a8a86"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

captions = []  # (figure name, caption text) collected for FIG_INDEX.md
notes = []     # data-availability notes


def save(fig, name, caption):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    captions.append((name, caption))
    print(f"saved {name}.pdf/.png")


def bar_labels(ax, bars, fmt="{:.2f}", log=False, fontsize=7.5, rotation=0):
    """Value labels above bars, in text ink (never the series color)."""
    for b in bars:
        h = b.get_height()
        y = h * 1.06 if log else h
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, y),
                    ha="center", va="bottom", fontsize=fontsize,
                    rotation=rotation, color=INK,
                    xytext=(0, 1), textcoords="offset points")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pooled_tau(shard_path):
    """Cycle-pooled tau = sum(all taus)/count(all taus) over every row.

    Returns None when the file is missing, empty, or header-only.
    """
    if not os.path.exists(shard_path):
        return None
    tot, n = 0, 0
    with open(shard_path, newline="") as f:
        for row in csv.DictReader(f):
            taus = row.get("taus", "")
            if not taus:
                continue
            vals = [int(x) for x in taus.split(";") if x != ""]
            tot += sum(vals)
            n += len(vals)
    if n == 0:
        return None
    return tot / n


# ============================================================================
# FIG_A_tppl : target W4A4 PPL ladder, grouped by seed, log y
# ============================================================================
def fig_a():
    rows = read_csv(os.path.join(RD, "tables", "target_ppl.csv"))
    # The CSV contains duplicated append rows from re-runs:
    # dedupe by (seed, arm), keeping the LAST occurrence.
    dedup = {}
    for r in rows:
        dedup[(r["seed"], r["arm"])] = r
    arms = ["T-PPL0_fp16", "T-PPL1_rtn_norot", "T-PPL2_hadamard",
            "T-PPL3_R1only", "T-PPL4_R1R2"]
    labels = ["fp16", "RTN\n(w4a4 norot)", "Hadamard", "R1-only", "R1+R2"]
    seeds = ["s0", "s1", "s2"]
    seed_colors = [BLUE, ORANGE, AQUA]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.26
    for j, (seed, c) in enumerate(zip(seeds, seed_colors)):
        xs, ys = [], []
        for i, arm in enumerate(arms):
            r = dedup.get((seed, arm))
            if r is None:
                continue
            xs.append(i + (j - 1) * w)
            ys.append(float(r["wikitext2_ppl"]))
        bars = ax.bar(xs, ys, width=w - 0.03, color=c, label=seed, zorder=3)
        bar_labels(ax, bars, fmt="{:.2f}", log=True, fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("target quantization arm")
    ax.set_ylabel("WikiText-2 perplexity (log scale)")
    ax.set_title("Target W4A4 PPL ladder (Llama-3.1, seeds s0/s1/s2)")
    ax.grid(axis="y", zorder=0)
    ax.legend(title="seed", ncol=3, loc="upper right")
    save(fig, "FIG_A_tppl",
         "Target-model WikiText-2 perplexity for the W4A4 quantization ladder "
         "(fp16 reference, RTN w4a4 without rotation, Hadamard, R1-only, "
         "R1+R2), three rotation seeds per arm; y is log-scaled because RTN "
         "(~190 PPL) dwarfs the rotated arms (~8.8-10.9). Rows deduplicated "
         "by (seed, arm) keeping the last append. "
         "Source: tables/target_ppl.csv.")


# ============================================================================
# FIG_B_draft_ladder : draft block CE per arm
# ============================================================================
def fig_b():
    rows = read_csv(os.path.join(RD, "tables", "draft_quality.csv"))
    order = ["D-P0_fp16", "D-P1_w4a4_norot", "D-P2_w4a4_random",
             "D-P3_w4a4_R1D", "D-P4_w4a4_R1D_R2D"]
    labels = ["fp16", "RTN\n(norot)", "random\nrotation", "R1D", "R1D+R2D"]
    by_arm = {r["arm"]: float(r["block_ce"]) for r in rows}
    vals = [by_arm[a] for a in order]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = [GREY, RED, BLUE, BLUE, BLUE]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62, zorder=3)
    bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("draft quantization arm")
    ax.set_ylabel("block cross-entropy")
    ax.set_title("Draft ladder: block CE under W4A4")
    ax.grid(axis="y", zorder=0)
    ax.legend(handles=[Patch(color=GREY, label="fp16 reference"),
                       Patch(color=RED, label="no rotation"),
                       Patch(color=BLUE, label="rotated W4A4")],
              loc="upper right", fontsize=8)
    save(fig, "FIG_B_draft_ladder",
         "Draft-model block cross-entropy per quantization arm: fp16 "
         "reference, W4A4 RTN without rotation, random rotation, R1D, and "
         "R1D+R2D. Rotations recover part of the RTN gap (7.23 to 4.47) but "
         "remain far from the fp16 draft (2.41). "
         "Source: tables/draft_quality.csv.")


# ============================================================================
# FIG_C_qat : (left) pilot LR sweep, (right) main runs init->best dumbbells
# ============================================================================
def fig_c():
    def load(name):
        with open(os.path.join(RD, "rotations", "draft",
                               name + ".pt.summary.json")) as f:
            return json.load(f)

    pilots = {t: load("QATpilot_" + t)
              for t in ["3e-3", "1e-2", "3e-2", "1e-1", "1e-1b", "3e-1", "1e0"]}
    mains = [("Q2 @ 3e-1", load("QAT_Q2")),
             ("Q3 @ 3e-1", load("QAT_Q3")),
             ("Q5 @ 3e-1", load("QAT_Q5")),
             ("Q5 @ 1e-1", load("QAT_Q5_lr1e-1")),
             ("Q5 @ 3e-2", load("QAT_Q5_lr3e-2"))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.7))

    # ---- left: pilot best_val_ce vs lr (log x) --------------------------
    init = pilots["3e-3"]["init_val_ce"]
    axL.axhline(init, ls="--", lw=1, color=GREY, zorder=1)
    axL.annotate(f"init CE = {init:.3f}", (0.0032, init), fontsize=7.5,
                 color=INK2, va="bottom")
    moved_x, moved_y = [], []
    stuck_x, stuck_y = [], []
    for tag, d in pilots.items():
        if tag == "1e-1b":
            continue  # plotted separately as a second marker at lr=1e-1
        if d["best_step"] == 0:  # never improved -> at init level
            stuck_x.append(d["lr"]); stuck_y.append(d["best_val_ce"])
        else:
            moved_x.append(d["lr"]); moved_y.append(d["best_val_ce"])
    order = sorted(range(len(moved_x)), key=lambda i: moved_x[i])
    moved_x = [moved_x[i] for i in order]; moved_y = [moved_y[i] for i in order]
    axL.plot(moved_x, moved_y, "-o", color=BLUE, ms=6, lw=1.4, zorder=3,
             label="improved (best CE)")
    axL.plot(stuck_x, stuck_y, "o", mfc="white", mec=BLUE, mew=1.4, ms=6,
             ls="none", zorder=3, label="no improvement (at init)")
    b = pilots["1e-1b"]
    axL.plot([b["lr"]], [b["best_val_ce"]], "s", color=ORANGE, ms=6, zorder=3,
             label="repeat run (1e-1b)")
    for x, y in zip(moved_x, moved_y):
        axL.annotate(f"{y:.3f}", (x, y), xytext=(0, -11),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=INK)
    axL.annotate(f"{b['best_val_ce']:.3f}", (b["lr"], b["best_val_ce"]),
                 xytext=(0, 5), textcoords="offset points", ha="center",
                 fontsize=7, color=INK)
    axL.set_xscale("log")
    axL.set_xlabel("learning rate (log scale)")
    axL.set_ylabel("best validation CE (100-step pilot)")
    axL.set_title("QAT pilot LR sweep (arm Q2)")
    axL.grid(axis="y", zorder=0)
    axL.legend(fontsize=7.5, loc="lower left")

    # ---- right: mains init -> best dumbbells ----------------------------
    for i, (label, d) in enumerate(mains):
        i0, bst = d["init_val_ce"], d["best_val_ce"]
        if bst < i0:
            axR.annotate("", (i, bst), (i, i0),
                         arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
            axR.plot([i], [i0], "o", mfc="white", mec=BLUE, mew=1.4, ms=7)
            axR.plot([i], [bst], "o", color=BLUE, ms=7)
            axR.annotate(f"{i0:.3f}", (i, i0), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=7,
                         color=INK)
            axR.annotate(f"{bst:.3f}", (i, bst), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=7,
                         color=INK)
        else:  # no movement
            axR.plot([i], [i0], "o", mfc="white", mec=GREY, mew=1.6, ms=8)
            axR.annotate(f"{i0:.3f}\n(no move)", (i, i0), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=7,
                         color=INK2)
    axR.set_xticks(range(len(mains)))
    axR.set_xticklabels([m[0] for m in mains], fontsize=8)
    axR.set_xlim(-0.5, len(mains) - 0.3)
    axR.set_ylim(1.8, 7.8)
    axR.set_xlabel("main QAT run (arm @ learning rate)")
    axR.set_ylabel("validation CE")
    axR.set_title("Main QAT runs: init → best CE")
    axR.grid(axis="y", zorder=0)
    axR.legend(handles=[
        Line2D([], [], marker="o", mfc="white", mec=BLUE, ls="none", ms=7,
               label="init CE"),
        Line2D([], [], marker="o", color=BLUE, ls="none", ms=7,
               label="best CE"),
        Line2D([], [], marker="o", mfc="white", mec=GREY, mew=1.6, ls="none",
               ms=8, label="no improvement")],
        fontsize=7.5, loc="upper right")

    save(fig, "FIG_C_qat",
         "Draft-rotation QAT. Left: 100-step pilot LR sweep on arm Q2 - "
         "best validation CE vs learning rate (log x); open markers sit at "
         "the init CE level for runs that never improved (3e-3, 1e-2, 1e0); "
         "the orange square is the repeat run 1e-1b at lr=1e-1. Right: main "
         "runs as init->best dumbbells - Q2 and Q3 improve at lr 3e-1, Q5 "
         "does not move at 3e-1 or 1e-1 and only improves at 3e-2 "
         "(2.644->2.389). Source: rotations/draft/QAT*.pt.summary.json.")


# ============================================================================
# FIG_D_al_ladder : final 4-dataset AL ladder
# ============================================================================
def fig_d():
    rows = read_csv(os.path.join(FIDI, "tables", "final_4dataset_al.csv"))
    ds = ["mtbench", "gsm8k", "humaneval", "sharegpt"]
    ds_colors = [BLUE, ORANGE, AQUA, MAGENTA]
    kept, m2 = [], None
    for r in rows:
        if r["tag"] == "M2_vsqraw":
            m2 = r
            continue
        kept.append(r)
    xlabels = [f"{r['tag']}\n{r['method']}" if len(r["method"]) <= 26
               else f"{r['tag']}" for r in kept]
    short = {"M0_fp16": "FP16", "M1_rtn": "naive\nW4A4 RTN",
             "M3_vsq": "vanilla\nSpinQuant", "M5_rconly": "+ R_C",
             "M5_p2rc": "+ P2 + R_C", "M6_q5bp2": "Final\n(+QAT Q5)"}
    xlabels = [f"{short.get(r['tag'], r['tag'])}\n[{r['tag']}]" for r in kept]

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    w = 0.19
    for j, (d, c) in enumerate(zip(ds, ds_colors)):
        xs = [i + (j - 1.5) * w for i in range(len(kept))]
        ys = [float(r[d]) for r in kept]
        bars = ax.bar(xs, ys, width=w - 0.02, color=c, label=d, zorder=3)
        bar_labels(ax, bars, fmt="{:.2f}", fontsize=6.3, rotation=90)
    means = [float(r["mean_4ds"]) for r in kept]
    ax.plot(range(len(kept)), means, "-D", color=INK, ms=5, lw=1.2, zorder=4,
            label="mean_4ds")
    for i, m in enumerate(means):
        ax.annotate(f"{m:.2f}", (i, m), xytext=(10, 4),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=INK, fontweight="bold")
    ax.set_xticks(range(len(kept)))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylim(0, max(float(r[d]) for r in kept for d in ds) * 1.28)
    ax.set_xlabel("method")
    ax.set_ylabel("accepted length (pooled tau)")
    ax.set_title("AL ladder across 4 datasets")
    ax.grid(axis="y", zorder=0)
    ax.legend(ncol=5, loc="upper left", fontsize=8)
    if m2 is not None:
        ax.annotate("excluded: M2_vsqraw (interface-broken, mtbench only) "
                    f"tau={float(m2['mtbench']):.2f}",
                    xy=(0.99, 0.84), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color=INK2, style="italic")
    save(fig, "FIG_D_al_ladder",
         "Accepted-length (AL) ladder on mtbench/gsm8k/humaneval/sharegpt "
         "with the 4-dataset mean overlaid (black diamonds): FP16, naive "
         "W4A4 RTN, vanilla SpinQuant (M3), +R_C (M5_rconly), +P2+R_C "
         "(M5_p2rc), and the final system with QAT draft weights "
         "(M6_q5bp2). The interface-broken M2_vsqraw row (mtbench-only, "
         "tau=1.00) is excluded and noted as an annotation. "
         "Source: FIDI tables/final_4dataset_al.csv.")


# ============================================================================
# FIG_E_validation : gsm8kvalid pooled tau for all existing V_* arms
# ============================================================================
V_TAGS = ["V_M3", "V_M4a_p2", "V_M4b_mp3", "V_M5_rconly", "V_M5_rc",
          "V_G1", "V_G1_rc", "V_M3id", "V_Q2", "V_Q3", "V_Q5", "V_Q5b",
          "V_Q5bp2", "V_restq", "V_restk", "V_restv", "V_resto",
          "V_onlyq", "V_onlyk", "V_onlyv", "V_onlyo", "V_I7m3", "V_I7m5"]


def v_family(tag):
    if tag.startswith("V_I7"):
        return ("I7 interface", RED)
    if tag.startswith("V_Q"):
        return ("Q-family (QAT)", GREEN)
    if tag.startswith(("V_rest", "V_only")):
        return ("component arms", GREY)
    if tag.startswith("V_G1"):
        return ("G1 gate", ORANGE)
    return ("M-family", BLUE)


def fig_e():
    taus, skipped, nrows = {}, [], {}
    for t in V_TAGS:
        p = os.path.join(RD, "shards", f"al__{t}__w4a4__gsm8kvalid.csv")
        v = pooled_tau(p)
        if v is None:
            skipped.append(t + (" (missing)" if not os.path.exists(p)
                                else " (empty)"))
        else:
            taus[t] = v
            nrows[t] = len(read_csv(p))
    if skipped:
        notes.append("FIG_E skipped V_* arms (no shard data yet): "
                     + ", ".join(skipped))
    full = max(nrows.values())
    partial = [f"{t} ({nrows[t]}/{full} prompts)" for t in sorted(nrows)
               if nrows[t] < full]
    if partial:
        notes.append("FIG_E partial V_* shards (jobs still appending; "
                     "pooled tau computed over available rows): "
                     + ", ".join(partial))
    order = sorted(taus, key=lambda t: taus[t])

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(order) + 1.6))
    ys = range(len(order))
    cols = [v_family(t)[1] for t in order]
    bars = ax.barh(list(ys), [taus[t] for t in order], color=cols,
                   height=0.62, zorder=3)
    for b, t in zip(bars, order):
        star = "*" if nrows[t] < max(nrows.values()) else ""
        ax.annotate(f"{taus[t]:.3f}{star}", (b.get_width(), b.get_y()
                    + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK)
    for ref, ls in (("V_M3", "--"), ("V_M5_rc", ":")):
        if ref in taus:
            ax.axvline(taus[ref], ls=ls, lw=1.1, color=INK2, zorder=2)
            ax.annotate(f"{ref} = {taus[ref]:.3f}", (taus[ref], len(order) - .2),
                        rotation=90, va="top", ha="right", fontsize=7,
                        color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("pooled tau (gsm8kvalid)")
    ax.set_ylabel("validation arm")
    ax.set_title("Validation arms: cycle-pooled accepted length")
    ax.set_xlim(0, max(taus.values()) * 1.14)
    ax.grid(axis="x", zorder=0)
    fams, seen = [], set()
    for t in order:
        name, c = v_family(t)
        if name not in seen:
            seen.add(name)
            fams.append(Patch(color=c, label=name))
    ax.legend(handles=fams, loc="lower right", fontsize=7.5)
    extra = ""
    if skipped:
        extra += " Skipped (no data): " + ", ".join(skipped) + "."
    if partial:
        extra += " Partial shards: " + ", ".join(partial) + "."
    save(fig, "FIG_E_validation",
         "Cycle-pooled accepted length on the gsm8kvalid split for every "
         "validation arm with shard data, sorted ascending and colored by "
         "family (M-family blue, Q-family green, component arms grey, G1 "
         "gate orange, I7 interface red). Dashed/dotted reference lines "
         "mark V_M3 and V_M5_rc. Values are a snapshot; validation jobs "
         "were still appending shards at render time. "
         "Source: shards/al__V_*__w4a4__gsm8kvalid.csv." + extra)


# ============================================================================
# FIG_F_mechanism : H_t kurtosis (log y) and A4 NMSE, counterfactual+deployed
# ============================================================================
def fig_f():
    mech = {r["tensor"]: r
            for r in read_csv(os.path.join(RD, "tables",
                                           "mechanism_stats.csv"))}
    htk = {(r["config"], r["tensor"]): float(r["kurtosis"])
           for r in read_csv(os.path.join(FIDI, "tables", "ht_stats.csv"))
           if r["dataset"] == "gsm8k"}
    htn = {(r["config"], r["tensor"]): float(r["nmse_mean"])
           for r in read_csv(os.path.join(FIDI, "tables",
                                          "ht_qparam_stats.csv"))
           if r["dataset"] == "gsm8k"}

    cf_labels = ["M1 naive", "M3 vsq", "M5 +R_C"]
    cf_tensors = ["M1_naiveHt", "M3_vsqHt", "M5_rcHt"]
    dep_labels = ["R1\nHt_dep", "R3\nHt_dep", "R3\nHt_rc_dep"]
    dep_keys = [("R1", "S3_Ht_dep"), ("R3", "S3_Ht_dep"),
                ("R3", "S3_Ht_rc_dep")]

    cf_kurt = [float(mech[t]["kurtosis"]) for t in cf_tensors]
    dep_kurt = [htk[k] for k in dep_keys]
    cf_nmse = [float(mech[t]["a4_nmse"]) for t in cf_tensors]
    dep_nmse = [htn[k] for k in dep_keys]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    xs_cf = [0, 1, 2]
    xs_dep = [3.6, 4.6, 5.6]
    rc_mask_cf = [False, False, True]
    rc_mask_dep = [False, False, True]

    def panel(ax, vals_cf, vals_dep, fmt, log):
        c_cf = [AQUA if rc else BLUE for rc in rc_mask_cf]
        c_dep = [AQUA if rc else VIOLET for rc in rc_mask_dep]
        b1 = ax.bar(xs_cf, vals_cf, color=c_cf, width=0.7, zorder=3)
        b2 = ax.bar(xs_dep, vals_dep, color=c_dep, width=0.7, zorder=3)
        bar_labels(ax, list(b1) + list(b2), fmt=fmt, log=log, fontsize=7.5)
        ax.set_xticks(xs_cf + xs_dep)
        ax.set_xticklabels(cf_labels + dep_labels, fontsize=7.5)
        ax.grid(axis="y", zorder=0)
        ax.annotate("h-cache counterfactual", (1.0, 1.015),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8, color=INK2)
        ax.annotate("deployed forward (gsm8k)", (4.6, 1.015),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=8, color=INK2)

    panel(axL, cf_kurt, dep_kurt, "{:.1f}", log=True)
    axL.set_yscale("log")
    axL.set_ylim(1, max(cf_kurt + dep_kurt) * 4)
    axL.set_ylabel("H_t kurtosis (log scale)")
    axL.set_xlabel("arm / tensor")
    panel(axR, cf_nmse, dep_nmse, "{:.3f}", log=False)
    axR.set_ylim(0, max(cf_nmse + dep_nmse) * 1.22)
    axR.set_ylabel("A4 NMSE")
    axR.set_xlabel("arm / tensor")
    fig.suptitle("Mechanism: R_C Gaussianizes H_t and fixes A4 quantization",
                 fontsize=10, y=1.0)
    axL.legend(handles=[Patch(color=BLUE, label="no R_C (counterfactual)"),
                        Patch(color=VIOLET, label="no R_C (deployed)"),
                        Patch(color=AQUA, label="with R_C")],
               fontsize=7, loc="upper left")
    save(fig, "FIG_F_mechanism",
         "Why R_C works. Left: H_t kurtosis (log y) in the h-cache "
         "counterfactual (M1 69.5 / M3 69.4 / M5+R_C 3.0; "
         "tables/mechanism_stats.csv) and in the deployed forward on gsm8k "
         "(R1 Ht_dep 180.5, R3 Ht_dep 195.0, R3 Ht_rc_dep 2.9; FIDI "
         "tables/ht_stats.csv). Right: A4 NMSE for the same arms "
         "(mechanism_stats a4_nmse; FIDI ht_qparam_stats nmse_mean). R_C "
         "collapses the heavy tails by ~20-70x in kurtosis and cuts A4 NMSE "
         "by an order of magnitude in both the counterfactual and the "
         "deployed forward.")


# ============================================================================
# FIG_G_ctxbits : ctx-precision ablation, mtbench pooled tau
# ============================================================================
def fig_g():
    arms = [  # (shard basename, exact-tag label)
        ("al__M3_vsq__w4a4__mtbench.csv", "M3_vsq\n(ctx A4, no R_C)"),
        ("al__HP2_ctxA8_noRC__w4a4__mtbench.csv", "HP2_ctxA8_noRC\n(ctx A8, no R_C)"),
        ("al__HP1_ctxFP16__w4a4__mtbench.csv", "HP1_ctxFP16\n(ctx FP16)"),
        ("al__M5_rconly__w4a4__mtbench.csv", "M5_rconly\n(ctx A4 + R_C)"),
        ("al__HP0_ctxA8__w4a4__mtbench.csv", "HP0_ctxA8\n(ctx A8)"),
    ]
    vals = []
    for fn, lab in arms:
        v = pooled_tau(os.path.join(RD, "shards", fn))
        vals.append((lab, v))
    vals = [v for v in vals if v[1] is not None]
    vals.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    cols = [AQUA if "R_C" in l and "no R_C" not in l else BLUE
            for l, _ in vals]
    bars = ax.bar(range(len(vals)), [v for _, v in vals], color=cols,
                  width=0.6, zorder=3)
    bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l for l, _ in vals], fontsize=7.5)
    ax.set_xlabel("arm (exact tag, context precision)")
    ax.set_ylabel("pooled tau (mtbench)")
    ax.set_title("Context-precision ablation")
    ax.set_ylim(0, max(v for _, v in vals) * 1.25)
    ax.grid(axis="y", zorder=0)
    ax.legend(handles=[Patch(color=BLUE, label="no R_C"),
                       Patch(color=AQUA, label="with R_C")],
              loc="upper left", fontsize=8)
    ax.annotate("reading: 8-bit ctx already recovers most of R_C's effect",
                xy=(0.99, 0.90), xycoords="axes fraction", ha="right",
                fontsize=8, color=INK2, style="italic")
    save(fig, "FIG_G_ctxbits",
         "Context-precision ablation on mtbench (cycle-pooled tau), sorted "
         "ascending: M3_vsq (ctx A4, no R_C), HP2_ctxA8_noRC (ctx A8, no "
         "R_C), HP1_ctxFP16 (ctx FP16), M5_rconly (ctx A4 + R_C), and "
         "HP0_ctxA8. Raising context activations from 4 to 8 bits without "
         "R_C already recovers most of R_C's benefit, locating the failure "
         "in ctx quantization of heavy-tailed H_t. "
         "Source: shards/al__{M3_vsq,HP*,M5_rconly}__w4a4__mtbench.csv.")


# ============================================================================
# FIG_H_forest : bootstrap deltas with CIs for preregistered comparisons
# ============================================================================
def fig_h():
    comps = [("P1_M0vM3", "P1: M0 vs M3", BLUE),
             ("P4b_M3vM5p2rc", "P4b: M3 vs M5(P2+R_C)", ORANGE),
             ("P3_M5rcVsM5p2rc", "P3: M5rc vs M5(P2+R_C)", AQUA),
             ("P5_M5vM6", "P5: M5 vs M6", VIOLET),
             ("GAP_M0vM6", "GAP: M0 vs M6", MAGENTA)]
    dsets = ["mtbench", "gsm8k", "humaneval", "sharegpt"]
    boot = {d: json.load(open(os.path.join(RD, "stats",
                                           f"bootstrap_{d}.json")))
            for d in dsets}
    holm = json.load(open(os.path.join(RD, "stats", "holm_adjusted.json")))

    rows = []    # (y, dataset label, delta, lo, hi, color, significant)
    glabels = [] # (y, group label) drawn on their own row above each group
    y = 0.0
    missing = []
    for comp, clabel, color in comps:
        glabels.append((y, clabel))
        y += 1.0
        for d in dsets:
            key = f"{d}:{comp}"
            entry = boot[d].get(key)
            if entry is None:
                missing.append(key)
                continue
            sig = bool(holm.get(key, {}).get("significant", False))
            rows.append((y, f"{d}", entry["delta"], entry["ci"][0],
                         entry["ci"][1], color, sig))
            y += 1.0
        y += 0.7  # gap between comparison groups

    fig, ax = plt.subplots(figsize=(7.2, 0.28 * (len(rows) + len(glabels))
                                    + 2.2))
    ax.axvline(0, color=INK2, lw=1, zorder=2)
    for (yy, dlab, delta, lo, hi, color, sig) in rows:
        ax.plot([lo, hi], [yy, yy], "-", color=color, lw=1.6, zorder=3)
        if sig:
            ax.plot([delta], [yy], "o", color=color, ms=6.5, zorder=4)
        else:
            ax.plot([delta], [yy], "o", mfc="white", mec=color, mew=1.6,
                    ms=6.5, zorder=4)
        ax.annotate(f"{delta:+.2f}", (hi, yy), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7,
                    color=INK)
    for (yy, clabel) in glabels:
        ax.annotate(clabel, (0.01, yy), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=8, color=INK,
                    fontweight="bold")
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.8, y - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("delta pooled tau (b - a), 95% bootstrap CI")
    ax.set_ylabel("comparison / dataset")
    ax.set_title("Preregistered comparisons: bootstrap deltas "
                 "(Holm-adjusted significance)")
    ax.grid(axis="x", zorder=0)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=INK2, ls="none", ms=6.5,
               label="significant (Holm)"),
        Line2D([], [], marker="o", mfc="white", mec=INK2, mew=1.6, ls="none",
               ms=6.5, label="not significant")],
        loc="lower right", fontsize=7.5)
    if missing:
        notes.append("FIG_H missing bootstrap keys: " + ", ".join(missing))
    save(fig, "FIG_H_forest",
         "Forest plot of the key preregistered comparisons (P1 M0vM3, P4b "
         "M3vM5p2rc, P3 M5rcVsM5p2rc, P5 M5vM6, GAP M0vM6), one row per "
         "(comparison, dataset): delta pooled tau with 95% bootstrap CI "
         "whiskers around the zero line. Solid markers are Holm-adjusted "
         "significant; hollow markers are not (only humaneval GAP_M0vM6 is "
         "non-significant - the final system closes the FP16 gap there). "
         "Sources: stats/bootstrap_{mtbench,gsm8k,humaneval,sharegpt}.json, "
         "stats/holm_adjusted.json.")


# ============================================================================
# FIG_INDEX.md
# ============================================================================
def write_index():
    lines = ["# VSQ study - figure index", "",
             f"All figures saved as PDF+PNG (150 dpi) in `{FIGS}`.", ""]
    for name, cap in captions:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(cap)
        lines.append("")
    if notes:
        lines.append("## Data notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    with open(os.path.join(FIGS, "FIG_INDEX.md"), "w") as f:
        f.write("\n".join(lines))
    print("saved FIG_INDEX.md")


def main():
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
    fig_f()
    fig_g()
    fig_h()
    write_index()
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
