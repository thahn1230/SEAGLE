#!/usr/bin/env python
"""Figures for the bitwidth-AL component causality study (CPU-only).

Reads:
  --run-dir           matrix run dir (shards/ + analysis/ from analyze_*.py)
  fixed artifacts     artifacts/bitwidth_al_component_causality/
                        fixed_tree/{rounds.csv, categories_summary.json}
                        verifier_consistency/target_logit_consistency.csv
                        equivalence_forensics/summary.json  (optional)

Writes PNGs to artifacts/bitwidth_al_component_causality/figures/<tag>__*.png
where <tag> is the run-dir basename (pilot and final coexist).
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality")
FIG = os.path.join(ART, "figures")
ROWS_T = ["T16", "T8", "T4"]
COLS_D = ["D16", "D8", "D4"]


def rd_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, tag, name):
    p = os.path.join(FIG, f"{tag}__{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {p}", flush=True)


def fig_matrix_heatmap(an, tag):
    m = np.full((3, 3), np.nan)
    for r in an:
        t, d = r["cell"].split("_")
        m[ROWS_T.index(t), COLS_D.index(d)] = float(r["al_mean"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(m, cmap="viridis", vmin=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center",
                    color="white" if m[i, j] < np.nanmax(m) * 0.75 else "black")
    ax.set_xticks(range(3), COLS_D)
    ax.set_yticks(range(3), ROWS_T)
    ax.set_xlabel("draft precision")
    ax.set_ylabel("target precision")
    ax.set_title("Mean accepted length (tau), Target x Draft")
    fig.colorbar(im, shrink=0.85)
    save(fig, tag, "01_matrix_heatmap")


def fig_contrast_forest(cons, tag):
    cons = [c for c in cons if c["contrast"] != "vs_stock"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(cons) + 1.6))
    ys = np.arange(len(cons))[::-1]
    for y, c in zip(ys, cons):
        d, lo, hi = float(c["diff"]), float(c["ci_lo"]), float(c["ci_hi"])
        col = {"different": "#c0392b", "practically_equivalent": "#27ae60",
               "inconclusive": "#7f8c8d"}[c["verdict"]]
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.plot(d, y, "o", color=col, ms=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys, [f'{c["contrast"]}: {c["cell"]} - {c["baseline"]}'
                       for c in cons], fontsize=8)
    ax.set_xlabel("paired AL difference (95% cluster-bootstrap CI)")
    ax.set_title("Target/draft precision contrasts")
    save(fig, tag, "02_contrast_forest")


def fig_interaction(inter, tag):
    fig, ax = plt.subplots(figsize=(6, 4))
    cells = [r["cell"] for r in inter]
    obs = [float(r["observed"]) for r in inter]
    pred = [float(r["additive_pred"]) for r in inter]
    x = np.arange(len(cells))
    ax.bar(x - 0.2, obs, 0.4, label="observed", color="#2c6fbb")
    ax.bar(x + 0.2, pred, 0.4, label="additive prediction", color="#b8cbe0")
    ax.set_xticks(x, cells, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AL")
    ax.set_title("Non-separability: observed vs additive row+column model")
    ax.legend()
    save(fig, tag, "03_interaction")


def fig_depth_hists(dh, tag):
    cells = sorted({r["cell"] for r in dh})
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    pos = {f"{t}_{d}": (ROWS_T.index(t), COLS_D.index(d))
           for t in ROWS_T for d in COLS_D}
    for cell in cells:
        if cell not in pos:
            continue
        i, j = pos[cell]
        rows = [r for r in dh if r["cell"] == cell]
        depths = [int(r["accepted_depth"]) for r in rows]
        fracs = [float(r["frac"]) for r in rows]
        axes[i, j].bar(depths, fracs, color="#2c6fbb")
        axes[i, j].set_title(cell, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("accepted draft tokens/cycle")
    for axr in axes:
        axr[0].set_ylabel("fraction")
    fig.suptitle("Accepted-depth distributions per cell")
    save(fig, tag, "04_depth_hists")


def fig_grader(tag):
    p = os.path.join(ART, "fixed_tree", "categories_summary.json")
    if not os.path.exists(p):
        return
    summ = json.load(open(p))
    per = summ["per_target"]
    cats = sorted({c for t in per.values() for c in t["categories"]})
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cats))
    w = 0.38
    for k, (tprec, off) in enumerate((("W8A8", -w / 2), ("W4A4", w / 2))):
        vals = [int(per[tprec]["categories"].get(c, 0)) for c in cats]
        ax.bar(x + off, vals, w, label=f"{tprec} target")
    ax.set_xticks(x, cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rounds (of %d)" % per["W4A4"]["n_rounds"])
    ax.set_title("Fixed-tree grader: round categories vs fp16 baseline\n"
                 "(fp16 recheck flips: %d)"
                 % summ["fp16_recheck"]["n_accept_flips"])
    ax.legend()
    save(fig, tag, "05_grader_categories")

    rounds = rd_csv(os.path.join(ART, "fixed_tree", "rounds.csv"))
    if rounds:
        fig, ax = plt.subplots(figsize=(6, 4))
        for tprec, col in (("W8A8", "#27ae60"), ("W4A4", "#c0392b")):
            ds = [int(r["accept_delta"]) for r in rounds
                  if r["target"] == tprec]
            lo, hi = min(ds), max(ds)
            bins = np.arange(lo - 0.5, hi + 1.5)
            ax.hist(ds, bins=bins, alpha=0.6, label=tprec, color=col)
        ax.set_xlabel("accept-length delta vs fp16 (same tree)")
        ax.set_ylabel("rounds")
        ax.set_title("Fixed-tree re-grading: accept deltas")
        ax.legend()
        save(fig, tag, "06_grader_deltas")


def fig_vc(tag):
    rows = rd_csv(os.path.join(ART, "verifier_consistency",
                               "target_logit_consistency.csv"))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    targets = ["fp16", "fake_W8A8", "fake_W4A4"]
    for k, t in enumerate(targets):
        vals = [float(r["top1_agree"]) for r in rows if r["target"] == t]
        x = np.random.default_rng(k).uniform(-0.12, 0.12, len(vals)) + k
        ax.plot(x, vals, "o", alpha=0.65, ms=6)
        ax.plot([k - 0.25, k + 0.25], [np.mean(vals)] * 2, "k-", lw=2)
    ax.set_xticks(range(3), targets)
    ax.set_ylabel("top-1 agreement vs full-sequence logits")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title("Verifier path consistency (chunked/incremental vs full)")
    save(fig, tag, "07_verifier_consistency")


def fig_quality(tag):
    p = os.path.join(ART, "fixed_tree", "categories_summary.json")
    if not os.path.exists(p):
        return
    q = json.load(open(p))["quality"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    names = [r["target"] for r in q]
    ppl = [float(r["wikitext_ppl"]) for r in q]
    ax.bar(names, ppl, color=["#2c6fbb", "#27ae60", "#c0392b"])
    for i, v in enumerate(ppl):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("WikiText-2 PPL (token-level CE)")
    ax.set_title("Target quality by precision")
    save(fig, tag, "08_target_quality")


def fig_components(comp_dir, tag, stock_al=3.4158):
    rows = rd_csv(os.path.join(comp_dir, "analysis", "component_al.csv"))
    if not rows:
        return
    MODES = ["w8a16", "w16a8", "w8a8", "w4a16", "w16a4", "w4a4"]
    fams = ["draft_embed", "draft_head", "draft_ar", "draft_recurrent",
            "draft_first", "target_embedding", "target_lm_head",
            "target_body"]
    have = {(r["family"], r["mode"]): float(r["al_mean"]) for r in rows}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(fams))
    w = 0.13
    cmap = plt.get_cmap("viridis")
    for k, m in enumerate(MODES):
        vals = [have.get((f, m), np.nan) for f in fams]
        ax.bar(x + (k - 2.5) * w, vals, w, label=m.upper(),
               color=cmap(k / (len(MODES) - 1)))
    ax.axhline(stock_al, color="k", lw=0.8, ls="--")
    ax.text(len(fams) - 0.4, stock_al + 0.03, "stock fp16", fontsize=8)
    ax.set_xticks(x, fams, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AL (n=20)")
    ax.set_title("Single-component precision ablations (all else fp16)")
    ax.set_ylim(0, 4.0)
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    save(fig, tag, "09_component_ablations")

    # branch recovery figure
    order = ["branch_A_fullconcat_W16A4", "branch_B_branchwise_W16A4",
             "branch_C_eFP16_hA4", "branch_C_eA4_hFP16",
             "branch_A_fullconcat_W4A4", "branch_B_branchwise_W4A4"]
    vals = {r["config"]: float(r["al_mean"]) for r in rows
            if r["group"] == "branch"}
    got = [(c, vals[c]) for c in order if c in vals]
    if got:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([c.replace("branch_", "").replace("_", "\n", 1)
                for c, _ in got], [v for _, v in got], color="#2c6fbb")
        ax.axhline(stock_al, color="k", lw=0.8, ls="--")
        ax.set_ylabel("AL (n=20)")
        ax.set_title("Branchwise vs full-concat activation quantization")
        plt.setp(ax.get_xticklabels(), fontsize=8)
        save(fig, tag, "10_branchwise_recovery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--comp-dir", default=None)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rd = args.run_dir.rstrip("/")
    tag = os.path.basename(rd)
    an = rd_csv(os.path.join(rd, "analysis", "matrix_al.csv"))
    cons = rd_csv(os.path.join(rd, "analysis", "paired_contrasts.csv"))
    inter = rd_csv(os.path.join(rd, "analysis", "interaction.csv"))
    dh = rd_csv(os.path.join(rd, "analysis", "depth_hist.csv"))
    if an:
        fig_matrix_heatmap(an, tag)
    if cons:
        fig_contrast_forest(cons, tag)
    if inter:
        fig_interaction(inter, tag)
    if dh:
        fig_depth_hists(dh, tag)
    fig_grader(tag)
    fig_vc(tag)
    fig_quality(tag)
    if args.comp_dir:
        fig_components(args.comp_dir, tag)
    print("[plot] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
