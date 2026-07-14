#!/usr/bin/env python
"""B2 study figures (PNG+PDF). Style per the dataviz method: form by job, one
axis, thin marks, recessive grid, categorical slots in fixed order (validated
set: worst adjacent CVD dE 24.2), values/labels in ink colors (never series
color), descriptive config names ("Target W4A4 / Draft FP16"), units + n on
every figure. Run AFTER analyze_b2_results.py."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)

# categorical slots, fixed order (dataviz reference palette, light mode)
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"

LABELS = {
    "Q00_targetFP16_draftFP16_stock": "Target FP16 / Draft FP16 (stock)",
    "Q10_targetW4A4_draftFP16_archA": "Target W4A4 / Draft FP16 (A-explicit)",
    "Q10s_targetW4A4_draftFP16_B2split": "Target W4A4 / Draft FP16 (B2 split)",
    "TQ1_targetW4A16_draftFP16_archA": "Target W4A16 / Draft FP16",
    "TQ4_targetW4A4KV4_draftFP16_archA": "Target W4A4KV4 / Draft FP16",
    "FP02_B2split_draftFP16": "Target FP16(rot) / Draft FP16 (B2 split)",
    "DQ_first_only_W4A4": "Draft W4A4: projection_first ONLY",
    "DQ_recurrent_only_W4A4": "Draft W4A4: projection_recurrent ONLY",
    "DQ_both_proj_W4A4": "Draft W4A4: both projections",
    "DQ_ar_only_W4A4": "Draft W4A4: AR head ONLY",
    "Q01_targetFP16rot_draftW4A4_full": "Target FP16(rot) / Draft W4A4 (full)",
    "DQ_first_only_W4A16": "Draft W4A16: projection_first ONLY",
    "DQ_recurrent_only_W4A16": "Draft W4A16: projection_recurrent ONLY",
    "Q11_targetW4A4_draftW4A4_B2split": "Target W4A4 / Draft W4A4",
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[b2plot] {name}")


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    eff = pd.read_csv(os.path.join(ART, "summary_tables", "b2_effects.csv"))
    dep = pd.read_csv(os.path.join(ART, "summary_tables", "b2_depth_alphas.csv"))
    raw = pd.read_csv(os.path.join(ART, "raw_acceptance_traces",
                                   "acceptance_matrix_prompt_level.csv"))
    piv = raw.pivot_table(index="prompt_id", columns="config",
                          values="mean_acceptance")
    n_p = piv.shape[0]

    # ---- fig 1: acceptance by configuration (dot + bootstrap CI) ----------
    order = [c for c in LABELS if c in piv.columns]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(order) + 1.2))
    for i, cfg in enumerate(order):
        v = piv[cfg].dropna().values
        lo, hi = boot_ci(v)
        y = len(order) - 1 - i
        ax.plot([lo, hi], [y, y], color=C[0], lw=2, solid_capstyle="round")
        ax.plot(v.mean(), y, "o", color=C[0], ms=6)
        ax.annotate(f"{v.mean():.2f}", (v.mean(), y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[c] for c in reversed(order)], fontsize=8,
                       color=INK)
    ax.set_xlabel(f"mean accepted tokens per verification round "
                  f"(greedy, MT-bench n={n_p} prompts, 64 new tok; "
                  f"dot=mean, bar=95% bootstrap CI)", fontsize=8, color=INK2)
    ax.set_title("EAGLE-1 acceptance by configuration (B2 split study)",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, "fig1_acceptance_by_config")

    # ---- fig 2: paired deltas with equivalence band -----------------------
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(eff) + 1.4))
    eps = eff["epsilon"].max()
    ax.axvspan(-eps, eps, color=GRID, alpha=0.55, lw=0,
               label=f"practical-equivalence band (±{eps:.3f})")
    ax.axvline(0, color=INK2, lw=0.8)
    fam_col = {"target_only": C[0], "rotation_control": C[1],
               "draft_only": C[2], "both": C[4],
               "both_vs_targetonly": C[4], "both_vs_draftonly": C[4]}
    for i, r in eff.iterrows():
        y = len(eff) - 1 - i
        col = fam_col.get(r["family"], C[5])
        ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color=col, lw=2,
                solid_capstyle="round")
        ax.plot(r["paired_delta_mean"], y, "o", color=col, ms=6)
        ax.annotate(f"{r['paired_delta_mean']:+.2f}",
                    (r["paired_delta_mean"], y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(eff)))
    ax.set_yticklabels(
        [f"{LABELS.get(r['treatment'], r['treatment'])}\n   vs "
         f"{LABELS.get(r['baseline'], r['baseline'])}"
         for _, r in eff.iloc[::-1].iterrows()], fontsize=7, color=INK)
    ax.set_xlabel(f"paired Δ accepted tokens per round (95% paired bootstrap "
                  f"CI, {n_p} prompts, 10k resamples)", fontsize=8, color=INK2)
    ax.set_title("Paired acceptance effects (blue=target-only, aqua=rotation "
                 "control, yellow=draft-only, violet=both)", fontsize=9,
                 color=INK, loc="left")
    style(ax)
    save(fig, "fig2_paired_deltas_ci")

    # ---- fig 3: first vs recurrent projection quantization ----------------
    groups = ["projection_first\nONLY", "projection_recurrent\nONLY",
              "both\nprojections", "AR head\nONLY", "full draft\npath"]
    w4a4 = [piv[c].mean() for c in
            ["DQ_first_only_W4A4", "DQ_recurrent_only_W4A4",
             "DQ_both_proj_W4A4", "DQ_ar_only_W4A4",
             "Q01_targetFP16rot_draftW4A4_full"]]
    w4a16 = [piv["DQ_first_only_W4A16"].mean(),
             piv["DQ_recurrent_only_W4A16"].mean(), np.nan, np.nan, np.nan]
    base = piv["FP02_B2split_draftFP16"].mean()
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhline(base, color=INK2, lw=1, ls="--")
    ax.annotate(f"fp16 draft baseline {base:.2f}", (4.4, base),
                fontsize=7.5, color=INK2, va="bottom", ha="right")
    b1 = ax.bar(x - 0.19, w4a4, 0.34, color=C[0], label="fake W4A4 (w+a)")
    b2 = ax.bar(x + 0.19, w4a16, 0.34, color=C[1], label="fake W4A16 (w only)")
    for bars in (b1, b2):
        for b in bars:
            if not np.isnan(b.get_height()):
                ax.annotate(f"{b.get_height():.2f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8, color=INK)
    ax.set_ylabel(f"mean accepted tokens/round (n={n_p})", fontsize=8,
                  color=INK2)
    ax.set_title("Draft-component quantization ablation "
                 "(target = rotated FP16; draft embed/head fp16 isolated)",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_first_vs_recurrent_ablation")

    # ---- fig 4: acceptance-by-depth curves (two panels, <=4 lines each) ---
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    panels = [
        ("target-side", ["Q00_targetFP16_draftFP16_stock",
                         "Q10_targetW4A4_draftFP16_archA",
                         "TQ1_targetW4A16_draftFP16_archA",
                         "Q11_targetW4A4_draftW4A4_B2split"]),
        ("draft-side (target rot-FP16)", ["FP02_B2split_draftFP16",
                                          "DQ_first_only_W4A4",
                                          "DQ_recurrent_only_W4A4",
                                          "Q01_targetFP16rot_draftW4A4_full"]),
    ]
    for ax, (ttl, cfgs) in zip(axes, panels):
        for k, cfg in enumerate(cfgs):
            g = dep[dep["config"] == cfg]
            if not len(g):
                continue
            ax.plot(g["depth"], g["alpha"], "-o", color=C[k], lw=2, ms=4,
                    label=LABELS.get(cfg, cfg))
        ax.set_xlabel("draft depth d", fontsize=8, color=INK2)
        ax.set_title(ttl, fontsize=9, color=INK, loc="left")
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6.6, frameon=False, loc="upper right")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("α_d = P(accept ≥ d | ≥ d−1)\nchain-style, accepted "
                       "paths", fontsize=8, color=INK2)
    fig.suptitle(f"Acceptance by draft depth (greedy, n={n_p} prompts)",
                 fontsize=10, color=INK, x=0.12, ha="left")
    save(fig, "fig4_acceptance_by_depth")

    # ---- fig 5: FP16 gate — correct paths vs negative controls ------------
    gate = json.load(open(os.path.join(ART, "fp_equivalence.json")))
    gcfg = {r["config"]: r["mean_acceptance"] for r in gate["configs"]}
    names = ["FP00_stock", "FP01_A_explicit", "FP01b_A_folded", "FP02_B2_split",
             "FP02x_explicit_Mgamma", "FP03_single_folded", "FP04_gamma_omitted",
             "N1_naive", "N2_gamma_elementwise", "N4_wrong_orientation",
             "N5_fold_both_halves"]
    pretty = ["stock", "A explicit", "A folded", "B2 split", "explicit M_γ",
              "NC: single folded", "NC: γ omitted", "NC: naive",
              "NC: elementwise γ", "NC: wrong orientation", "NC: γ on e-block"]
    cols = [C[0]] * 5 + [C[2]] * 6
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bars = ax.bar(range(len(names)), [gcfg[n] for n in names], 0.62, color=cols)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(pretty, rotation=35, ha="right", fontsize=7.5, color=INK)
    ax.set_ylabel("mean accepted tokens/round\n(fp16, n=8 prompts, 48 tok)",
                  fontsize=8, color=INK2)
    ax.set_title("STOP GATE A: correct implementations (blue) are equivalent; "
                 "negative controls (yellow) degrade", fontsize=9.5, color=INK,
                 loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig5_gate_negative_controls")

    # ---- fig 6: target PPL vs acceptance ----------------------------------
    pts = [("FP16", 6.945, piv["Q00_targetFP16_draftFP16_stock"].mean()),
           ("W4A16", 8.938, piv["TQ1_targetW4A16_draftFP16_archA"].mean()),
           ("W4A4", 10.627, piv["Q10_targetW4A4_draftFP16_archA"].mean()),
           ("W4A4KV4", 10.939, piv["TQ4_targetW4A4KV4_draftFP16_archA"].mean())]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (nm, x_, y_) in enumerate(pts):
        ax.plot(x_, y_, "o", color=C[0], ms=7)
        ax.annotate(f"Target {nm}", (x_, y_), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("target WikiText-2 PPL (lower = better quality)", fontsize=8,
                  color=INK2)
    ax.set_ylabel(f"mean accepted tokens/round\n(draft FP16, n={n_p})",
                  fontsize=8, color=INK2)
    ax.set_title("Target degradation vs acceptance (target-only quantization)",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig6_ppl_vs_acceptance")

    # ---- fig 7: factorial interaction --------------------------------------
    q00 = piv["Q00_targetFP16_draftFP16_stock"].mean()
    q10 = piv["Q10_targetW4A4_draftFP16_archA"].mean()
    q01 = piv["Q01_targetFP16rot_draftW4A4_full"].mean()
    q11 = piv["Q11_targetW4A4_draftW4A4_B2split"].mean()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot([0, 1], [q00, q10], "-o", color=C[0], lw=2, label="Draft FP16")
    ax.plot([0, 1], [q01, q11], "-o", color=C[2], lw=2, label="Draft W4A4 (full)")
    for x_, y_ in [(0, q00), (1, q10), (0, q01), (1, q11)]:
        ax.annotate(f"{y_:.2f}", (x_, y_), xytext=(6, 3),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Target FP16", "Target W4A4"], fontsize=9, color=INK)
    ax.set_ylabel(f"mean accepted tokens/round (n={n_p})", fontsize=8,
                  color=INK2)
    inter = (q11 - q01) - (q10 - q00)
    ax.set_title(f"Target × draft quantization interaction "
                 f"(non-additivity = {inter:+.2f})", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig7_factorial_interaction")

    # ---- fig 8: real-kernel dispatch/latency (Gate B) ----------------------
    rk = pd.read_csv(os.path.join(ART, "real_w4a4_projections.csv"))
    rk4 = rk[rk["backend"] == "real_packed_W4A4"]
    if len(rk4) == 2 and "ms_per_call_real" in rk4:
        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        y = np.arange(2)
        ax.barh(y + 0.18, rk4["ms_per_call_real"], 0.32, color=C[0],
                label="real QuaRot INT4×INT4")
        ax.barh(y - 0.18, rk4["ms_per_call_fp16"], 0.32, color=C[1],
                label="fp16 GEMM")
        for yy, r in zip(y, rk4.itertuples()):
            ax.annotate(f"{r.ms_per_call_real:.2f} ms", (r.ms_per_call_real, yy + 0.18),
                        xytext=(4, -3), textcoords="offset points", fontsize=7.5,
                        color=INK)
            ax.annotate(f"{r.ms_per_call_fp16:.2f} ms", (r.ms_per_call_fp16, yy - 0.18),
                        xytext=(4, -3), textcoords="offset points", fontsize=7.5,
                        color=INK)
        ax.set_yticks(y); ax.set_yticklabels(rk4["projection"], fontsize=8,
                                             color=INK)
        ax.set_xlabel("ms per call ([64,8192]×[8192,4096], RTX4090, first call "
                      "includes one-time JIT)", fontsize=7.5, color=INK2)
        ax.set_title("Real packed W4A4 kernel dispatch (both projections, "
                     "separate scales)", fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
        style(ax)
        save(fig, "fig8_real_kernel_latency")

    print(f"[b2plot] DONE -> {FIG}")


if __name__ == "__main__":
    sys.exit(main())
