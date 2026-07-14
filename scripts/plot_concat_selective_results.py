#!/usr/bin/env python
"""Concat-selective study figures (PNG+PDF). Dataviz method: one axis, thin
marks, fixed categorical slots (validated set), ink-colored text, descriptive
labels, units+n everywhere. Regenerates tiny fp64 depth-error curves (CPU) for
the correct path and negative controls. Run after analyze + VC + Gate B."""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)

C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"

LBL = {
    "Q00_targetFP16_draftFP16_stock": "Target FP16 / Draft FP16 (stock)",
    "Q10_targetW4A4_draftFP16_archA": "Target W4A4 / Draft FP16",
    "Q11_new_targetW4A4_draftW4A4_concat": "Target W4A4 / Draft W4A4 (concat-sel.)",
    "Q11_prevB2_targetW4A4_draftW4A4": "Target W4A4 / Draft W4A4 (prev B2)",
    "CS_fp16_baseline": "Target FP16(rot) / Draft FP16 (concat-sel.)",
    "DQ_first_only_W4A4": "Draft W4A4: first preR ONLY",
    "DQ_recurrent_only_W4A4": "Draft W4A4: recurrent preR ONLY",
    "DQ_both_proj_W4A4": "Draft W4A4: both projections",
    "DQ_ar_only_W4A4": "Draft W4A4: AR head ONLY",
    "Q01_new_draft_full_W4A4": "Draft W4A4 full (concat-selective)",
    "Q01_prevB2_draft_full_W4A4": "Draft W4A4 full (prev B2, rotated emb)",
    "DQ_first_only_W4A16": "Draft W4A16: first preR ONLY",
    "DQ_recurrent_only_W4A16": "Draft W4A16: recurrent preR ONLY",
    "DQ_embed_only_W4A16": "Draft W4A16: embedding table ONLY",
}


def style(ax, axis="x"):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[csplot] {name}")


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def tiny_depth_curves():
    """CPU fp64: depth-error curves for correct path + NCs on the real tiny
    decoder (regenerated deterministically)."""
    import torch
    import torch.nn as nn
    from b2_common import rel_l2, tiny_setup
    from eagle_spinquant import rotation_aware as ra
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveProjection, PostProjectionR1,
        build_concat_selective_weights)
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=5)
    m.load_state_dict(sd)
    ref, h = [], h_t
    for k in ids:
        h = m(h, input_ids=k); ref.append(h)
    conv, _ = ra.convert_draft_state(sd, R1, gamma, mode="r1")
    new_sd = {k: v.clone() for k, v in sd.items()}
    for k, v in conv.items():
        if k.startswith("layers.0."):
            new_sd[k] = v
    W_first, W_rec, bias = build_concat_selective_weights(sd, R1, gamma)

    def lin(Wt):
        l = nn.Linear(Wt.shape[1], Wt.shape[0], bias=True)
        l.weight.data = Wt.double(); l.bias.data = bias.double()
        return l.double()

    def run(selects, nc=None, Wr=None):
        m.load_state_dict({k: v.to(m.fc.weight.dtype) for k, v in new_sd.items()})
        split = ConcatSelectiveProjection(lin(W_first),
                                          lin(Wr if Wr is not None else W_rec),
                                          PostProjectionR1(R1).double(), nc)
        of = m.fc; m.fc = split
        outs, h = [], a_t
        try:
            for k, ids_k in enumerate(ids):
                split.select = selects[k] if isinstance(selects, list) else selects(k)
                h = m(h, input_ids=ids_k)
                outs.append(h)
        finally:
            m.fc = of
        return [rel_l2(o @ R1.t(), r) for o, r in zip(outs, ref)]

    first_then_rec = ["first"] + ["recurrent"] * 4
    _, W_rec_bad, _b = build_concat_selective_weights(sd, R1, gamma,
                                                      nc="orig_PL_recurrent")
    curves = {
        "correct (concat-selective)": run(first_then_rec),
        "NC: first reused recurrently": run(["first"] * 5),
        "NC: recurrent used for first": run(["recurrent"] * 5),
        "NC: original PL on recurrent": run(first_then_rec, Wr=W_rec_bad),
        "NC: output R omitted": run(first_then_rec, nc="no_output_R"),
    }
    with open(os.path.join(ART, "fp64_algebra", "nc_depth_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)
    return curves


def main():
    raw = pd.read_csv(os.path.join(ART, "raw_acceptance",
                                   "cs_matrix_prompt_level.csv"))
    piv = raw.pivot_table(index="prompt_id", columns="config",
                          values="mean_acceptance")
    n_p = piv.shape[0]
    eff = pd.read_csv(os.path.join(ART, "summary_tables", "cs_effects.csv"))
    gate = json.load(open(os.path.join(ART, "fp16_equivalence",
                                       "fp_equivalence.json")))

    # fig1: architecture schematic
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec=C[0]):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.2, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.3))
    box(0.02, 0.72, 0.20, 0.16, "draft embedding e\nORIGINAL basis (fp16)")
    box(0.02, 0.40, 0.20, 0.16, "FIRST: a = n·R1\n(fused target tail)")
    box(0.02, 0.10, 0.20, 0.16, "RECURRENT: r_d = h_d·R1\n(rotated AR output)")
    box(0.30, 0.56, 0.16, 0.16, "concat(e, ·)\n[embed | feature]", "#f6f6f4", INK2)
    box(0.52, 0.66, 0.22, 0.18,
        "projection_first_preR\n[W_e | W_h·D_γ·R1], b", "#fdf3e2", C[2])
    box(0.52, 0.30, 0.22, 0.18,
        "projection_recurrent_preR\n[W_e | W_h·R1], b", "#fdf3e2", C[2])
    box(0.80, 0.48, 0.17, 0.18, "post-projection\nR1 (explicit)", "#e9f6ef", C[1])
    arrow(0.22, 0.80, 0.30, 0.68); arrow(0.22, 0.48, 0.30, 0.62)
    arrow(0.22, 0.18, 0.30, 0.58)
    arrow(0.46, 0.66, 0.52, 0.74); arrow(0.46, 0.60, 0.52, 0.40)
    arrow(0.74, 0.74, 0.80, 0.60); arrow(0.74, 0.38, 0.80, 0.52)
    ax.text(0.985, 0.57, "→ rotated AR head\n(R1-conjugated),\nhead = W_lm·R1",
            fontsize=8, color=INK, va="center")
    ax.set_title("Concat-selective rotation architecture: original embedding, "
                 "hidden-only folds, explicit post-PL R1", fontsize=10,
                 color=INK, loc="left")
    save(fig, "fig01_architecture")

    # fig2+3: explicit vs folded equivalence (gate + tiny chain)
    tc = json.load(open(os.path.join(ART, "fp64_algebra", "tiny_chain.json")))
    per = {r["config"]: r for r in gate["configs"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    names = ["F2_concat_selective_explicit", "F3_concat_selective_folded",
             "F1_prev_B2_rotated_embedding"]
    pretty = ["explicit", "folded preR", "prev B2"]
    b = axes[0].bar(pretty, [per[n]["mean_acceptance"] for n in names],
                    0.55, color=[C[0], C[1], C[4]])
    for bb in b:
        axes[0].annotate(f"{bb.get_height():.4f}",
                         (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color=INK)
    axes[0].set_ylabel("accepted tokens/round (fp16, n=8×48)", fontsize=8,
                       color=INK2)
    axes[0].set_title("fp16 e2e: all three identical (exact-match 1.0)",
                      fontsize=9, color=INK, loc="left")
    style(axes[0], "y")
    axes[1].semilogy([r["depth"] for r in tc],
                     [max(r["feature_rel_l2"], 1e-12) for r in tc], "-o",
                     color=C[0], label="feature rel-L2 (fp64 tiny chain)")
    axes[1].set_xlabel("draft depth"); axes[1].set_ylim(1e-12, 1e-3)
    axes[1].set_title("folded-vs-original algebra error by depth", fontsize=9,
                      color=INK, loc="left")
    axes[1].legend(fontsize=7.5, frameon=False)
    style(axes[1], "y")
    save(fig, "fig02_03_explicit_vs_folded_equivalence")

    # fig5+6: FP error by depth + NC curves (tiny fp64)
    curves = tiny_depth_curves()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for k, (name, ys) in enumerate(curves.items()):
        ax.semilogy(range(1, 6), [max(y, 1e-12) for y in ys], "-o",
                    color=C[k % 6], lw=2, ms=4, label=name)
    ax.set_xlabel("draft depth", fontsize=8, color=INK2)
    ax.set_ylabel("feature rel-L2 vs original chain (fp64)", fontsize=8,
                  color=INK2)
    ax.set_title("Correct path stays at the fp32 floor; negative controls "
                 "diverge (tiny real decoder)", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax, "y")
    save(fig, "fig05_06_depth_error_and_negative_controls")

    # fig4+7+9: acceptance across configs incl. architecture comparison
    order = [c for c in LBL if c in piv.columns]
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(order) + 1.2))
    for i, cfgn in enumerate(order):
        v = piv[cfgn].dropna().values
        lo, hi = boot_ci(v)
        y = len(order) - 1 - i
        col = C[4] if "prevB2" in cfgn else C[0]
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(v.mean(), y, "o", color=col, ms=6)
        ax.annotate(f"{v.mean():.2f}", (v.mean(), y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LBL[c] for c in reversed(order)], fontsize=8, color=INK)
    ax.set_xlabel(f"mean accepted tokens/round (greedy, n={n_p} prompts, 64 tok; "
                  f"95% bootstrap CI; violet = previous rotated-embedding B2)",
                  fontsize=8, color=INK2)
    ax.set_title("Acceptance across precision combinations and architectures",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, "fig04_07_09_acceptance_matrix_and_architectures")

    # fig8: first vs recurrent quantization
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    groups = ["first preR\nONLY", "recurrent preR\nONLY", "both", "AR head\nONLY",
              "full draft"]
    w4a4 = [piv[c].mean() for c in ("DQ_first_only_W4A4", "DQ_recurrent_only_W4A4",
                                    "DQ_both_proj_W4A4", "DQ_ar_only_W4A4",
                                    "Q01_new_draft_full_W4A4")]
    w4a16 = [piv["DQ_first_only_W4A16"].mean(),
             piv["DQ_recurrent_only_W4A16"].mean(), np.nan, np.nan, np.nan]
    x = np.arange(5)
    base = piv["CS_fp16_baseline"].mean()
    ax.axhline(base, color=INK2, lw=1, ls="--")
    b1 = ax.bar(x - 0.19, w4a4, 0.34, color=C[0], label="fake W4A4")
    b2 = ax.bar(x + 0.19, w4a16, 0.34, color=C[1], label="fake W4A16 (w only)")
    for bars in (b1, b2):
        for bb in bars:
            if not np.isnan(bb.get_height()):
                ax.annotate(f"{bb.get_height():.2f}",
                            (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8, color=INK)
    ax.set_ylabel(f"accepted tokens/round (n={n_p})", fontsize=8, color=INK2)
    ax.set_title(f"Draft-component quantization (concat-selective; fp16 "
                 f"baseline {base:.2f})", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax, "y")
    save(fig, "fig08_first_vs_recurrent_quant")

    # fig10: PPL vs acceptance (this study's targets)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    pts = [("Target FP16", 6.945, piv["Q00_targetFP16_draftFP16_stock"].mean()),
           ("Target W4A4", 10.627, piv["Q10_targetW4A4_draftFP16_archA"].mean())]
    for nm, x_, y_ in pts:
        ax.plot(x_, y_, "o", color=C[0], ms=7)
        ax.annotate(nm, (x_, y_), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("WikiText-2 PPL", fontsize=8, color=INK2)
    ax.set_ylabel(f"accepted tokens/round (draft FP16, n={n_p})", fontsize=8,
                  color=INK2)
    ax.set_title("Target degradation vs acceptance", fontsize=9.5, color=INK,
                 loc="left")
    style(ax, "both" if False else "y")
    save(fig, "fig10_ppl_vs_acceptance")

    # fig11: real-kernel e2e (if Gate B done)
    p = os.path.join(ART, "real_w4a4_dispatch", "real_e2e_summary.json")
    if os.path.isfile(p):
        rs = json.load(open(p))
        names = [r["config"] for r in rs["configs"]]
        vals = [r["mean_acceptance"] for r in rs["configs"]]
        pretty = [n.replace("_target", "\ntarget ").replace("_draft", " / draft ")
                  for n in names]
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        b = ax.bar(range(len(names)), vals, 0.55, color=[C[0], C[1], C[2]])
        for bb, v in zip(b, vals):
            ax.annotate(f"{v:.2f}", (bb.get_x() + bb.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(pretty, fontsize=7, color=INK)
        ax.set_ylabel("accepted tokens/round", fontsize=8, color=INK2)
        ax.set_title("REAL packed W4A4 end-to-end (QuaRot CUTLASS in "
                     "generation; KV fp16)", fontsize=9.5, color=INK, loc="left")
        style(ax, "y")
        save(fig, "fig11_real_kernel_e2e")

    # fig12: verifier consistency (if VC done)
    p = os.path.join(ART, "verifier_consistency", "target_logit_consistency.csv")
    if os.path.isfile(p):
        vc = pd.read_csv(p)
        g = vc.groupby(["target", "path"]).agg(
            top1=("top1_agree", "mean"), cert=("certified_frac", "mean"),
            rel=("rel_l2", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        xs = np.arange(len(g))
        b = ax.bar(xs, g["top1"], 0.55,
                   color=[C[0] if t == "fp16" else C[2] for t in g["target"]])
        for bb, (t1, ce) in zip(b, zip(g["top1"], g["cert"])):
            ax.annotate(f"top1 {t1:.3f}\ncert {ce:.2f}",
                        (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{t}\n{p_}" for t, p_ in zip(g["target"], g["path"])],
                           fontsize=7.5, color=INK)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("top-1 agreement vs full-sequence forward", fontsize=8,
                      color=INK2)
        ax.set_title("Verifier execution-shape consistency (blue=fp16 target, "
                     "yellow=fake-W4A4 target; cert = ||Δ||∞<margin/2 fraction)",
                     fontsize=9, color=INK, loc="left")
        style(ax, "y")
        save(fig, "fig12_verifier_consistency")

    print(f"[csplot] DONE -> {FIG}")


if __name__ == "__main__":
    sys.exit(main())
