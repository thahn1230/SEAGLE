#!/usr/bin/env python
"""Aggregate all LRAS study shards into the required summary artifacts:
paired bootstrap stats (10k resamples, seed 0), per-prompt/per-cycle files,
heatmaps and study figures. CPU-only.
"""
import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOT_N, BOOT_SEED = 10000, 0
TREE_SIZE = 26


def load_rows(path):
    df = pd.read_csv(path)
    df["acceptance_list"] = df["acceptance_list"].map(json.loads)
    return df


def cycle_metrics(df):
    """Derive the full §7.1 metric set from per-prompt cycle lists."""
    taus = [t for lst in df.acceptance_list for t in lst]
    n_cycles = len(taus)
    committed = sum(taus)
    accepted_draft = committed - n_cycles
    drafted = n_cycles * (TREE_SIZE - 1)          # draft nodes per cycle
    per_prompt = df.acceptance_list.map(lambda l: float(np.mean(l)))
    depth_hist = {}
    for t in taus:
        depth_hist[t - 1] = depth_hist.get(t - 1, 0) + 1
    return dict(
        n_prompts=len(df),
        mean_tau=float(per_prompt.mean()),
        mean_accepted_draft_tokens=float(per_prompt.mean()) - 1.0,
        sd_tau=float(per_prompt.std(ddof=1)),
        se_tau=float(per_prompt.std(ddof=1) / math.sqrt(len(df))),
        total_generated_tokens=int(df.n_new_tokens.sum()),
        total_cycles=int(n_cycles),
        total_drafted_nodes=int(drafted),
        accepted_draft_tokens=int(accepted_draft),
        acceptance_rate=float(accepted_draft / max(drafted, 1)),
        avg_accepted_depth=float(np.mean([t - 1 for t in taus])),
        verifier_calls_per_token=float(n_cycles / max(committed, 1)),
        rejected_work_ratio=float(1 - accepted_draft / max(drafted, 1)),
        exact_match_rate=float(df.exact_match.mean())
        if "exact_match" in df else None,
        first_rejection_depth_hist=json.dumps(
            {str(k): v for k, v in sorted(depth_hist.items())}),
    )


def boot_ci(x, seed=BOOT_SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (BOOT_N, len(x)))].mean(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def paired_delta(x, base, seed=BOOT_SEED):
    d = np.asarray(x, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (BOOT_N, len(d)))].mean(1)
    return (float(d.mean()), float(np.quantile(b, .025)),
            float(np.quantile(b, .975)))


def per_prompt_series(df):
    return df.sort_values("prompt_id").acceptance_list.map(
        lambda l: float(np.mean(l))).to_numpy(), \
        df.sort_values("prompt_id").prompt_id.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir

    # ---------------- Study A: matrix ----------------
    cells = {}
    for f in glob.glob(os.path.join(rd, "matrix", "shards", "al__*.csv")):
        df = load_rows(f)
        for cid, g in df.groupby("cell"):
            cells[cid] = g.reset_index(drop=True)
    rows, per_prompt_rows, cyc_rows = [], [], []
    base = per_prompt_series(cells["T16_D16"])[0] if "T16_D16" in cells else None
    for cid in ["T16_D16", "T16_D8", "T16_D4", "T8_D16", "T8_D8", "T8_D4",
                "T4_D16", "T4_D8", "T4_D4"]:
        if cid not in cells:
            continue
        df = cells[cid]
        m = cycle_metrics(df)
        x, pids = per_prompt_series(df)
        lo, hi = boot_ci(x)
        m.update(cell=cid, ci95_lo=round(lo, 4), ci95_hi=round(hi, 4))
        if base is not None and cid != "T16_D16":
            d, dlo, dhi = paired_delta(x, base)
            m.update(delta_vs_T16_D16=round(d, 4),
                     delta_ci95_lo=round(dlo, 4), delta_ci95_hi=round(dhi, 4),
                     rel_pct=round(100 * d / np.mean(base), 2))
        rows.append(m)
        for pid, v in zip(pids, x):
            per_prompt_rows.append(dict(cell=cid, prompt_id=int(pid),
                                        mean_tau=round(float(v), 4)))
        for _, r in df.iterrows():
            for ci, t in enumerate(r.acceptance_list):
                cyc_rows.append(dict(cell=cid, prompt_id=int(r.prompt_id),
                                     cycle=ci, tau=int(t)))
    md = pd.DataFrame(rows)
    col1 = ["cell", "mean_tau", "mean_accepted_draft_tokens", "ci95_lo",
            "ci95_hi", "delta_vs_T16_D16", "delta_ci95_lo", "delta_ci95_hi",
            "rel_pct"]
    md.to_csv(os.path.join(rd, "precision_matrix_summary.csv"), index=False,
              columns=[c for c in col1 + [c for c in md.columns
                                          if c not in col1]
                       if c in md.columns])
    md.to_json(os.path.join(rd, "precision_matrix_summary.json"),
               orient="records", indent=2)
    pd.DataFrame(per_prompt_rows).to_csv(
        os.path.join(rd, "precision_matrix_per_prompt.csv"), index=False)
    pd.DataFrame(cyc_rows).to_parquet(
        os.path.join(rd, "precision_matrix_per_cycle.parquet"), index=False)

    if len(md) == 9:
        M = np.zeros((3, 3))
        Dl = np.zeros((3, 3))
        T, Dn = ["T16", "T8", "T4"], ["D16", "D8", "D4"]
        for i, t in enumerate(T):
            for j, d in enumerate(Dn):
                r = md[md.cell == f"{t}_{d}"].iloc[0]
                M[i, j] = r.mean_tau
                Dl[i, j] = 0 if f"{t}_{d}" == "T16_D16" else r.delta_vs_T16_D16
        for name, mat, cmap, fmt in (
                ("precision_matrix_heatmap.png", M, "viridis", "{:.3f}"),
                ("precision_matrix_delta_heatmap.png", Dl, "RdBu_r", "{:+.3f}")):
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            im = ax.imshow(mat, cmap=cmap)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center",
                            va="center",
                            color="white" if cmap == "viridis"
                            and mat[i, j] < mat.max() * .8 else "black")
            ax.set_xticks(range(3), Dn)
            ax.set_yticks(range(3), T)
            ax.set_xlabel("draft precision")
            ax.set_ylabel("target precision")
            ax.set_title("mean tau (learned rotation)" if "delta" not in name
                         else "paired delta vs T16_D16")
            fig.colorbar(im, shrink=.8)
            fig.savefig(os.path.join(rd, name), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---------------- Studies B/C: grids ----------------
    for study_name, patt, out_prefix in (
            ("component", "grid__*__draft_*", "component_sensitivity"),
            ("component", "grid__*__diag_*", "component_sensitivity"),
            ("projection", "grid__*__projection_*", "projection_scale"),
            ("projection", "grid__*__full_draft_*", "projection_scale")):
        pass  # handled jointly below

    grid = {}
    for f in glob.glob(os.path.join(rd, "shards", "grid__*.csv")):
        tag = os.path.basename(f)[6:-4]
        grid[tag] = load_rows(f)
    comp_rows, comp_pp = [], []
    proj_rows, proj_pp = [], []
    for tag, df in sorted(grid.items()):
        target, name = tag.split("__", 1)
        m = cycle_metrics(df)
        x, pids = per_prompt_series(df)
        lo, hi = boot_ci(x)
        base_tag = f"{target}__draft_all_FP16" if not name.startswith(
            ("projection", "full_draft")) else f"{target}__projection_FP16"
        m.update(config=name, target=target,
                 ci95_lo=round(lo, 4), ci95_hi=round(hi, 4))
        if base_tag in grid and tag != base_tag:
            bx, _ = per_prompt_series(grid[base_tag])
            d, dlo, dhi = paired_delta(x, bx)
            m.update(delta_vs_fp16=round(d, 4), delta_ci95_lo=round(dlo, 4),
                     delta_ci95_hi=round(dhi, 4),
                     rel_pct=round(100 * d / np.mean(bx), 2))
        bucket = (comp_rows if name.startswith(("draft_", "diag_"))
                  else proj_rows)
        bucket.append(m)
        pp = (comp_pp if name.startswith(("draft_", "diag_")) else proj_pp)
        for pid, v in zip(pids, x):
            pp.append(dict(target=target, config=name, prompt_id=int(pid),
                           mean_tau=round(float(v), 4)))
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(rd, "component_sensitivity_summary.csv"), index=False)
        pd.DataFrame(comp_pp).to_csv(
            os.path.join(rd, "component_sensitivity_per_prompt.csv"),
            index=False)
    if proj_rows:
        pd.DataFrame(proj_rows).to_csv(
            os.path.join(rd, "projection_scale_summary.csv"), index=False)
        pd.DataFrame(proj_pp).to_csv(
            os.path.join(rd, "projection_scale_per_prompt.csv"), index=False)

    # component bars
    cdf = pd.DataFrame(comp_rows)
    for tgt in ("fp16", "w4a4"):
        g = cdf[(cdf.target == tgt) & cdf.config.str.startswith("draft_")]
        if g.empty:
            continue
        g = g.sort_values("mean_tau")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(g.config, g.mean_tau, xerr=[g.mean_tau - g.ci95_lo,
                                            g.ci95_hi - g.mean_tau],
                color="#2c6fbb")
        ax.set_xlabel("mean tau")
        ax.set_title(f"Draft component W4A4 sensitivity — target {tgt}")
        fig.savefig(os.path.join(
            rd, f"component_sensitivity_bar_target_{tgt}.png"),
            dpi=150, bbox_inches="tight")
        plt.close(fig)
    # first vs recurrent figure
    fr = cdf[cdf.config.str.contains("projection_first_only|"
                                     "projection_recurrent_only|"
                                     "projection_all")]
    if not fr.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.35
        for k, tgt in enumerate(("fp16", "w4a4")):
            g = fr[fr.target == tgt]
            labs = [c.replace("draft_projection_", "").replace("_W4A4", "")
                    for c in g.config]
            ax.bar(np.arange(len(g)) + (k - .5) * w, g.mean_tau, w,
                   label=f"target {tgt}")
        ax.set_xticks(range(len(labs)), labs)
        ax.set_ylabel("mean tau")
        ax.set_title("Projection: first vs recurrent vs both (W4A4)")
        ax.legend()
        fig.savefig(os.path.join(rd, "projection_first_vs_recurrent.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    # projection acceptance figure
    pdf = pd.DataFrame(proj_rows)
    if not pdf.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        order = ["projection_FP16", "projection_W4A4_shared_scale",
                 "projection_W4A4_separate_e_h_scales",
                 "projection_W4A4_embedding_scaled_shared_scale",
                 "full_draft_W4A4_with_baseline_projection",
                 "full_draft_W4A4_with_separate_scales",
                 "full_draft_W4A4_with_embedding_scaled"]
        w = .35
        for k, tgt in enumerate(("fp16", "w4a4")):
            g = pdf[pdf.target == tgt].set_index("config").reindex(order)
            ax.bar(np.arange(len(order)) + (k - .5) * w, g.mean_tau, w,
                   label=f"target {tgt}")
        ax.set_xticks(range(len(order)),
                      [o.replace("projection_", "P.").replace(
                          "full_draft_", "FD.") for o in order],
                      rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("mean tau")
        ax.set_title("Projection scale variants and full-draft transfer")
        ax.legend()
        fig.savefig(os.path.join(rd, "projection_scale_acceptance.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("[summarize] wrote summaries for",
          len(rows), "matrix cells,", len(comp_rows), "component rows,",
          len(proj_rows), "projection rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
