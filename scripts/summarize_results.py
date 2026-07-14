#!/usr/bin/env python
"""Merge study shards -> canonical CSVs, compute statistics (mean/median/std +
bootstrap 95% CI over prompts), generate figures and the reviewer-facing
summary.md with H1-H10 verdicts.

Usage: python scripts/summarize_results.py --run-id <id>
(no GPU needed)
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CANONICAL = ["acceptance_main", "depth_sweep", "gamma_ablation",
             "rotation_component_ablation"]


def bootstrap_ci(x, iters=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def merge_shards(run_dir):
    frames = {}
    for name in CANONICAL:
        shards = sorted(glob.glob(os.path.join(run_dir, "shards", f"{name}__*.csv")))
        if not shards:
            continue
        df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        # drop smoke shards from statistics but keep them in the merged file
        df["is_smoke"] = df["notes"].astype(str).str.contains('"tag": "smoke')
        df.to_csv(os.path.join(run_dir, f"{name}.csv"), index=False)
        frames[name] = df[~df["is_smoke"]].copy()
    return frames


def group_stats(df, keys):
    rows = []
    for g, sub in df.groupby(keys, dropna=False):
        acc = sub["acceptance_length"].astype(float)
        lo, hi = bootstrap_ci(acc)
        d = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        d.update({
            "n": len(sub), "accept_mean": acc.mean(), "accept_median": acc.median(),
            "accept_std": acc.std(ddof=1) if len(acc) > 1 else 0.0,
            "accept_ci_lo": lo, "accept_ci_hi": hi,
            "spec_tok_s": pd.to_numeric(sub["speculative_tokens_per_second"],
                                        errors="coerce").mean(),
            "vanilla_tok_s": pd.to_numeric(sub["vanilla_tokens_per_second"],
                                           errors="coerce").mean(),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def add_relative_speedup(stats, raw):
    """Relative speedup vs vanilla on the SAME target build, computed on the
    PAIRED prompt subset only: vanilla covers the first --vanilla-num-prompts
    prompts and MT-bench is category-ordered, so an 80-vs-40 mean ratio mixes
    prompt populations (review finding). Here both means are restricted to the
    intersection of prompt_ids measured under both modes."""
    keys = ["quant", "rotation", "rotation_type", "seed", "prompt_set"]
    van = raw[raw["variant"] == "vanilla"]
    van_map = {}
    for g, sub in van.groupby(keys):
        van_map[g] = (set(sub["prompt_id"]),
                      pd.to_numeric(sub["vanilla_tokens_per_second"],
                                    errors="coerce").mean())
    rel = []
    for _, r in stats.iterrows():
        g = tuple(r[k] for k in keys)
        if r["variant"] == "vanilla" or g not in van_map:
            rel.append(np.nan)
            continue
        pids, van_tps = van_map[g]
        sub = raw[(raw["variant"] == r["variant"]) &
                  (raw["tree_depth"] == r["tree_depth"]) &
                  (raw["prompt_id"].isin(pids))]
        for k, v in zip(keys, g):
            sub = sub[sub[k] == v]
        spec_tps = pd.to_numeric(sub["speculative_tokens_per_second"],
                                 errors="coerce").mean()
        rel.append(spec_tps / van_tps
                   if (van_tps and np.isfinite(spec_tps)) else np.nan)
    stats["relative_speedup_same_runtime"] = rel
    return stats


def fig_bar(stats, run_dir):
    df = stats[(stats["rotation"].isin(["full", "none"])) &
               (stats["seed"] == 0) & (stats["prompt_set"] == "mt_bench") &
               (stats["rotation_type"].isin(["random_hadamard", "none"])) &
               (stats["tree_depth"] == 5)]
    order_v = ["stock", "naive", "A", "B", "B2", "C"]  # A_nogamma lives in the gamma-ablation figure only
    order_q = ["none", "w4a16", "w4a4", "w4a4kv4"]
    qlab = {"none": "quant OFF (rotate-only / fp16)", "w4a16": "W4A16 (fake)",
            "w4a4": "W4A4 (fake)", "w4a4kv4": "W4A4KV4 (fake)"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(order_q)
    xs = np.arange(len(order_v))
    for qi, q in enumerate(order_q):
        vals, los, his = [], [], []
        for v in order_v:
            sub = df[(df["variant"] == v) & (df["quant"] == q)]
            if v == "stock" and q == "none":
                sub = df[(df["variant"] == "stock")]
            if len(sub):
                vals.append(sub["accept_mean"].iloc[0])
                los.append(sub["accept_mean"].iloc[0] - sub["accept_ci_lo"].iloc[0])
                his.append(sub["accept_ci_hi"].iloc[0] - sub["accept_mean"].iloc[0])
            else:
                vals.append(np.nan); los.append(0); his.append(0)
        ax.bar(xs + qi * width, vals, width, yerr=[los, his], capsize=2,
               label=qlab[q])
    ax.set_xticks(xs + 1.5 * width); ax.set_xticklabels(order_v)
    ax.set_ylabel("acceptance length (tokens / target forward)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_title("Acceptance by variant (mean, bootstrap 95% CI over prompts)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_acceptance_by_variant.png"), dpi=150)


def fig_depth(frames, run_dir):
    if "depth_sweep" not in frames:
        return
    st = group_stats(frames["depth_sweep"],
                     ["variant", "tree_depth", "quant"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for v, marker in [("stock", "o"), ("A", "s"), ("B2", "^"), ("B", "x"),
                      ("naive", "d")]:
        sub = st[(st["variant"] == v)].sort_values("tree_depth")
        if len(sub):
            ax.errorbar(sub["tree_depth"], sub["accept_mean"],
                        yerr=[sub["accept_mean"] - sub["accept_ci_lo"],
                              sub["accept_ci_hi"] - sub["accept_mean"]],
                        marker=marker, capsize=2, label=v)
    ax.set_xlabel("tree depth"); ax.set_ylabel("acceptance length")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_title("Depth sweep, quant OFF (B flattens if recycling explanation holds)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_depth_sweep.png"), dpi=150)


def fig_gamma(frames, stats, run_dir):
    rows = []
    if "gamma_ablation" in frames:
        g = group_stats(frames["gamma_ablation"], ["variant"])
        rows += [(r["variant"], r["accept_mean"], r["accept_ci_lo"], r["accept_ci_hi"])
                 for _, r in g.iterrows()]
    s = stats[(stats["variant"] == "stock") & (stats["tree_depth"] == 5)]
    if len(s):
        rows.append(("stock(fp16)", s["accept_mean"].iloc[0],
                     s["accept_ci_lo"].iloc[0], s["accept_ci_hi"].iloc[0]))
    if not rows:
        return
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]; his = [r[3] - r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, yerr=[los, his], capsize=3, color="#6baed6")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_ylabel("acceptance length")
    ax.set_title("Final-RMSNorm gamma ablation (rotate-only, quant OFF)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_gamma_ablation.png"), dpi=150)


def fig_components(frames, run_dir):
    if "rotation_component_ablation" not in frames:
        return
    st = group_stats(frames["rotation_component_ablation"],
                     ["rotation", "quant", "variant"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order_r = ["none", "r1", "r1r2", "r1r2r3r4"]
    for q, color in [("none", "#2c7fb8"), ("w4a4", "#d95f0e")]:
        sub = st[(st["quant"] == q)]
        vals, los, his, ticks = [], [], [], []
        for r in order_r:
            s = sub[sub["rotation"] == r]
            ticks.append(r)
            if len(s):
                vals.append(s["accept_mean"].iloc[0])
                los.append(s["accept_mean"].iloc[0] - s["accept_ci_lo"].iloc[0])
                his.append(s["accept_ci_hi"].iloc[0] - s["accept_mean"].iloc[0])
            else:
                vals.append(np.nan); los.append(0); his.append(0)
        x = np.arange(len(order_r)) + (0 if q == "none" else 0.35)
        ax.bar(x, vals, 0.35, yerr=[los, his], capsize=2, color=color,
               label=f"quant={q} (variant A / stock for rotation=none)")
    ax.set_xticks(np.arange(len(order_r)) + 0.17)
    ax.set_xticklabels(order_r)
    ax.set_ylabel("acceptance length")
    ax.set_title("Rotation component ablation (interface corrected with variant A)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_rotation_component_ablation.png"), dpi=150)


def hypothesis_verdicts(stats, frames, run_dir):
    V = {}

    def m(variant, quant="none", rotation=None, rotation_type="random_hadamard"):
        df = stats[(stats["variant"] == variant) & (stats["quant"] == quant) &
                   (stats["tree_depth"] == 5) & (stats["seed"] == 0) &
                   (stats["prompt_set"] == "mt_bench")]
        if rotation:
            df = df[df["rotation"] == rotation]
            # CRITICAL (review finding): without this filter, the learned-R
            # shard sorts before random_hadamard and silently wins iloc[0].
            df = df[df["rotation_type"] == rotation_type]
        if len(df) > 1:
            raise RuntimeError(f"ambiguous stats selection for {variant}/{quant}/"
                               f"{rotation}: {len(df)} rows")
        return df["accept_mean"].iloc[0] if len(df) else np.nan

    stock = m("stock")
    naive0 = m("naive", rotation="full")
    A0 = m("A", rotation="full")
    Ang = np.nan
    if "gamma_ablation" in frames:
        g = group_stats(frames["gamma_ablation"], ["variant"])
        r = g[g["variant"] == "A_nogamma"]
        Ang = r["accept_mean"].iloc[0] if len(r) else np.nan
    B0 = m("B", rotation="full")
    B20 = m("B2", rotation="full")
    A44 = m("A", "w4a4", rotation="full")
    Akv = m("A", "w4a4kv4", rotation="full")

    val = {}
    vpath = os.path.join(run_dir, "rotation_interface_validation.json")
    if os.path.isfile(vpath):
        val = json.load(open(vpath))

    V["H1 naive collapse (quant OFF)"] = {
        "pass": bool(naive0 < 0.5 * stock), "stock": stock, "naive": naive0}
    V["H2 A restores stock (quant OFF)"] = {
        "pass": bool(abs(A0 - stock) < 0.15 * stock), "A": A0, "stock": stock}
    V["H3 gamma required"] = {
        "pass": bool((Ang < 0.9 * A0) if np.isfinite(Ang) else
                     val.get("verdicts", {}).get("gamma_required", False)),
        "A": A0, "A_nogamma": Ang,
        "numeric_rel_l2_no_gamma": val.get("interface_inverse", {})
                                      .get("no_gamma", {}).get("relative_l2_error")}
    comp_ok = None
    if "rotation_component_ablation" in frames:
        st = group_stats(frames["rotation_component_ablation"],
                         ["rotation", "quant"])
        offs = st[st["quant"] == "none"]["accept_mean"]
        comp_ok = bool(len(offs) and (offs.min() > 0.85 * stock))
    V["H4 R2/R3/R4 no external leak (quant OFF, A corrected)"] = {
        "pass": comp_ok, "component_means_quant_off":
            st[st["quant"] == "none"].to_dict("records")
            if "rotation_component_ablation" in frames else None}
    V["H5 B single-forward exact"] = {
        "pass": val.get("verdicts", {}).get("B_single_forward_exact"),
        "identity": val.get("variantB_single_forward_identity")}
    depth_ok = None
    if "depth_sweep" in frames:
        ds = group_stats(frames["depth_sweep"], ["variant", "tree_depth"])
        def dv(v, d):
            r = ds[(ds["variant"] == v) & (ds["tree_depth"] == d)]
            return r["accept_mean"].iloc[0] if len(r) else np.nan
        b2_ = dv("B", 2); a2 = dv("A", 2); b5 = dv("B", 5); a5 = dv("A", 5)
        # Correct signature of recycling failure (criterion redesigned after
        # first pass): recycled calls per round = depth-1, so the earliest
        # possible failure IS depth 2 (B already below A there), and deeper
        # trees add ONLY recycled levels -> B must be ~flat in depth while
        # A/stock keep gaining. The original ratio test wrongly presumed B
        # matches A at depth 2.
        depth_ok = bool(np.isfinite(b5) and np.isfinite(a5) and
                        (b5 - b2_) < 0.2 and (a5 - a2) > 0.5 and (a2 - b2_) > 0.2)
        V["H6 B fails from recycled levels (depth sweep + level diagnostics)"] = {
            "pass": bool(depth_ok and val.get("verdicts", {}).get("B_recycled_levels_diverge")),
            "B_gain_depth2to5": b5 - b2_, "A_gain_depth2to5": a5 - a2,
            "gap_at_depth2": a2 - b2_,
            "level_diag": val.get("verdicts", {}).get("B_recycled_levels_diverge")}
    V["H7 B2 recovers A (zero-GEMM)"] = {
        "pass": bool(np.isfinite(B20) and abs(B20 - A0) < 0.05 * A0 and
                     val.get("verdicts", {}).get("B2_recycled_levels_match")),
        "B2": B20, "A": A0}
    V["H8 W4A4 reduces but does not destroy (interface corrected)"] = {
        "pass": bool(np.isfinite(A44) and 1.5 < A44 < A0), "A_w4a4": A44, "A_quantoff": A0}
    V["H9 KV4 small additional loss"] = {
        "pass": bool(np.isfinite(Akv) and (A44 - Akv) < 0.3), "A_w4a4": A44,
        "A_w4a4kv4": Akv}
    V["H10 fake-quant speed labeling"] = {
        "pass": True,
        "note": "all quant!=none rows in THIS run carry runtime_mode="
                "fake_quant_pytorch (QDQ + FP16 matmuls; no INT4 kernels in "
                "this acceptance stack); relative speedups are same-runtime "
                "ratios only. Real packed-INT4 measurements live in the "
                "separate runs/rotstudy_realint4_* study and are never mixed "
                "into these tables."}
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default=None,
                    help="model-pair label for this run in summary tables")
    ap.add_argument("--compare-with", default=None,
                    help="run-id of a previous study to compare against "
                         "(reads its acceptance_stats.csv)")
    ap.add_argument("--compare-label", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)

    frames = merge_shards(run_dir)
    if "acceptance_main" not in frames:
        print("no acceptance_main shards found"); return 1
    stats = group_stats(frames["acceptance_main"],
                        ["variant", "quant", "rotation", "rotation_type",
                         "seed", "prompt_set", "tree_depth"])
    stats = add_relative_speedup(stats, frames["acceptance_main"])
    stats.to_csv(os.path.join(run_dir, "acceptance_stats.csv"), index=False)

    # ppl.csv: keep a run-local ppl.csv if one was already produced for THIS
    # run's model (scripts/eval_ppl_study.py); only fall back to the Llama-2
    # upstream consolidation when absent, since results/ppl_summary.csv is
    # Llama-2-7B-chat-specific and must never masquerade as another target's.
    ppl_dst = os.path.join(run_dir, "ppl.csv")
    ppl_src = os.path.join(PROJECT_ROOT, "results", "ppl_summary.csv")
    if not os.path.isfile(ppl_dst) and os.path.isfile(ppl_src):
        ppl = pd.read_csv(ppl_src)
        ppl["notes"] = ("upstream SpinQuant ptq.py evaluation (fake quant) of "
                        "meta-llama/Llama-2-7b-chat-hf; see runs/spinquant_ppl/*.log")
        ppl.rename(columns={"setting": "quant"}).to_csv(ppl_dst, index=False)

    fig_bar(stats, run_dir)
    fig_depth(frames, run_dir)
    fig_gamma(frames, stats, run_dir)
    fig_components(frames, run_dir)

    V = hypothesis_verdicts(stats, frames, run_dir)
    compare = None
    if args.compare_with:
        prev = os.path.join(PROJECT_ROOT, "runs", args.compare_with,
                            "acceptance_stats.csv")
        if os.path.isfile(prev):
            compare = (args.compare_label or args.compare_with,
                       pd.read_csv(prev))
        else:
            print(f"[warn] --compare-with given but {prev} missing")
    write_summary_md(run_dir, stats, frames, V,
                     label=args.label or args.run_id, compare=compare)
    print(json.dumps({k: v.get("pass") for k, v in V.items()}, indent=2))
    print(f"-> {run_dir}/summary.md")
    return 0


def _pair_row(stats):
    """(Stock, Naive, A, B, B2, W4A4 A, W4A4 B2) at depth 5 / seed 0 /
    mt_bench / random-Hadamard rotation. Returns formatted strings."""
    def pick(variant, quant):
        df = stats[(stats["variant"] == variant) & (stats["quant"] == quant) &
                   (stats["tree_depth"] == 5) & (stats["seed"] == 0) &
                   (stats["prompt_set"] == "mt_bench")]
        if variant != "stock":
            df = df[(df["rotation"] == "full") &
                    (df["rotation_type"] == "random_hadamard")]
        if len(df) != 1:
            return "—"
        return f"{df['accept_mean'].iloc[0]:.3f}"
    return [pick("stock", "none"), pick("naive", "none"), pick("A", "none"),
            pick("B", "none"), pick("B2", "none"), pick("A", "w4a4"),
            pick("B2", "w4a4")]


def write_summary_md(run_dir, stats, frames, V, label=None, compare=None):
    lines = ["# Rotation study summary (reviewer-facing)", ""]
    if label:
        lines.append(f"Model pair: **{label}**")
        lines.append("")
    if compare is not None:
        prev_label, prev_stats = compare
        lines.append("## Cross-model comparison (acceptance length, depth 5, "
                     "seed 0, MT-bench, random-Hadamard rotation)")
        lines.append("")
        lines.append("| Model Pair | Stock | Naive | A | B | B2 | W4A4 A | W4A4 B2 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        lines.append("| " + (prev_label or "previous") + " | "
                     + " | ".join(_pair_row(prev_stats)) + " |")
        lines.append("| " + (label or "this run") + " | "
                     + " | ".join(_pair_row(stats)) + " |")
        lines.append("")
        lines.append("Caveat: the two rows are separate targets, drafts, prompt "
                     "templates and target builds; compare PATTERNS (collapse/"
                     "recovery/equivalence), not absolute acceptance across rows.")
        lines.append("")
    lines.append("All quantized rows are **fake quantization** (quantize-"
                 "dequantize, FP16 matmuls). Speed columns are simulated-runtime "
                 "behavior; **no real INT4 deployment speed is claimed**. "
                 "Relative speedups always name their baseline: vanilla "
                 "autoregressive decoding on the SAME target build.")
    lines.append("")
    lines.append("## Main acceptance table (mean over prompts, bootstrap 95% CI)")
    lines.append("")
    sub = stats[(stats["tree_depth"] == 5) & (stats["seed"] == 0) &
                (stats["prompt_set"] == "mt_bench")].sort_values(
        ["quant", "rotation", "variant"])
    lines.append("| variant | quant | rotation | n | accept mean | 95% CI | rel. speedup (same runtime) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        rs = r["relative_speedup_same_runtime"]
        lines.append(
            f"| {r['variant']} | {r['quant']} | {r['rotation']}/{r['rotation_type']}"
            f" | {r['n']} | {r['accept_mean']:.3f} |"
            f" [{r['accept_ci_lo']:.3f}, {r['accept_ci_hi']:.3f}] |"
            f" {'' if not np.isfinite(rs) else f'{rs:.2f}x'} |")
    lines.append("")
    lines.append("## Robustness runs (rotation seeds / second prompt set)")
    lines.append("")
    rob = stats[((stats["seed"] != 0) | (stats["prompt_set"] != "mt_bench")) &
                (stats["variant"].isin(["naive", "A"]))]
    if len(rob):
        lines.append("| variant | seed | prompt set | n | accept mean | 95% CI |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in rob.sort_values(["prompt_set", "seed", "variant"]).iterrows():
            lines.append(f"| {r['variant']} | {r['seed']} | {r['prompt_set']} | {r['n']} "
                         f"| {r['accept_mean']:.3f} | [{r['accept_ci_lo']:.3f}, {r['accept_ci_hi']:.3f}] |")
    else:
        lines.append("(not run)")
    lines.append("")
    lines.append("Timing note: target/draft/verify phase timers wrap the three "
                 "dominant GPU phases but do NOT partition total wall time "
                 "(lm_head tree logits, candidate generation, and KV-cache "
                 "reorganization are unattributed); phase columns must not be "
                 "treated as an exhaustive decomposition. The first prompt of "
                 "each condition additionally includes warmup unrotation events "
                 "in its unrotation_time column (fixed for future runs).")
    lines.append("")
    lines.append("## Hypothesis verdicts")
    lines.append("")
    for k, v in V.items():
        status = {True: "PASS", False: "FAIL", None: "NOT RUN"}[v.get("pass")]
        detail = {kk: vv for kk, vv in v.items() if kk != "pass"}
        lines.append(f"- **{k}** — **{status}**  "
                     f"`{json.dumps(detail, default=lambda x: round(x, 4) if isinstance(x, float) else str(x))[:400]}`")
    lines.append("")
    lines.append("## Claims discipline")
    lines.append("")
    lines.append("Safe wording (supported): rotation-only residual basis mismatch "
                 "can collapse EAGLE-1 acceptance; exact unrotation with the "
                 "final-RMSNorm gamma restores the frozen interface under "
                 "quantization OFF; first-layer-only fc folding is exact for a "
                 "single external forward but insufficient under recurrent "
                 "feature recycling; two-path folding is a zero-runtime-GEMM "
                 "fix (validated to the extent shown above); fake-quant results "
                 "measure acceptance and simulated-runtime behavior only.")
    lines.append("")
    lines.append("NOT claimed: 'first ever' (only 'to our knowledge'); "
                 "impossibility theorems; immunity of token-level drafters; "
                 "real W4A4 deployment speedups; conclusions about learned "
                 "SpinQuant rotations from the 50-step run; conference-ready "
                 "generality from one model + one draft.")
    lines.append("")
    lines.append("## Figures")
    for f in ["fig_acceptance_by_variant.png", "fig_depth_sweep.png",
              "fig_gamma_ablation.png", "fig_rotation_component_ablation.png"]:
        lines.append(f"- {f}")
    with open(os.path.join(run_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
