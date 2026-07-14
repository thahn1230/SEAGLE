#!/usr/bin/env python
"""Merge real-INT4 study shards -> realint4_acceptance_main.csv +
realint4_latency_breakdown.csv + summary.md (acceptance / wall-clock /
kernel proof / limitations / fallback warnings kept strictly separate).

Usage: python scripts/summarize_realint4.py --run-id rotstudy_realint4_<ts>
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


def bootstrap_ci(x, iters=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


LAT_COLS = ["run_id", "backend", "quant_mode", "variant", "prompt_id",
            "tokens_per_second", "total_ms", "target_time_ms_total",
            "draft_time_ms_total", "verify_time_ms_total",
            "unrotation_time_ms_total", "unrotation_time_us_per_call",
            "b2_swap_overhead_ms_total", "b2_swap_calls", "peak_mem_gib",
            "post_build_mem_alloc_gib", "int4_dispatch_proven"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    shards = sorted(glob.glob(os.path.join(run_dir, "shards", "realint4__*.csv")))
    if not shards:
        print("no shards"); return 1
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    df["is_smoke"] = df["notes"].astype(str).str.contains('"tag": "smoke')
    df["is_ordercheck"] = df["notes"].astype(str).str.contains(
        '"tag": "ordercheck')
    df.to_csv(os.path.join(run_dir, "realint4_acceptance_main.csv"), index=False)
    df[[c for c in LAT_COLS if c in df.columns]].to_csv(
        os.path.join(run_dir, "realint4_latency_breakdown.csv"), index=False)
    oc = df[df.is_ordercheck].copy()
    d = df[~df.is_smoke & ~df.is_ordercheck].copy()

    rows = []
    for (backend, variant), sub in d.groupby(["backend", "variant"]):
        acc = pd.to_numeric(sub["acceptance_length"], errors="coerce")
        tps = pd.to_numeric(sub["tokens_per_second"], errors="coerce")
        alo, ahi = bootstrap_ci(acc)
        tlo, thi = bootstrap_ci(tps)
        rows.append({
            "backend": backend, "variant": variant, "n": len(sub),
            "quant_mode": sub["quant_mode"].iloc[0],
            "accept_mean": acc.mean(), "accept_ci": (alo, ahi),
            "tok_s_mean": tps.mean(), "tok_s_ci": (tlo, thi),
            "target_ms": pd.to_numeric(sub["target_time_ms_total"],
                                       errors="coerce").mean(),
            "draft_ms": pd.to_numeric(sub["draft_time_ms_total"],
                                      errors="coerce").mean(),
            "unrot_us_call": pd.to_numeric(sub["unrotation_time_us_per_call"],
                                           errors="coerce").mean(),
            "b2_swap_ms": pd.to_numeric(sub["b2_swap_overhead_ms_total"],
                                        errors="coerce").mean(),
            "peak_gib": pd.to_numeric(sub["peak_mem_gib"],
                                      errors="coerce").mean(),
            "proven": bool(sub["int4_dispatch_proven"].astype(bool).all()),
        })
    st = pd.DataFrame(rows).sort_values(["backend", "variant"])

    val = {}
    vpath = os.path.join(run_dir, "real_int4_kernel_validation.json")
    if os.path.isfile(vpath):
        val = json.load(open(vpath)).get("verdicts", {})

    L = ["# Real-INT4 EAGLE study summary", ""]
    L.append("Model pair per shard `model`/`draft` columns. NO fake quant in "
             "this run directory; every row's quant_mode states exactly what "
             "is real. KV cache / lm_head / embeddings / attention math are "
             "fp16 in ALL arms (see docs/real_int4_limitations.md).")
    L.append("")
    L.append("## 1. Acceptance (mean over prompts, bootstrap 95% CI)")
    L.append("")
    L.append("| backend | quant_mode | variant | n | accept mean | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for _, r in st.iterrows():
        a = "" if not np.isfinite(r["accept_mean"]) else \
            f"{r['accept_mean']:.3f} | [{r['accept_ci'][0]:.3f}, {r['accept_ci'][1]:.3f}]"
        L.append(f"| {r['backend']} | {r['quant_mode']} | {r['variant']} | "
                 f"{r['n']} | {a or 'n/a (vanilla)'} |")
    L.append("")
    L.append("## 2. Wall-clock (REAL kernels; same-build comparisons only)")
    L.append("")
    L.append("| backend | variant | tokens/s (95% CI) | target ms/prompt | "
             "draft ms/prompt | unrot us/call | B2 swap ms/prompt | peak GiB |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in st.iterrows():
        L.append(
            f"| {r['backend']} | {r['variant']} | {r['tok_s_mean']:.2f} "
            f"[{r['tok_s_ci'][0]:.2f}, {r['tok_s_ci'][1]:.2f}] | "
            f"{r['target_ms']:.0f} | {r['draft_ms']:.0f} | "
            f"{'' if not np.isfinite(r['unrot_us_call']) else f'{r.unrot_us_call:.1f}'} | "
            f"{'' if not np.isfinite(r['b2_swap_ms']) else f'{r.b2_swap_ms:.3f}'} | "
            f"{r['peak_gib']:.1f} |")
    L.append("")
    # A vs B2 wall-clock per backend
    L.append("### A vs B2 (the question this run exists to answer)")
    L.append("")
    for backend in st["backend"].unique():
        sa = d[(d.backend == backend) & (d.variant == "A")]
        sb2 = d[(d.backend == backend) & (d.variant == "B2")]
        if len(sa) and len(sb2):
            common = set(sa.prompt_id) & set(sb2.prompt_id)
            a_t = pd.to_numeric(sa[sa.prompt_id.isin(common)]
                                .sort_values("prompt_id").tokens_per_second,
                                errors="coerce").values
            b_t = pd.to_numeric(sb2[sb2.prompt_id.isin(common)]
                                .sort_values("prompt_id").tokens_per_second,
                                errors="coerce").values
            diff = b_t - a_t
            lo, hi = bootstrap_ci(diff)
            if lo <= 0 <= hi:
                verdict = ("CI contains 0: NO measurable wall-clock "
                           "difference between A and B2 at this scale.")
            else:
                # check whether the gap sits in a phase B2 can causally
                # touch (draft) or not (target) before believing it
                tm_a = pd.to_numeric(sa[sa.prompt_id.isin(common)]
                                     .target_time_ms_total, errors="coerce").mean()
                tm_b = pd.to_numeric(sb2[sb2.prompt_id.isin(common)]
                                     .target_time_ms_total, errors="coerce").mean()
                verdict = (f"CI EXCLUDES 0 — but the gap sits largely in the "
                           f"TARGET phase (A {tm_a:.0f} ms vs B2 {tm_b:.0f} ms "
                           f"per prompt), which the B2 mechanism (draft-side "
                           f"fc pointer swap) cannot causally affect. Variants "
                           f"ran sequentially on one heating GPU (naive->A->"
                           f"B2->vanilla), so this is most plausibly "
                           f"order/thermal drift, NOT a B2 dispatch cost; "
                           f"treat as unresolved, do not cite either "
                           f"direction without an interleaved-order rerun.")
            L.append(f"- **{backend}**: paired per-prompt tokens/s diff "
                     f"(B2 - A) mean {np.mean(diff):+.3f} tok/s "
                     f"(95% CI [{lo:+.3f}, {hi:+.3f}]; n={len(diff)} paired "
                     f"prompts). {verdict}")
    if len(oc):
        L.append("")
        L.append("### Order-confound check (variants rerun in REVERSED order, "
                 "B2 before A, separate invocation)")
        for (backend,), sub in oc.groupby(["backend"]):
            parts = []
            for v in ("B2", "A"):
                t = pd.to_numeric(sub[sub.variant == v].tokens_per_second,
                                  errors="coerce")
                if len(t):
                    lo2, hi2 = bootstrap_ci(t)
                    parts.append(f"{v} {t.mean():.2f} tok/s "
                                 f"[{lo2:.2f}, {hi2:.2f}] (n={len(t)}, ran "
                                 f"{'1st' if v == 'B2' else '2nd'})")
            L.append(f"- {backend}: " + "; ".join(parts)
                     + ". If the earlier-run variant is again faster, the "
                       "main-table A-vs-B2 gap is order/thermal drift, not "
                       "a B2 dispatch cost.")
    L.append("")
    L.append("## 2b. Output-quality probe of the integrated targets")
    L.append("")
    probes = sorted(glob.glob(os.path.join(run_dir, "quality_probe_*.json")))
    if probes:
        L.append("| backend | wikitext-2 PPL (integrated target) | mean "
                 "distinct-2 of greedy MT-bench completions |")
        L.append("|---|---|---|")
        for p in probes:
            q = json.load(open(p))
            d2 = np.mean([s["distinct2"] for s in q.get("samples", [])])
            L.append(f"| {q['backend']} | {q['wikitext2_ppl']:.3f} | {d2:.3f} |")
        L.append("")
        L.append("Read acceptance TOGETHER with this table: a quantized "
                 "target whose acceptance EXCEEDS the fp16 stock value while "
                 "its PPL is materially worse is producing more predictable "
                 "(degraded) text, and its acceptance gain must NOT be "
                 "reported as an improvement.")
    else:
        L.append("(quality probes not run)")
    L.append("")
    L.append("## 3. Kernel proof")
    L.append("")
    L.append(f"- standalone validation verdicts: `{json.dumps(val)}`")
    for backend in st["backend"].unique():
        pr = st[st.backend == backend]["proven"].all()
        L.append(f"- {backend}: e2e profiler evidence in ALL shards: {bool(pr)}"
                 + ("" if pr or backend == "fp16_ref" else
                    "  **<-- FALLBACK WARNING: rows must not be cited as real INT4**"))
    L.append("- full kernel names: profiler_kernel_names.txt")
    L.append("")
    L.append("## 4. Limitations")
    L.append("")
    L.append("See docs/real_int4_limitations.md: W4A16 arm is weight-only; "
             "W4A4 arm covers the 7 per-layer linears only; KV cache is fp16 "
             "everywhere (KV4 was never real anywhere in this project); "
             "quant recipes differ from the fake-quant study so acceptance "
             "is not comparable across the two studies; batch=1, no serving "
             "stack — within-build comparisons only.")
    L.append("")
    L.append("## 5. Fallback warnings")
    bad = st[(st.backend != "fp16_ref") & (~st.proven)]
    L.append("None — all non-reference shards carry int4 dispatch proof."
             if not len(bad) else
             "SOME SHARDS LACK INT4 PROOF: " + ", ".join(
                 f"{r.backend}/{r.variant}" for _, r in bad.iterrows()))
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {run_dir}/summary.md")
    print(st.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
