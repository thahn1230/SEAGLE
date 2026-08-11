#!/usr/bin/env python
"""Ordering-experiment report -> docs/DRAROT_OPTIMIZATION_ORDER_GS_W4A4.md.
Defensive about missing artifacts. Usage: _order_report.py <run_dir>"""
import csv, json, os, subprocess, sys

rd = sys.argv[1]
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]


def tau(tag, ds):
    p = os.path.join(rd, "shards", f"al__{tag}__int4__{ds}.csv")
    if not os.path.exists(p):
        return None
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return sum(ts) / max(len(ts), 1)


def four(tag):
    v = [tau(tag, d) for d in DS]
    return None if any(x is None for x in v) else v + [sum(v) / 4]


def jload(p):
    p = os.path.join(rd, p)
    return json.load(open(p)) if os.path.exists(p) else None


holm = jload("stats/holm_adjusted.json") or {}
boot = {}
for ds in DS:
    p = os.path.join(rd, "stats", f"bootstrap_pairs_{ds}.json")
    if os.path.exists(p):
        boot[ds] = {r["name"]: r for r in json.load(open(p))
                    if r.get("status") == "ok"}
geo = jload("geometry/r5_order_geometry.json")
aggs = {n: jload(f"stats/aggregate_boot_{n}.json")
        for n in ("QF_rdg_T4_s2_vs_ARMD_T4", "B9_T4_vs_ARMD_T4",
                  "ARMD_T4_vs_P3D_T4")}
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip()

ROWS = [("R5", [3.1645, 3.6875, 3.2351, 3.8282, 3.4788]),
        ("R5 -> QAT", [3.1989, 3.6265, 3.2845, 3.7065, 3.4541]),
        ("QAT (no R5)", four("QFR_ep3g_T4")),
        ("QAT -> R5 (ARM D)", four("ARMD_T4")),
        ("R5 -> QAT -> R5(reopt) (ARM F)", four("P3D_T4")),
        ("R5 -> QAT -> R5(fresh ctl)", four("P3C_T4"))]
armd, armb = four("ARMD_T4"), [3.1989, 3.6265, 3.2845, 3.7065, 3.4541]

L = []
w = L.append
w("# Does optimization order explain the QAT-DraRot interaction? "
  "(GS, W4A4)")
w("")
w(f"Run `{rd}` · git `{git}` · hypothesis tested: \"large-space weight "
  f"adaptation should precede small-space acceptance-aware basis "
  f"adaptation\" — stated as CONDITIONAL optimization (R5* depends on "
  f"W), never justified by parameter count alone.")
w("")
w("## Main ordering table")
w("")
w("| Optimization order | MT | GSM8K | ShareGPT | HumanEval | Avg |")
w("|---|---:|---:|---:|---:|---:|")
for lbl, v in ROWS:
    w(f"| {lbl} | " + (" | ".join(f"{x:.4f}" for x in v) + " |"
                       if v else "pending |" * 5))
w("")
if armd:
    d = armd[4] - armb[4]
    w(f"**Delta_order = Avg(QAT->R5) - Avg(R5->QAT) = {d:+.4f}**")
    ag = aggs.get("QF_rdg_T4_s2_vs_ARMD_T4")
    if ag:
        w(f"(aggregate bootstrap: CI {ag['ci']}, p = "
          f"{ag['p_two_sided']})")
    w("")


def block(name, label):
    rows = []
    for ds in DS:
        r = boot.get(ds, {}).get(name)
        if not r:
            continue
        h = holm.get(f"{ds}:{name}", {})
        rows.append(
            f"| {ds} | {r['delta_b_minus_a']:+.4f} "
            f"[{r['ci_delta'][0]:+.4f}, {r['ci_delta'][1]:+.4f}] | "
            f"{r['p_two_sided']:.4f} | "
            f"{'SIG' if h.get('reject_05') else 'n.s.'} |")
    if not rows:
        return []
    return [f"**{label}**", "",
            "| dataset | delta [95% CI] | raw p | Holm |",
            "|---|---|---:|---|"] + rows + [""]


w("## Statistics")
w("")
for n, lb in (("order_primary", "QAT->R5 vs R5->QAT (primary)"),
              ("vs_r5only", "QAT->R5 vs R5-only"),
              ("vs_qatonly", "QAT->R5 vs QAT-only"),
              ("reopt_vs_b", "R5->QAT->R5reopt vs R5->QAT"),
              ("reopt_vs_armD", "warm-reopt(F) vs QAT->R5(D)"),
              ("fresh_vs_warm", "fresh vs warm R5 on same weights")):
    L += block(n, lb)
if geo:
    w("## Basis geometry (§19)")
    w("")
    w("```json")
    w(json.dumps(geo, indent=1))
    w("```")
    w("")
traj = {}
for s in (500, 1000, 1500, 2000, 2500, 3000):
    t = tau(f"ARMDS{s}", "mtbench")
    if t is not None:
        traj[s] = t
if traj:
    w("## ARM D training trajectory (mtbench official tau at rotation "
      "snapshots)")
    w("")
    w("| step | " + " | ".join(str(s) for s in traj) + " |")
    w("| tau | " + " | ".join(f"{v:.4f}" for v in traj.values()) + " |")
    w("")
open("docs/DRAROT_OPTIMIZATION_ORDER_GS_W4A4.md", "w").write(
    "\n".join(L) + "\n")
open(os.path.join(rd, "reports_order_done"), "w").write("done\n")
print("[order-report] -> docs/DRAROT_OPTIMIZATION_ORDER_GS_W4A4.md")
