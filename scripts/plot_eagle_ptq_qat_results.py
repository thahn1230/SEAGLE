#!/usr/bin/env python
"""All 14 required study plots (spec section 21), PNG + PDF, with the
underlying CSV saved next to each figure. Reads tables/ + stats/ produced
by analyze_qat_attainment.py / bootstrap_eagle_tau.py and the raw shards.
Skips (with a note) any plot whose inputs are missing; rerun any time.
"""
import argparse, csv, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_micro_al import read_shard, pooled_mal

C = {"ptq": "#1f77b4", "qat": "#d62728", "rot": "#2ca02c",
     "fp16": "#7f7f7f", "naive": "#9467bd", "ceil": "#ff7f0e"}


def savefig(fig, rd, name, rows=None, cols=None):
    p = os.path.join(rd, "plots")
    os.makedirs(p, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(p, f"{name}.png"), dpi=180)
    fig.savefig(os.path.join(p, f"{name}.pdf"))
    plt.close(fig)
    if rows is not None:
        with open(os.path.join(p, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols or
                               sorted({k for r in rows for k in r}))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[plot] {name}")


def load_primary(rd):
    p = os.path.join(rd, "tables", "primary_table.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        if r["status"] == "ok" and r["tau"]:
            out[r["id"]] = dict(tau=float(r["tau"]), target=r["target"],
                                desc=r["draft"],
                                std=float(r["seed_std"] or 0)
                                if r.get("seed_std") else 0.0)
    return out


def bar(ax, ids, prim, colors, labels=None):
    xs = range(len(ids))
    vals = [prim[i]["tau"] for i in ids]
    errs = [prim[i].get("std", 0) for i in ids]
    ax.bar(xs, vals, yerr=errs, capsize=3,
           color=[colors[i] for i in range(len(ids))])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels or ids, rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel("Average Acceptance Length tau")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    return [dict(id=i, tau=prim[i]["tau"], std=prim[i].get("std", 0))
            for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    prim = load_primary(rd)
    verd = {}
    vp = os.path.join(rd, "stats", "verdicts.json")
    if os.path.exists(vp):
        verd = json.load(open(vp))
    oracle = {}
    for t in ("fp16", "int4"):
        p = os.path.join(rd, "oracle", f"oracle_{t}_mtbench.json")
        if os.path.exists(p):
            oracle[t] = json.load(open(p))

    def have(*ids):
        return all(i in prim for i in ids)

    # 1 all combinations
    ids = [i for i in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C7b",
                       "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                       "N1", "N2", "N3", "N3b") if i in prim]
    if ids:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        rows = bar(ax, ids, prim,
                   ["#1f77b4"] * len(ids),
                   [f"{i}\n({prim[i]['target']})" for i in ids])
        ax.set_title("Acceptance length, all combinations (MT-bench 80, "
                     "greedy)")
        savefig(fig, rd, "acceptance_length_all_combinations", rows)

    # 2/3 ptq vs qat per target
    for name, tgt, cells in (
            ("ptq_vs_qat_fp16_target", "fp16",
             ["C1", "N1", "N3", "C2", "C11", "C9", "C3", "C8"]),
            ("ptq_vs_qat_int4_target", "int4",
             ["C4", "N2", "N3b", "C5", "C12", "C10", "C7", "C6"])):
        ids = [i for i in cells if i in prim]
        if not ids:
            continue
        cmap = {"C1": C["fp16"], "C4": C["fp16"], "N1": C["naive"],
                "N2": C["naive"], "N3": C["naive"], "N3b": C["naive"],
                "C2": C["ptq"], "C5": C["ptq"], "C11": C["rot"],
                "C12": C["rot"], "C9": C["ptq"], "C10": C["ptq"],
                "C3": C["qat"], "C7": C["qat"], "C8": C["ceil"],
                "C6": C["ceil"]}
        fig, ax = plt.subplots(figsize=(7.5, 4))
        rows = bar(ax, ids, prim, [cmap[i] for i in ids])
        if tgt in oracle:
            ax.axhline(oracle[tgt]["oracle_tau_measured"], ls="--",
                       color="k", lw=1, label="oracle-draft ceiling")
            ax.legend(fontsize=8)
        ax.set_title(f"PTQ vs QAT, {tgt} target")
        savefig(fig, rd, name, rows)

    # 4 P1/P2/P3/QAT
    ids = [i for i in ("N2", "N3b", "C5", "C12", "C7") if i in prim]
    if ids:
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = bar(ax, ids, prim,
                   [C["naive"], C["naive"], C["ptq"], C["rot"], C["qat"]]
                   [:len(ids)],
                   ["naive (P1)", "P2 scales", "P3 (shared R_T)",
                    "P3 + local R_D", "QAT"][:len(ids)])
        ax.set_title("Structural decomposition vs QAT (INT4 target)")
        savefig(fig, rd, "p1_p2_p3_qat_comparison", rows)

    # 5 oracle-normalized
    if oracle:
        rows = []
        fig, ax = plt.subplots(figsize=(8, 4))
        ids = [i for i in ("C1", "C2", "C3", "C4", "C5", "C7", "C11",
                           "C12") if i in prim]
        us, ls = [], []
        for i in ids:
            t = prim[i]["target"]
            if t not in oracle:
                continue
            u = (prim[i]["tau"] - 1) / \
                (oracle[t]["oracle_tau_measured"] - 1)
            us.append(u)
            ls.append(f"{i}\n({t})")
            rows.append(dict(id=i, target=t, oracle_util=round(u, 4)))
        ax.bar(range(len(us)), us, color="#1f77b4")
        ax.set_xticks(range(len(us)))
        ax.set_xticklabels(ls, fontsize=8)
        ax.set_ylabel("U_oracle = (tau-1)/(tau_oracle-1)")
        ax.set_title("Oracle-normalized acceptance utilization")
        savefig(fig, rd, "oracle_normalized_acceptance", rows)

    # 6 training cost vs acceptance
    cost_p = os.path.join(rd, "tables", "preparation_cost.json")
    if os.path.exists(cost_p) and prim:
        cost = json.load(open(cost_p))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        rows = []
        pts = {"C2": 0.35, "C5": 0.35, "C11": 3.0, "C12": 3.0}
        for i, gh in pts.items():         # PTQ cells: calibration cost
            if i in prim:
                ax.scatter(gh, prim[i]["tau"], c=C["ptq"] if gh < 1
                           else C["rot"], s=45)
                ax.annotate(i, (gh, prim[i]["tau"]), fontsize=8)
                rows.append(dict(id=i, gpu_hours=gh,
                                 tau=prim[i]["tau"]))
        for tag, m in cost.items():
            cid = tag.split("_")[0]
            if cid in prim and m.get("gpu_hours"):
                ax.scatter(m["gpu_hours"], prim[cid]["tau"], c=C["qat"],
                           s=45)
                ax.annotate(cid, (m["gpu_hours"], prim[cid]["tau"]),
                            fontsize=8)
                rows.append(dict(id=cid, gpu_hours=m["gpu_hours"],
                                 tau=prim[cid]["tau"]))
        ax.set_xlabel("preparation GPU-hours (log)")
        ax.set_xscale("log")
        ax.set_ylabel("tau")
        ax.set_title("Preparation cost vs acceptance")
        savefig(fig, rd, "training_cost_vs_acceptance", rows)

    # 7/8 acceptance by depth + first rejection
    dep_p = os.path.join(rd, "tables", "acceptance_by_depth.json")
    if os.path.exists(dep_p):
        dep = json.load(open(dep_p))
        for name, title in (("acceptance_by_depth",
                             "Per-cycle advance distribution"),
                            ("first_rejection_distribution",
                             "First-rejection position (advance-1)")):
            fig, ax = plt.subplots(figsize=(7.5, 4))
            rows = []
            for cid, hist in dep.items():
                ks = sorted(int(k) for k in hist)
                off = 0 if name == "acceptance_by_depth" else -1
                ax.plot([k + off for k in ks],
                        [hist[str(k)] for k in ks], marker="o", ms=3,
                        label=cid)
                for k in ks:
                    rows.append(dict(id=cid, depth=k + off,
                                     frac=hist[str(k)]))
            ax.set_xlabel("tokens" if name == "acceptance_by_depth"
                          else "accepted draft tokens before rejection")
            ax.set_ylabel("fraction of cycles")
            ax.legend(fontsize=7, ncol=3)
            ax.set_title(title)
            savefig(fig, rd, name, rows)

    # 9 seed variance
    sv_p = os.path.join(rd, "tables", "seed_variance.json")
    if os.path.exists(sv_p):
        sv = json.load(open(sv_p))
        if sv:
            fig, ax = plt.subplots(figsize=(5, 4))
            for cid, col in (("C3", C["qat"]), ("C7", "#8c564b")):
                pts = [r for r in sv if r["id"] == cid]
                if pts:
                    ax.scatter([r["seed"] for r in pts],
                               [r["tau"] for r in pts], label=cid,
                               c=col, s=50)
            ax.set_xlabel("seed")
            ax.set_ylabel("tau")
            ax.set_xticks([0, 1, 2])
            ax.legend()
            ax.set_title("QAT seed variance")
            savefig(fig, rd, "seed_variance_qat", sv)

    # 10 target teacher mismatch
    ids = [i for i in ("C6", "C13", "C7", "C14") if i in prim]
    if len(ids) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        rows = bar(ax, ids, prim, ["#1f77b4"] * len(ids),
                   [{"C6": "C6 (INT4 teacher)",
                     "C13": "C13 (FP16 teacher)",
                     "C7": "C7 (INT4 teacher QAT)",
                     "C14": "C14 (FP16 teacher QAT)"}[i] for i in ids])
        ax.set_title("Teacher-target mismatch (INT4 deployment)")
        savefig(fig, rd, "target_teacher_mismatch", rows)

    # 11 fp16 retraining + PTQ vs QAT
    ids = [i for i in ("C2", "C9", "C3", "C5", "C10", "C7") if i in prim]
    if ids:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        rows = bar(ax, ids, prim,
                   [C["ptq"], C["ceil"], C["qat"]] * 2)
        ax.set_title("Retrain-then-PTQ vs direct QAT")
        savefig(fig, rd, "fp16_retraining_ptq_vs_qat", rows)

    # 12 recovery ratio
    att = verd.get("attainment", {})
    rows = []
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, vals = [], []
    for t in ("fp16", "int4"):
        rrr = att.get(t, {}).get("recovery_ratio")
        if rrr is not None:
            xs.append(t)
            vals.append(rrr)
            rows.append(dict(target=t, recovery_ratio=rrr))
    if vals:
        ax.bar(xs, vals, color=C["ptq"])
        ax.axhline(0.90, ls="--", c="k", lw=1, label="criterion B (0.90)")
        ax.set_ylabel("structural_recovery / recoverable_gap")
        ax.legend()
        ax.set_title("Training-free structural recovery ratio")
        savefig(fig, rd, "recovery_ratio_training_free", rows)
    else:
        plt.close(fig)

    # 13 target quality vs acceptance
    tq_p = os.path.join(rd, "tables", "target_quality.json")
    if os.path.exists(tq_p) and prim:
        tq = json.load(open(tq_p))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        rows = []
        for t, base in (("fp16", "C1"), ("int4", "C5")):
            if t in tq and base in prim:
                ax.scatter(tq[t]["wikitext2_ppl"], prim[base]["tau"],
                           s=60, label=f"{t} (ppl "
                           f"{tq[t]['wikitext2_ppl']:.2f})")
                rows.append(dict(target=t, ppl=tq[t]["wikitext2_ppl"],
                                 tau=prim[base]["tau"]))
        ax.set_xlabel("WikiText-2 PPL of target")
        ax.set_ylabel("tau of best strict-PTQ system")
        ax.legend(fontsize=8)
        ax.set_title("Target quality vs acceptance")
        savefig(fig, rd, "target_quality_vs_acceptance", rows)

    # 14 speculative fidelity
    rows = []
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, ems, mps = [], [], []
    for t in ("fp16", "int4"):
        p = os.path.join(rd, "tables", f"speculative_fidelity_{t}.json")
        if os.path.exists(p):
            s = json.load(open(p))
            xs.append(t)
            ems.append(s["exact_match_rate"])
            mps.append(s["mean_prefix_agreement"])
            rows.append(s)
    if xs:
        w = 0.35
        ax.bar([i - w / 2 for i in range(len(xs))], ems, w,
               label="exact-match rate")
        ax.bar([i + w / 2 for i in range(len(xs))], mps, w,
               label="mean prefix agreement")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title("Sequential vs EAGLE-tree target fidelity")
        savefig(fig, rd, "speculative_fidelity", rows)
    else:
        plt.close(fig)
    print("[plot] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
