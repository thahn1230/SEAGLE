#!/usr/bin/env python3
"""Closure section 33: 8 figures + raw CSV backing. Reads the computed
pareto CSVs (no hard-coded summary values) + closure tables."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/thahn1230/SEAGLE"
NR = os.path.join(REPO, "runs/eagle1_qanchor_closure_20260816")
OUT = os.path.join(REPO, "results/qanchor_closure/figures")
FR = os.path.join(REPO, "results/qanchor_closure/frontier")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(FR,
                                             "pareto_frontier_dq.csv"))))
for r in rows:
    for k in ("mean4", "D_Q", "H_Q_all", "H_Q_controlled", "H_Q_down"):
        r[k] = float(r[k])

FAM_STYLE = {
    "ptq": ("black", "*", 220), "plain": ("tab:gray", "o", 60),
    "soft-cell": ("tab:cyan", "^", 70),
    "hard-budget": ("tab:red", "s", 80),
    "lora": ("tab:olive", "X", 80),
    "joint-basis": ("tab:purple", "D", 80),
}


def scatter(xkey, xlabel, fname, title):
    plt.figure(figsize=(7.5, 5.2))
    seen = set()
    for r in rows:
        c, m, s = FAM_STYLE[r["family"]]
        lb = r["family"] if r["family"] not in seen else None
        seen.add(r["family"])
        plt.scatter(r[xkey], r["mean4"], c=c, marker=m, s=s, label=lb,
                    zorder=3, alpha=0.9)
    front = sorted([r for r in rows
                    if r[f"pareto_optimal_{xkey}"] == "True"],
                   key=lambda r: r[xkey])
    if front:
        plt.plot([r[xkey] for r in front], [r["mean4"] for r in front],
                 "k--", lw=0.8, alpha=0.5, zorder=2)
    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel(xlabel)
    plt.ylabel("official micro-tau mean4")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()


# Figures 1-3
scatter("D_Q", "dequantized drift D_Q (frozen anchor scales)",
        "fig1_pareto_dq.png", "Corrected frontier: tau vs D_Q")
scatter("H_Q_all", "H_Q over ALL deployment-visible codes",
        "fig2_pareto_hq_all.png", "Corrected frontier: tau vs H_Q_all")
scatter("H_Q_controlled", "H_Q over hard-budget-CONTROLLED folds",
        "fig3_pareto_hq_controlled.png",
        "Diagnostic frontier: tau vs H_Q_controlled")

# Figure 4: plain LR frontier vs anchored (tau vs LR, seed points)
plt.figure(figsize=(7.5, 5))
LRX = {"3e-7": 3e-7, "1e-6": 1e-6, "3e-6": 3e-6, "1e-5": 1e-5}
for obj, col in (("conv", "tab:blue"), ("hybrid", "tab:orange")):
    pts = [(LRX[r["lr"]], r["mean4"]) for r in rows
           if r["family"] == "plain" and r["objective"] == obj
           and r["lr"] in LRX]
    if pts:
        plt.scatter([p[0] for p in pts], [p[1] for p in pts], c=col,
                    label=f"plain {obj} (seed pts)", s=55, alpha=0.85)
hb = [r["mean4"] for r in rows if r["family"] == "hard-budget"]
ptq = [r["mean4"] for r in rows if r["family"] == "ptq"][0]
plt.axhline(ptq, color="k", ls=":", lw=1, label=f"PTQ {ptq:.4f}")
if hb:
    plt.axhspan(min(hb), max(hb), color="tab:red", alpha=0.15,
                label=f"HB seed range [{min(hb):.3f},{max(hb):.3f}]")
plt.xscale("log")
plt.xlabel("plain-QAT learning rate")
plt.ylabel("mean4")
plt.title("Plain LR frontier vs hard-budget anchored band")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_lr_frontier_vs_anchored.png"),
            dpi=150)
plt.close()
with open(os.path.join(OUT, "fig4_data.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "objective", "lr", "seed", "mean4"])
    for r in rows:
        w.writerow([r["family"], r["objective"], r["lr"], r["seed"],
                    r["mean4"]])

# Figure 5: 3-seed mean4 plot
seedsets = {}
for r in rows:
    if r["family"] in ("hard-budget", "plain") and r["lr"] in ("1e-6",
                                                               "1e-5"):
        key = f"{r['family']}:{r['objective']}:{r['lr']}:{r['budget']}"
        seedsets.setdefault(key, []).append((int(r["seed"]),
                                             r["mean4"]))
plt.figure(figsize=(8, 5))
x = 0
xt = []
for key, vs in sorted(seedsets.items()):
    if len(vs) < 3:
        continue
    for s, v in sorted(vs):
        plt.scatter(x, v, c="tab:red" if "hard" in key else "tab:gray",
                    s=60)
    mean = sum(v for _, v in vs) / len(vs)
    plt.hlines(mean, x - 0.25, x + 0.25, color="k", lw=2)
    xt.append((x, key.replace("hard-budget", "HB").replace(
        "plain", "P")))
    x += 1
plt.axhline(ptq, color="k", ls=":", lw=1)
plt.xticks([a for a, _ in xt], [b for _, b in xt], rotation=30,
           fontsize=7)
plt.ylabel("mean4 (dots = seeds, bar = seed mean)")
plt.title("Training-seed closure (3-seed families)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_seed_closure.png"), dpi=150)
plt.close()
with open(os.path.join(OUT, "fig5_data.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["family_key", "seed", "mean4"])
    for key, vs in sorted(seedsets.items()):
        for s, v in sorted(vs):
            w.writerow([key, s, v])

# Figure 6: W_first causal ablation
ct = json.load(open(os.path.join(NR, "tables", "closure_taus.json")))
qt = json.load(open(os.path.join(
    REPO, "runs/eagle1_qat_qanchor_causal_20260814/tables/"
    "final_taus.json")))["taus"]
abl = [("PTQ (C0)", qt["CANON_B9"]["mean4"]),
       ("HB_controlled_only\n(= no_Wfirst)",
        ct["CF_HB_no_Wfirst"]["mean4"]),
       ("HB_no_WfirstH", ct["CF_HB_no_WfirstH"]["mean4"]),
       ("HB_WfirstH_only", ct["CF_HB_WfirstH_only"]["mean4"]),
       ("HB_Wfirst_only", ct["CF_HB_Wfirst_only"]["mean4"]),
       ("full HB (CHB)",
        qt["FIN_HB_B_hybrid_b0.005_s0_st3000"]["mean4"])]
plt.figure(figsize=(8, 4.6))
plt.bar(range(len(abl)), [v for _, v in abl],
        color=["black", "tab:blue", "tab:blue", "tab:orange",
               "tab:orange", "tab:red"])
plt.xticks(range(len(abl)), [n for n, _ in abl], fontsize=7)
plt.ylim(3.4, 3.66)
plt.ylabel("mean4")
plt.title("W_first deployment-code counterfactuals (HB hybrid b0.5% s0)")
for i, (_, v) in enumerate(abl):
    plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_wfirst_ablation.png"), dpi=150)
plt.close()
with open(os.path.join(OUT, "fig6_data.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["counterfactual", "mean4"])
    w.writerows(abl)

# Figure 7: module-wise flip composition
sel = json.load(open(os.path.join(
    REPO, "results/qanchor_closure/selectivity/"
    "selectivity_closure.json")))["down_audit"]
names = list(sel.keys())
plt.figure(figsize=(8, 4.6))
for i, n in enumerate(names):
    a = sel[n]
    tot = a["total_changed_codes"]
    wf = a["wfirst_h_fraction_of_all"]
    dn = a["down_fraction_of_all_flips"]
    other = 1 - wf - dn
    plt.bar(i, wf, color="tab:purple",
            label="W_first h" if i == 0 else None)
    plt.bar(i, dn, bottom=wf, color="tab:brown",
            label="down" if i == 0 else None)
    plt.bar(i, other, bottom=wf + dn, color="tab:gray",
            label="other sites" if i == 0 else None)
    plt.text(i, 1.02, f"{tot/1e6:.1f}M", ha="center", fontsize=7)
plt.xticks(range(len(names)), names, rotation=20, fontsize=7)
plt.ylabel("fraction of that model's changed codes")
plt.title("Where each model's code flips live")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig7_flip_composition.png"), dpi=150)
plt.close()

# Figure 8: HB vs low-LR overlap
ov = json.load(open(os.path.join(
    REPO, "results/qanchor_closure/selectivity/"
    "selectivity_closure.json")))["overlaps"]
ks = list(ov.keys())
plt.figure(figsize=(8, 4.2))
plt.bar(range(len(ks)), [ov[k]["jaccard_all"] for k in ks],
        color="tab:green")
plt.xticks(range(len(ks)), [k.replace("__vs__", "\nvs ") for k in ks],
           fontsize=7)
plt.ylabel("flipped-index Jaccard (all sites)")
plt.title("HB flip sets barely overlap plain low-LR flip sets")
for i, k in enumerate(ks):
    plt.text(i, ov[k]["jaccard_all"] + 0.005,
             f"{ov[k]['jaccard_all']:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig8_overlap.png"), dpi=150)
plt.close()
print("figures ->", OUT, sorted(os.listdir(OUT)))
