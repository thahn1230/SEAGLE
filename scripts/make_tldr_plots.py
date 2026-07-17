#!/usr/bin/env python
"""Generate all TLDR study figures from run-dir tables/shards (CPU).

Each figure is skipped gracefully if its inputs are missing, so the
script can be re-run as results land. Writes PNGs to <run>/plots/.
"""
import csv, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RD = sys.argv[1]
PD = os.path.join(RD, "plots")
os.makedirs(PD, exist_ok=True)
DS = ["mtbench", "sharegpt", "c4", "gsm8k", "humaneval"]


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summary():
    return {r["shard"]: r for r in
            rows(os.path.join(RD, "tables", "micro_al_summary.csv"))}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PD, name), dpi=150)
    plt.close(fig)
    print(f"[plots] {name}")


S = summary()


def get(shard):
    r = S.get(shard)
    return (float(r["micro_al"]), float(r["ci_lo"]),
            float(r["ci_hi"])) if r else None


# 1. Causal panel per dataset (C0-C4)
PANEL = ["T16_IDENTITY_D8", "TR16_ROTATED_D8", "T8_ROTATED_D8",
         "TR16_RESTORED_D8", "T8_RESTORED_D8"]
LBL = ["C0 identity\nT16/D8", "C1 rot-FP16\nexposed",
       "C2 W8\nexposed", "C3 rot-FP16\nrestored", "C4 W8\nrestored"]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.4), sharey=True)
ok = False
for ax, ds in zip(axes, DS):
    vals = [get(f"panel__{ds}__{c}") for c in PANEL]
    if not all(vals):
        continue
    ok = True
    m = [v[0] for v in vals]
    err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
    ax.bar(range(5), m, yerr=err, capsize=3,
           color=["#888", "#c44", "#c44", "#4a4", "#4a4"])
    ax.set_xticks(range(5), LBL, fontsize=7)
    ax.set_title(ds)
axes[0].set_ylabel("micro-AL")
if ok:
    save(fig, "fig01_causal_panel.png")
else:
    plt.close(fig)

# 2. 4x4 KV4 matrix heatmap
T_ORD = ["T16_KV16", "T8_KV16", "T4_KV16", "T4_KV4"]
D_ORD = ["D16_KV16", "D8_KV16", "D4P3_KV16", "D4P3_KV4"]
M = np.full((4, 4), np.nan)
for i, t in enumerate(T_ORD):
    for j, d in enumerate(D_ORD):
        v = get(f"kv4mat__{t}__{d}")
        if v:
            M[i, j] = v[0]
if np.isfinite(M).all():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, cmap="viridis")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="w", fontsize=10)
    ax.set_xticks(range(4), D_ORD, rotation=20)
    ax.set_yticks(range(4), T_ORD)
    ax.set_xlabel("draft config"); ax.set_ylabel("target config")
    ax.set_title("micro-AL, MT-Bench 80 (Phase E)")
    fig.colorbar(im)
    save(fig, "fig02_kv4_matrix.png")

# 3. Context-length sensitivity
ctx_files = sorted(glob.glob(os.path.join(RD, "shards", "ctx__*.csv")))
if ctx_files:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for p in ctx_files:
        rr = rows(p)
        if not rr:
            continue
        L = [int(r["ctx_len"]) for r in rr]
        y = [float(r["micro_al"]) for r in rr]
        ax.plot(L, y, marker="o", label=rr[0]["config"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("micro-AL")
    ax.legend(fontsize=8); ax.set_title("Phase F: KV4 context scaling")
    save(fig, "fig03_context_sensitivity.png")

# 4. Rotation screening: candidates vs shared per dataset (t4)
tags = sorted({os.path.basename(p).split("__")[1]
               for p in glob.glob(os.path.join(RD, "shards",
                                               "drot__*__t4__*.csv"))})
if tags:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / max(len(tags), 1)
    for k, tag in enumerate(tags):
        vals = [get(f"drot__{tag}__t4__{d}") for d in DS]
        xs = [i + k * w for i, v in enumerate(vals) if v]
        ys = [v[0] for v in vals if v]
        ax.bar(xs, ys, width=w, label=tag)
    ax.set_xticks([i + 0.4 for i in range(len(DS))], DS)
    ax.set_ylabel("micro-AL (t4, 20 prompts)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Phase G screening: draft rotations vs shared R_T")
    save(fig, "fig04_rotation_screening_t4.png")

# 5. Transfer matrix (rotation x deployed target)
ROT = ["shared_RT", "spec_T8", "spec_T4", "spec_T4KV4", "mixed"]
TGT = ["t8", "t4", "t4kv4"]
TM = np.full((len(ROT), len(TGT)), np.nan)
for i, rot in enumerate(ROT):
    for j, tg in enumerate(TGT):
        vs = [get(f"drot__{rot}__{tg}__{d}") for d in DS]
        vs = [v[0] for v in vs if v]
        if len(vs) == len(DS):
            TM[i, j] = float(np.mean(vs))
if np.isfinite(TM).any():
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(TM, cmap="magma")
    for i in range(len(ROT)):
        for j in range(len(TGT)):
            if np.isfinite(TM[i, j]):
                ax.text(j, i, f"{TM[i, j]:.3f}", ha="center",
                        va="center", color="w", fontsize=9)
    ax.set_xticks(range(len(TGT)), TGT)
    ax.set_yticks(range(len(ROT)), ROT)
    ax.set_xlabel("deployed target"); ax.set_ylabel("draft rotation")
    ax.set_title("mean micro-AL over 5 datasets (screening)")
    fig.colorbar(im)
    save(fig, "fig05_transfer_matrix.png")

# 6. Objective ablation (wave1) on mtbench t4
abl = sorted({os.path.basename(p).split("__")[1]
              for p in glob.glob(os.path.join(
                  RD, "shards", "drot__*__t4__mtbench.csv"))})
if abl:
    pairs = [(t, get(f"drot__{t}__t4__mtbench")) for t in abl]
    pairs = [(t, v) for t, v in pairs if v]
    pairs.sort(key=lambda x: -x[1][0])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [t for t, _ in pairs]
    m = [v[0] for _, v in pairs]
    err = [[v[0] - v[1] for _, v in pairs],
           [v[2] - v[0] for _, v in pairs]]
    cols = ["#4a4" if t == "shared_RT" else "#48c" for t in names]
    ax.bar(range(len(m)), m, yerr=err, capsize=3, color=cols)
    ax.set_xticks(range(len(m)), names, rotation=30, ha="right",
                  fontsize=8)
    ax.set_ylabel("micro-AL (t4, mtbench-20)")
    ax.set_title("All rotation candidates vs shared R_T (green)")
    save(fig, "fig06_objective_ablation.png")

# 7. Incremental KV PPL (rows are config x kv_bits)
inc = rows(os.path.join(RD, "tables", "incremental_kv_ppl.csv"))
if inc:
    by = {}
    for r in inc:
        wa = r["config"].split("_kv")[0]
        by.setdefault(wa, {})[int(r["kv_bits"])] = float(
            r["incremental_ce"])
    labels = [w for w in by if 16 in by[w] and 4 in by[w]]
    if labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, [by[w][16] for w in labels], 0.4, label="KV16")
        ax.bar(x + 0.2, [by[w][4] for w in labels], 0.4, label="KV4")
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylabel("incremental CE (nats/token)"); ax.legend()
        ax.set_title("B2: incremental-decode CE, KV16 vs KV4")
        save(fig, "fig07_incremental_kv_ce.png")

# 8. tau histograms for the deployment diagonal
th = {r["shard"]: r for r in rows(os.path.join(RD, "tables",
                                               "tau_histograms.csv"))}
diag = ["kv4mat__T16_KV16__D16_KV16", "kv4mat__T8_KV16__D8_KV16",
        "kv4mat__T4_KV16__D4P3_KV16", "kv4mat__T4_KV4__D4P3_KV4"]
if all(d in th for d in diag):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(1, 8)
    for d in diag:
        y = [float(th[d][f"tau{t}"]) for t in x]
        ax.plot(x, y, marker="o",
                label=d.replace("kv4mat__", "").replace("__", "/"))
    ax.set_xlabel("tau (accepted+1)"); ax.set_ylabel("fraction")
    ax.legend(fontsize=8)
    ax.set_title("S27: tau distribution along deployment diagonal")
    save(fig, "fig08_tau_histograms.png")

# 9. Acceptance vs cycle bucket (KV4 error accumulation probe)
tb = {r["shard"]: r for r in rows(os.path.join(
    RD, "tables", "tau_by_cycle_bucket.csv"))}
if all(d in tb for d in diag):
    fig, ax = plt.subplots(figsize=(7, 4))
    buckets = ["cyc00_04", "cyc05_14", "cyc15plus"]
    x = np.arange(3)
    for d in diag:
        y = [float(tb[d][b]) for b in buckets if tb[d][b]]
        ax.plot(x[:len(y)], y, marker="s",
                label=d.replace("kv4mat__", "").replace("__", "/"))
    ax.set_xticks(x, ["cycles 0-4", "5-14", "15+"])
    ax.set_ylabel("micro-AL"); ax.legend(fontsize=8)
    ax.set_title("S27: acceptance vs generation progress")
    save(fig, "fig09_tau_by_cycle.png")

# 10. Verifier correctness distribution
vc = rows(os.path.join(RD, "tables", "verifier_correctness.csv"))
if vc:
    fig, ax = plt.subplots(figsize=(6.6, 4))
    cfgs = sorted({r["config"] for r in vc})
    data = [[float(r["frac"]) for r in vc if r["config"] == c]
            for c in cfgs]
    ax.boxplot(data, labels=cfgs)
    ax.set_ylabel("greedy prefix-match fraction (128 tok)")
    ax.set_title("S26: EAGLE output vs target-only AR greedy")
    save(fig, "fig10_verifier_correctness.png")

print("[plots] done")
