"""Analysis + plots (§22 survival, §31 figures, grid tables).

Reads shards/ and cycles/, writes:
  tables/al_summary.json         every arm x dataset cycle-pooled tau + macro
  tables/survival__<arms>.csv    P(L>=k) per arm (mtbench)
  plots/*.png/pdf                survival, grid heatmap, fc-fix ladder,
                                 branch stats, runtime breakdown
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_taus(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.extend(int(x) for x in r["taus"].split(";") if x)
    return rows


def all_shards(rd):
    out = {}
    for p in glob.glob(os.path.join(rd, "shards", "al__*.csv")):
        m = re.match(r"al__(.+)__(.+)__(.+)\.csv", os.path.basename(p))
        tag, tgt, ds = m.groups()
        out[(tag, tgt, ds)] = p
    return out


def survival(taus, B=10):
    n = len(taus)
    return {k: sum(t >= k for t in taus) / n for k in range(1, B + 1)}


def main(rd):
    os.makedirs(os.path.join(rd, "plots"), exist_ok=True)
    shards = all_shards(rd)
    summ = {}
    for (tag, tgt, ds), p in sorted(shards.items()):
        taus = load_taus(p)
        summ.setdefault(f"{tag}@{tgt}", {})[ds] = dict(
            tau=float(np.mean(taus)), n_cycles=len(taus))
    for k, v in summ.items():
        core = [v[d]["tau"] for d in ("mtbench", "gsm8k", "humaneval",
                                      "sharegpt") if d in v]
        if len(core) == 4:
            v["mean4ds"] = float(np.mean(core))
    json.dump(summ, open(os.path.join(rd, "tables", "al_summary.json"),
                         "w"), indent=1)

    # ---- survival curves (mtbench)
    arms = [("A0_T16D16", "fp16", "T16 D16 (fp16)"),
            ("G_T16D4", "fp16", "T16 D4 naive"),
            ("G_T16D4_p2", "fp16", "T16 D4 +P2"),
            ("IF_T16D4_rotH", "fp16", "T16 D4 +iface-rot"),
            ("G_T4D16", "w4a4", "T4 D16"),
            ("G_T4D4", "w4a4", "T4 D4 (folded)"),
            ("RC0", "w4a4", "T4 D4 +P2 (RC0)"),
            ("RC1", "w4a4", "T4 D4 +P2+R_C reuse (RC1)"),
            ("RC1xT8", "w8a8", "T8 D4 +P2+R_C reuse"),
            ("RC2", "w4a4", "T4 D4 +P2+R_C (RC2)")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for tag, tgt, label in arms:
        p = shards.get((tag, tgt, "mtbench"))
        if not p:
            continue
        s = survival(load_taus(p))
        ax.plot(list(s.keys()), list(s.values()), marker="o", ms=3,
                label=label)
        rows.append({"arm": label, **{f"P_ge_{k}": round(v, 4)
                                      for k, v in s.items()}})
    ax.set_xlabel("k (accepted tokens per block ≥ k)")
    ax.set_ylabel("P(L ≥ k)")
    ax.set_title("Block-position acceptance survival (MT-Bench, B=10)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(rd, "plots", f"survival_mtbench.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(rd, "tables", "survival_mtbench.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- 3x3 grid heatmap (4-ds mean)
    grid = np.full((3, 3), np.nan)
    best = {"fp16": "G_T16D4_p2", "w8a8": "RC1xT8", "w4a4": "RC1"}
    cells = {("fp16", 0): "A0_T16D16", ("fp16", 1): "G_T16D8",
             ("fp16", 2): "G_T16D4", ("w8a8", 0): "G_T8D16",
             ("w8a8", 1): "G_T8D8", ("w8a8", 2): "G_T8D4",
             ("w4a4", 0): "G_T4D16", ("w4a4", 1): "G_T4D8",
             ("w4a4", 2): "G_T4D4"}
    for (tgt, dj), tag in cells.items():
        ti = ["fp16", "w8a8", "w4a4"].index(tgt)
        v = summ.get(f"{tag}@{tgt}", {}).get("mean4ds")
        if v:
            grid[ti, dj] = v
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, cmap="viridis", vmin=1, vmax=4.5)
    for i in range(3):
        for j in range(3):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=12)
    # overlay best-recipe D4 values as a 4th column
    best_vals = [summ.get(f"{best[t]}@{t}", {}).get("mean4ds")
                 for t in ("fp16", "w8a8", "w4a4")]
    for i, v in enumerate(best_vals):
        if v:
            ax.text(2, i, f"{grid[i,2]:.2f}\n(best {v:.2f})", ha="center",
                    va="center", color="w", fontsize=9)
    ax.set_xticks(range(3), ["D16", "D8", "D4 (naive/folded)\n+best recipe"])
    ax.set_yticks(range(3), ["T16", "T8", "T4"])
    ax.set_title("DFlash 3x3 precision grid (4-dataset mean tau)")
    fig.colorbar(im)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(rd, "plots", f"grid_3x3.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- fc-fix ladder (mtbench-40 sensitivity arms)
    ladder = [("S16_fc", "naive"), ("S16_fc_mp3", "+MP3"),
              ("S16_fc_p2", "+P2"), ("S16_fc_rotH", "+rot"),
              ("S16_fc_rotH_mp3", "+rot+MP3"),
              ("S16_fc_rotH_p2", "+rot+P2"),
              ("S16_fc_rotH_p2_mp3", "+rot+P2+MP3")]
    vals, labs = [], []
    for tag, lab in ladder:
        p = shards.get((tag, "fp16", "mtbench"))
        if p:
            vals.append(np.mean(load_taus(p)))
            labs.append(lab)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(labs, vals, color="steelblue")
    ax.axvline(3.9, ls="--", c="gray", label="fp16 draft (~3.9)")
    ax.set_xlabel("cycle-pooled tau (W_c-only W4A4, MT-Bench-40)")
    ax.set_title("W_c quantization fixes")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(rd, "plots", f"fc_fix_ladder.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- runtime breakdown
    rts = sorted(glob.glob(os.path.join(rd, "tables", "runtime__*.json")))
    if rts:
        stages = ["prefill", "draft_forward", "target_verify",
                  "extract_concat", "ctx_transform", "embed_restore",
                  "head_restore"]
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(rts)
        for ai, p in enumerate(rts):
            r = json.load(open(p))
            vals = [r.get(s, {}).get("mean_ms", 0) for s in stages]
            ax.bar(np.arange(len(stages)) + ai * width, vals, width,
                   label=r["arm"])
        ax.set_xticks(np.arange(len(stages)) + 0.4, stages, rotation=30,
                      ha="right", fontsize=8)
        ax.set_ylabel("mean ms per call (fake-quant)")
        ax.set_title("Runtime breakdown (fake-quant timings)")
        ax.legend(fontsize=7)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(rd, "plots",
                                     f"runtime_breakdown.{ext}"), dpi=150,
                        bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({k: v.get("mean4ds") or
                      {d: round(x["tau"], 3) for d, x in v.items()
                       if isinstance(x, dict)}
                      for k, v in sorted(summ.items())}, indent=1))
    print("[analyze] DONE")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
