"""Four-stage viz: aggregate tables, head-wise heatmaps, write maps.

Produces (under --run-dir):
  tables/fourstage_rotation_stats.csv   (§23)
  tables/fourstage_rotation_delta.csv   (§24)
  tables/tensor_location_map.md         (§0/§29)
  tables/kv_cache_write_map.md          (§13)
  plots/FIG_E_headwise_*.png            (§15 per-head heatmaps)
Reads capstats__<ds>.json + cache_nmse__<ds>.csv from fourstage_capture.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DSS = ("mtbench", "gsm8k", "humaneval", "sharegpt")
NL = 5


def g(d, *ks, default=""):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    VR = args.run_dir
    caps = {}
    for ds in DSS:
        p = f"{VR}/tables/capstats__{ds}.json"
        if os.path.exists(p):
            caps[ds] = json.load(open(p))

    rows = []

    def add(stage, tensor, cond, ds, layer, st, qpr=None, kv4=""):
        rows.append({
            "stage": stage, "tensor": tensor, "condition": cond,
            "dataset": ds, "layer": layer,
            "rms": g(st, "rms"), "absmax": g(st, "absmax"),
            "p99": g(st, "p99_abs", default=g(st, "p99")),
            "p99_9": g(st, "p99_9_abs", default=g(st, "p99.9")),
            "kurtosis": g(st, "kurtosis"),
            "top01pct_energy": g(st, "channel", "top0.1pct_energy"),
            "a4_nmse": g(qpr or {}, "nmse_mean"),
            "w4_nmse": "NA", "kv4_nmse": kv4,
            "sqnr": ("" if not g(qpr or {}, "nmse_mean") else
                     round(-10 * np.log10(g(qpr or {}, "nmse_mean")
                                          + 1e-30), 2))})

    for ds, d in caps.items():
        S, Q = d["stats"], d["qparams"]
        for nm, stage, cond in (("A_before_R1", "A", "no_R1T"),
                                ("A_after_R1", "A", "R1T"),
                                ("C_Ht_preRC", "C", "QON_preRC"),
                                ("C_Ht_postRC", "C", "QON_postRC"),
                                ("C_Ht_FP", "C", "FP")):
            if nm in S:
                add(stage, nm, cond, ds, -1, S[nm], Q.get(nm))
        for cond in ("FP", "QOFF", "QON"):
            for li in range(NL):
                for kind in ("kctx", "vctx", "kdr", "vdr"):
                    nm = f"{cond}_D_{kind}_l{li}"
                    if nm in S:
                        kv4 = g(Q, f"KV4_D_{kind}_l{li}", "nmse_mean") \
                            if cond == "FP" else ""
                        add("D", f"D_{kind}", cond, ds, li, S[nm],
                            None, kv4)
    with open(f"{VR}/tables/fourstage_rotation_stats.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # -------- delta table (§24) pooled across datasets (mean of ds values)
    def mean_over_ds(tensor_prefix, cond, field, layer=-1):
        vals = []
        for r in rows:
            if (r["tensor"].startswith(tensor_prefix)
                    and r["condition"] == cond
                    and r["layer"] == layer and r[field] != ""):
                vals.append(float(r[field]))
        return float(np.mean(vals)) if vals else None

    nmse = {}
    for ds in DSS:
        p = f"{VR}/tables/cache_nmse__{ds}.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                k = (r["condition"], int(r["layer"]), r["kind"])
                nmse.setdefault(k, []).append(float(r["nmse"]))
    dl = []

    def drow(stage, bef, aft, b_am, a_am, b_ku, a_ku, b_q, a_q):
        fmt = lambda x: ("" if x is None else round(x, 4))
        dl.append({"Stage": stage, "Before": bef, "After": aft,
                   "absmax_before": fmt(b_am), "absmax_after": fmt(a_am),
                   "kurtosis_before": fmt(b_ku), "kurtosis_after": fmt(a_ku),
                   "quantNMSE_before": fmt(b_q), "quantNMSE_after": fmt(a_q)})
    drow("concat H_i", "no R1_T", "+R1_T",
         mean_over_ds("A_before", "no_R1T", "absmax"),
         mean_over_ds("A_after", "R1T", "absmax"),
         mean_over_ds("A_before", "no_R1T", "kurtosis"),
         mean_over_ds("A_after", "R1T", "kurtosis"),
         mean_over_ds("A_before", "no_R1T", "a4_nmse"),
         mean_over_ds("A_after", "R1T", "a4_nmse"))
    drow("H_t", "no R_C", "+R_C",
         mean_over_ds("C_Ht_preRC", "QON_preRC", "absmax"),
         mean_over_ds("C_Ht_postRC", "QON_postRC", "absmax"),
         mean_over_ds("C_Ht_preRC", "QON_preRC", "kurtosis"),
         mean_over_ds("C_Ht_postRC", "QON_postRC", "kurtosis"),
         mean_over_ds("C_Ht_preRC", "QON_preRC", "a4_nmse"),
         mean_over_ds("C_Ht_postRC", "QON_postRC", "a4_nmse"))
    for kind, lab in (("kctx", "ctx K cache"), ("vctx", "ctx V cache")):
        off = np.mean([np.mean(v) for (c, l, k), v in nmse.items()
                       if c == "QOFF" and k == kind])
        on = np.mean([np.mean(v) for (c, l, k), v in nmse.items()
                      if c == "QON" and k == kind])
        drow(lab, "R_C OFF", "R_C ON", None, None, None, None,
             float(off), float(on))
    with open(f"{VR}/tables/fourstage_rotation_delta.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dl[0].keys()))
        w.writeheader()
        w.writerows(dl)

    # -------- §15 head-wise heatmaps from headrms channel stats
    for li in range(NL):
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for col, kind in enumerate(("kctx", "vctx", "kdr", "vdr")):
            for r_i, cond in enumerate(("QOFF", "QON")):
                vals = []
                for ds in DSS:
                    st = caps.get(ds, {}).get("stats", {}).get(
                        f"{cond}_D_{kind}_l{li}_headrms")
                    if st and "channel" in st:
                        pass
                    ch = caps.get(ds, {}).get("stats", {}).get(
                        f"{cond}_D_{kind}_l{li}_headrms", {})
                    m = ch.get("mean")
                    if m is not None:
                        vals.append([ch.get("mean"), ch.get("rms"),
                                     ch.get("absmax")])
                A = np.array(vals) if vals else np.zeros((1, 3))
                im = axes[r_i, col].imshow(A, aspect="auto", cmap="viridis")
                axes[r_i, col].set_title(f"L{li} {kind} {cond}",
                                         fontsize=8)
                axes[r_i, col].set_xticks([0, 1, 2])
                axes[r_i, col].set_xticklabels(["mean", "rms", "absmax"],
                                               fontsize=7)
                axes[r_i, col].set_yticks(range(len(DSS)))
                axes[r_i, col].set_yticklabels(DSS, fontsize=7)
                plt.colorbar(im, ax=axes[r_i, col], fraction=0.04)
        fig.suptitle(f"Layer {li} head-RMS summary (per dataset)")
        fig.tight_layout()
        fig.savefig(f"{VR}/plots/FIG_E_headwise_L{li}.png", dpi=200)
        plt.close(fig)

    # -------- location + write maps
    open(f"{VR}/tables/tensor_location_map.md", "w").write("""# Tensor locations (basis-audit-verified)
| item | value |
|---|---|
| target hidden layer IDs | [1, 8, 15, 22, 29] (H1..H5) |
| hidden size / concat width | 4096 / 20480 |
| W_c shape | fc.weight [4096, 20480] (dflash/model.py:317) |
| pre-R1 H_i | rot-target hidden_states[l+1] @ R1_T^T (exact orthogonal unrotation; basis audit S1) |
| post-R1 H_i | rot_fp16/w4a4 target output.hidden_states[l+1] (deployed basis) |
| stock W_c | DFlashDraftModel.fc.weight (z-lab ckpt) |
| folded W_c | interfaces.fold_wc: per-block Wc_i @ R1_T (fp64 staging) |
| pre-R_C H_t | RotQuantDraft forward tap S3_Ht_dep (bare-rms, ORIGINAL basis — fold cancels R1_T) |
| post-R_C H_t | tap S3_Ht_rc_dep = deployed `Ht = Ht @ rc_matrix_buf` (class A, runtime) |
| draft layers | 5 |
| sample seed / plot rows | 0 / 256 pooled (evenly-spaced, pre-registered rule) |
""")
    open(f"{VR}/tables/kv_cache_write_map.md", "w").write("""# K/V cache write map (audited)
Single write point per layer: dflash/model.py:238 `past_key_values.update(...)`.
| tensor | source | norm/RoPE state | dtype | persistent? |
|---|---|---|---|---|
| context K | k_proj(H_t[@R_C]) rows [:prefix_len] | post k_norm, post RoPE | bf16 | YES (grows monotonically) |
| context V | v_proj(H_t[@R_C]) rows [:prefix_len] | raw projection (no norm/RoPE) | bf16 | YES |
| draft K | k_proj(input_ln(h)) rows [prefix_len:] | post k_norm, post RoPE | bf16 | NO — crop(start) discards every cycle (model.py:120) |
| draft V | v_proj(input_ln(h)) rows [prefix_len:] | raw | bf16 | NO — transient |
Capture taps: RotQuantDraft S4_k_stored_l{i}/S4_v_stored_l{i} (regression-gated
bitwise-neutral; exactly the tensors passed to the cache write in the stock
path). Flattening for plots: channel = head_index*128 + head_dim.
""")
    print(f"[finalize] stats rows={len(rows)} delta rows={len(dl)}")


if __name__ == "__main__":
    main()
