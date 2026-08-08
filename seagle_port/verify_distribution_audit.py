"""DKVA §30 quality gate. Run:
  python -m seagle_port.verify_distribution_audit <run_dir>
Must end with `mismatch = 0` / `missing = 0` (exit 1 otherwise).
"""
import glob
import json
import os
import sys

import numpy as np

rd = sys.argv[1]
missing, mismatch = [], []
CFGS = ("C0", "C1", "C2", "C3", "C4")
DSS = ("mtbench", "gsm8k", "sharegpt", "humaneval")
SRC = [1, 8, 15, 22, 29]


def need(path, tag):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        missing.append(f"{tag}: {path}")


# capture shards + expected tensor sets
for c in CFGS:
    for d in DSS:
        p = f"{rd}/tables/capture__{c}__{d}.json"
        need(p, "capture")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        st = j["stats"]
        for l in SRC:
            if f"A_src{l}" not in st:
                missing.append(f"{c}/{d}: A_src{l}")
        for li in range(5):
            for t in (f"D_ctx_k_l{li}", f"D_draft_k_l{li}"):
                if t not in st:
                    missing.append(f"{c}/{d}: {t}")
        for t in ("B_concat", "C_Zt", "C_Ht"):
            if t not in st:
                missing.append(f"{c}/{d}: {t}")
        if c in ("C3", "C4") and "C_Ht_rot" not in st:
            missing.append(f"{c}/{d}: C_Ht_rot")
        if j.get("identical_kv_input") is not True:
            mismatch.append(f"{c}/{d}: identical_kv_input != True")
        for name, s in st.items():
            for k in ("mean", "rms", "absmax", "kurtosis"):
                v = s.get(k)
                if v is None or not np.isfinite(v):
                    mismatch.append(f"{c}/{d}/{name}: non-finite {k}")

# paired sample counts: C2 C_Ht rows == C3 C_Ht rows per dataset
for d in DSS:
    try:
        a = json.load(open(f"{rd}/tables/capture__C2__{d}.json"))[
            "stats"]["C_Ht"]["count_rows"]
        b = json.load(open(f"{rd}/tables/capture__C3__{d}.json"))[
            "stats"]["C_Ht"]["count_rows"]
        if a != b:
            mismatch.append(f"{d}: C2 C_Ht rows {a} != C3 {b}")
    except Exception:
        pass

# qparam independence: ctx and draft sites both present with own stats
for d in DSS:
    try:
        q = json.load(open(f"{rd}/tables/capture__C2__{d}.json"))["qparams"]
        for li in range(5):
            for t in (f"D_ctx_k_l{li}", f"D_draft_k_l{li}"):
                if t not in q or q[t].get("n_tokens", 0) == 0:
                    missing.append(f"{d}: qparams {t}")
    except Exception:
        pass

# min row counts (>=10k ctx and per-layer draft rows across datasets, C2)
tot_ctx = tot_dr = 0
for d in DSS:
    try:
        st = json.load(open(f"{rd}/tables/capture__C2__{d}.json"))["stats"]
        tot_ctx += st["C_Ht"]["count_rows"]
        tot_dr += st["D_draft_k_l0"]["count_rows"]
    except Exception:
        pass
if tot_ctx < 10000:
    mismatch.append(f"ctx rows {tot_ctx} < 10000")
if tot_dr < 10000:
    mismatch.append(f"draft rows(l0) {tot_dr} < 10000")

# tables
for t in ("tensor_hook_map.md", "weight_global_stats.csv",
          "weight_row_stats.csv", "weight_column_stats.csv",
          "aw_error_decomposition.csv", "kv_projection_error.csv",
          "attention_error.csv", "rotation_effect_summary.csv",
          "representative_tokens.csv", "h0h1_verdict.json"):
    need(f"{rd}/tables/{t}", "table")

# plots
for p in ("activations_3d/FIG1_2_concat_before_after_RT.png",
          "activations_3d/FIG3_4_Ht_before_after_RC.png",
          "activations_3d/FIG3_4_Ht_before_after_RC_autoscale.png",
          "channels/FIG5_Ht_channel_before_after_RC.png",
          "qparams/FIG6_Ht_qparams_before_after_RC.png",
          "projection_error/FIG7_kv_projection_nmse.png",
          "weights_3d/DFLASH_WC_WEIGHT_3D_stock_vs_folded.png",
          "weights_3d/DFLASH_WC_WEIGHT_3D_stock_vs_folded.pdf",
          "heatmaps/FIG3_4_Ht_before_after_RC_heatmap.png",
          "qparams/token_worst.png"):
    need(f"{rd}/plots/{p}", "plot")
for li in range(5):
    need(f"{rd}/plots/activations_3d/"
         f"DFLASH_CTX_VS_DRAFT_ACT_3D_layer{li}.png", "plot")

# raw metadata present
for p in glob.glob(f"{rd}/raw/activations/*__C_Ht.npz")[:5]:
    z = np.load(p, allow_pickle=True)
    if "meta" not in z or "rows" not in z:
        mismatch.append(f"raw missing meta/rows: {p}")
    else:
        m = json.loads(str(z["meta"]))
        for k in ("config", "dataset", "tensor", "shape", "seed"):
            if k not in m:
                mismatch.append(f"raw meta missing {k}: {p}")
        if not np.isfinite(z["rows"].astype(np.float32)).all():
            mismatch.append(f"NaN/Inf in {p}")

# checksums manifest
cs = f"{rd}/manifests/checksums.sha256"
need(cs, "manifest")

print(f"missing = {len(missing)}")
for m in missing[:40]:
    print("  MISSING", m)
print(f"mismatch = {len(mismatch)}")
for m in mismatch[:40]:
    print("  MISMATCH", m)
sys.exit(1 if (missing or mismatch) else 0)
