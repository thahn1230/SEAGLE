"""§28 quality gate for the four-stage rotation visualization run.

Checks presence, pairing, same-Z discipline, layer/source coverage,
cache-labeling, rotation hashes, NaN/Inf, and raw-sample persistence.
Prints [OK]/[MISSING]/[MISMATCH] lines and a final
`missing=<n> mismatch=<n>`; exit 1 unless both are zero.
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np

DSS = ("mtbench", "gsm8k", "humaneval", "sharegpt")
NL = 5
MISS, MISM = [], []


def ok(label, detail=""):
    print(f"[OK]       {label} {detail}")


def miss(label, detail=""):
    MISS.append(label)
    print(f"[MISSING]  {label} {detail}")


def mism(label, detail=""):
    MISM.append(label)
    print(f"[MISMATCH] {label} {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    VR = args.run_dir

    # rotation hashes
    rot = json.load(open(f"{VR}/manifests/rotations.json"))
    for key in ("R1_T_R2_T", "R1_D_R2_D"):
        p = os.path.join("/home/thahn1230/dflash_workspace/dflash",
                         rot[key]["path"]) if not rot[key]["path"].startswith(
            "/") else rot[key]["path"]
        p2 = rot[key]["path"]
        for cand in (p, p2, f"/home/thahn1230/dflash_workspace/{p2}"):
            if os.path.exists(cand):
                sha = hashlib.sha256(open(cand, "rb").read()).hexdigest()
                (ok if sha == rot[key]["sha256"] else mism)(
                    f"rotation hash {key}", sha[:16])
                break
        else:
            miss(f"rotation file {key}")

    # stage stats presence per dataset
    for ds in DSS:
        p = f"{VR}/tables/capstats__{ds}.json"
        if not os.path.exists(p):
            miss(f"capstats {ds}")
            continue
        d = json.load(open(p))
        S = d["stats"]
        for nm in ("A_before_R1", "A_after_R1", "C_Ht_preRC",
                   "C_Ht_postRC", "C_Ht_FP"):
            (ok if nm in S else miss)(f"{ds} stats {nm}")
        n_cache = sum(1 for cond in ("FP", "QOFF", "QON")
                      for li in range(NL)
                      for kind in ("kctx", "vctx", "kdr", "vdr")
                      if f"{cond}_D_{kind}_l{li}" in S)
        (ok if n_cache == 60 else mism)(
            f"{ds} cache stats coverage", f"{n_cache}/60")
        for nm, st in S.items():
            for fld in ("rms", "absmax", "kurtosis"):
                v = st.get(fld)
                if v is not None and not np.isfinite(v):
                    mism(f"{ds} {nm} {fld} non-finite")

    # paired samples: identical shapes + identical manifest positions
    pairs = [("Hconcat_before_R1", "Hconcat_after_R1"),
             ("Ht_before_RC", "Ht_after_RC"),
             ("Ht_before_RC_A4dq", "Ht_after_RC_A4dq")]
    man = list(csv.DictReader(open(
        f"{VR}/tables/plot_sample_manifest.csv")))
    for ds in DSS:
        for a, b in pairs:
            pa = f"{VR}/raw_plot_samples/{ds}__{a}.npz"
            pb = f"{VR}/raw_plot_samples/{ds}__{b}.npz"
            if not (os.path.exists(pa) and os.path.exists(pb)):
                miss(f"sample pair {ds} {a}/{b}")
                continue
            A = np.load(pa)["rows"]
            B = np.load(pb)["rows"]
            (ok if A.shape == B.shape else mism)(
                f"pair shape {ds} {a}", f"{A.shape} vs {B.shape}")
            if not (np.isfinite(A).all() and np.isfinite(B).all()):
                mism(f"NaN/Inf in {ds} {a}/{b}")
            ma = [(r["prompt_id"], r["cycle_id"], r["token_position"])
                  for r in man if r["tensor"] == f"{ds}__{a}"]
            mb = [(r["prompt_id"], r["cycle_id"], r["token_position"])
                  for r in man if r["tensor"] == f"{ds}__{b}"]
            (ok if ma == mb and ma else mism)(
                f"pair positions identical {ds} {a}", f"n={len(ma)}")

    # cache samples off/on pairing (layers 0,2,4 sampled)
    for ds in DSS:
        for li in range(NL):
            for kind in ("kctx", "vctx", "kdr", "vdr"):
                pa = f"{VR}/raw_plot_samples/{ds}__L{li}_{kind}_cache_QOFF.npz"
                pb = f"{VR}/raw_plot_samples/{ds}__L{li}_{kind}_cache_QON.npz"
                (ok if os.path.exists(pa) and os.path.exists(pb)
                 else miss)(f"cache samples {ds} L{li} {kind}")

    # cache NMSE tables labeled FP vs W4A4
    for ds in DSS:
        p = f"{VR}/tables/cache_nmse__{ds}.csv"
        if not os.path.exists(p):
            miss(f"cache_nmse {ds}")
            continue
        rows = list(csv.DictReader(open(p)))
        conds = {r["condition"] for r in rows}
        (ok if conds == {"QOFF", "QON"} else mism)(
            f"cache NMSE conditions {ds}", str(conds))
        (ok if len({(r["layer"], r["kind"]) for r in rows}) == NL * 4
         else mism)(f"cache NMSE coverage {ds}")

    # plots + same-Z metadata
    md = json.load(open(f"{VR}/plots/plot_metadata.json"))
    for base in ("FIG_A_Hconcat_before_after_R1_sameZ",
                 "FIG_B_Wc_stock_vs_R1fold_sameZ",
                 "FIG_B_Wc_stockW4_vs_foldW4_sameZ",
                 "FIG_C_Ht_before_after_RC_sameZ",
                 "FIG_C2_Ht_A4_before_after_RC_sameZ"):
        (ok if os.path.exists(f"{VR}/plots/{base}.png")
         and os.path.exists(f"{VR}/plots/{base}.pdf")
         else miss)(f"plot {base}")
        (ok if base in md and md[base]["zmax"] > 0 else miss)(
            f"zmax metadata {base}")
    n_d = len(glob.glob(f"{VR}/plots/FIG_D_L*_cache_RCoff_on_sameZ.png"))
    (ok if n_d >= 20 else mism)("FIG_D cache figures", f"{n_d} >= 20")
    (ok if os.path.exists(
        f"{VR}/plots/FIG_ROTATION_FOUR_STAGE_SUMMARY.pdf") else miss)(
        "four-stage summary figure")
    (ok if glob.glob(f"{VR}/plots/WORST_CASE_DIAGNOSTIC*") else miss)(
        "worst-case diagnostic (labeled)")
    for t in ("tensor_location_map.md", "kv_cache_write_map.md",
              "fourstage_rotation_stats.csv",
              "fourstage_rotation_delta.csv", "plot_sample_manifest.csv"):
        (ok if os.path.exists(f"{VR}/tables/{t}") else miss)(f"table {t}")

    print(f"missing={len(MISS)} mismatch={len(MISM)}")
    raise SystemExit(1 if (MISS or MISM) else 0)


if __name__ == "__main__":
    main()
