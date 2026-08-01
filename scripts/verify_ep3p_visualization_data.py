#!/usr/bin/env python
"""Verify EP3-P visualization data reproducibility (spec §14, §15.9).

Recomputes, from the saved raw tensors (plot_data/ep3p_tensors.pt) and
the saved plotted matrices (plot_data/*.npz), the statistics reported
in tables/, and checks:
  - plotted before/after matrices regenerate exactly from raw tensors
    with the recorded grouping/aggregation,
  - region statistics in the CSVs match a fresh recomputation,
  - FP output error matches the reported values,
  - every figure NPZ has a metadata JSON with a norm range that
    contains the matrix.
Writes metadata/verification.json; non-zero exit on any mismatch.
"""
import argparse, csv, json, os, sys

import numpy as np
import torch

D = 4096


def group_rows(A, g, agg):
    v = A.reshape(A.shape[0], A.shape[1] // g, g)
    return v.mean(-1) if agg == "meanabs" else v.max(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    blob = torch.load(os.path.join(rd, "plot_data",
                                   "ep3p_tensors.pt"),
                      map_location="cpu", weights_only=False)
    man = json.load(open(os.path.join(
        rd, "metadata", "collection_manifest.json")))
    viz = json.load(open(os.path.join(
        rd, "metadata", "visualization_manifest.json")))
    m_first = man["m_first"]
    ga, aa = viz["act_group"], viz["act_agg"]
    checks = {}

    # 1. plotted matrix regeneration (first before/after)
    Xb = blob["X_first_raw"].float().numpy()
    Xa = Xb.copy(); Xa[:, :D] *= m_first
    for name, X in (("first_projection_input_before_3d", Xb),
                    ("first_projection_input_after_3d", Xa)):
        Z = np.load(os.path.join(rd, "plot_data", name + ".npz"))["Z"]
        G = group_rows(np.abs(X), ga, aa)
        checks[f"regen_{name}"] = bool(np.allclose(Z, G, atol=1e-5))

    # 2. stats CSV recomputation (first path RMS ratios)
    tab = {r["metric"]: r["value"] for r in csv.DictReader(
        open(os.path.join(rd, "tables",
                          "ep3p_first_statistics.csv")))}

    def rms(v):
        return float(np.sqrt((np.abs(v).astype(np.float64) ** 2)
                             .mean()))
    ratio_b = rms(Xb[:, :D]) / rms(Xb[:, D:])
    ratio_a = rms(Xa[:, :D]) / rms(Xa[:, D:])
    checks["stats_before_e_over_h_rms"] = bool(np.isclose(
        ratio_b, float(tab["before_e_over_h_rms"]), rtol=1e-6))
    checks["stats_after_e_over_h_rms"] = bool(np.isclose(
        ratio_a, float(tab["after_e_over_h_rms"]), rtol=1e-6))
    checks["after_ratio_is_m_times_before"] = bool(np.isclose(
        ratio_a / ratio_b, m_first, rtol=1e-4))

    # 3. FP output error re-check
    W0 = blob["W_before"].float()
    Wf = W0.clone(); Wf[:, :D] /= m_first
    Yb = torch.from_numpy(Xb).float() @ W0.t()
    Ya = torch.from_numpy(Xa).float() @ Wf.t()
    if blob.get("bias") is not None:
        Yb = Yb + blob["bias"].float()
        Ya = Ya + blob["bias"].float()
    maxerr = float((Ya - Yb).abs().max())
    checks["fp_max_error_matches"] = bool(np.isclose(
        maxerr, float(tab["fp_max_abs_error"]), rtol=1e-3,
        atol=1e-6))

    # 4. every NPZ has metadata with a covering norm range
    import glob
    bad = []
    for p in glob.glob(os.path.join(rd, "plot_data", "*.npz")):
        if p.endswith("ep3p_tensors.pt"):
            continue
        mp = p[:-4] + "_metadata.json"
        if not os.path.exists(mp):
            bad.append(os.path.basename(p) + ": no metadata")
            continue
        md = json.load(open(mp))
        z = np.load(p)
        key = "Z" if "Z" in z else "Z_left"
        if "norm" in md and md.get("scale") == "linear":
            lo, hi = md["norm"]
            zz = z[key]
            if zz.min() < lo - 1e-6 or zz.max() > hi + 1e-6:
                bad.append(os.path.basename(p) + ": norm range does "
                           "not cover data")
    checks["all_npz_have_covering_metadata"] = not bad
    out = dict(checks=checks, metadata_issues=bad,
               recomputed=dict(first_rms_ratio_before=ratio_b,
                               first_rms_ratio_after=ratio_a,
                               fp_max_error=maxerr))
    json.dump(out, open(os.path.join(
        rd, "metadata", "verification.json"), "w"), indent=1)
    ok = all(checks.values())
    print(f"[verify] {'PASS' if ok else 'FAIL'}: "
          + json.dumps(checks))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
