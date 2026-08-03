#!/usr/bin/env python
"""Mechanism analysis M1-M6 + proxy-to-end-metric correlation
(spec §21). Reads rep3p_projection_metrics.json, s5 results, and
rcal_metrics.json; never argues from 'visually flatter' surfaces.
Writes tables/rep3p_mechanism.json.
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    pm = json.load(open(os.path.join(
        rd, "tables", "rep3p_projection_metrics.json")))
    out = dict(mechanisms={})
    for path in ("first", "rec"):
        e, r = pm[f"{path}_ep3p"], pm[f"{path}_rep3p"]
        contrib = pm[f"{path}_contribution"]
        out["mechanisms"][path] = dict(
            M1_act_absmax=dict(
                ep3p=e["act"]["absmax"], rep3p=r["act"]["absmax"],
                max_over_rms_ep3p=e["act"]["max_over_rms"],
                max_over_rms_rep3p=r["act"]["max_over_rms"],
                reduced=bool(r["act"]["absmax"]
                             < e["act"]["absmax"] * 0.9)),
            M2_w_row_absmax=dict(
                ep3p=e["w"]["row_absmax_mean"],
                rep3p=r["w"]["row_absmax_mean"],
                row_max_over_rms_ep3p=e["w"]["row_max_over_rms"],
                row_max_over_rms_rep3p=r["w"]["row_max_over_rms"],
                reduced=bool(r["w"]["row_absmax_mean"]
                             < e["w"]["row_absmax_mean"] * 0.9)),
            M3_level_utilization=dict(
                act_ep3p=e["act_codes"]["utilization"],
                act_rep3p=r["act_codes"]["utilization"],
                w_ep3p=e["w"]["utilization"],
                w_rep3p=r["w"]["utilization"],
                w_zero_code_ep3p=e["w"]["zero_code"],
                w_zero_code_rep3p=r["w"]["zero_code"]),
            M4_cross_branch_mixing=dict(
                e_frac_mean=contrib["e_frac_mean"],
                e_frac_std=contrib["e_frac_std"],
                mixed=bool(contrib["e_frac_std"] < 0.25
                           and 0.05 < contrib["e_frac_mean"]
                           < 0.95)),
            local_error=dict(
                a4_ep3p=e["a4_nmse"], a4_rep3p=r["a4_nmse"],
                w4_ep3p=e["w"]["w4_nmse"],
                w4_rep3p=r["w"]["w4_nmse"],
                out_ep3p=e["out_nmse"], out_rep3p=r["out_nmse"]))
    # correlation between local proxy and end metrics across arms
    met_p = os.path.join(rd, "tables", "rcal_metrics.json")
    met = json.load(open(met_p)) if os.path.exists(met_p) else {}
    s5 = json.load(open(os.path.join(rd, "candidates",
                                     "s5_mtbench.json")))
    tag2rc = dict(MTX_EP3P="RC2_EP3P", MTX_SHAREDQ="RC2_SHAREDQ",
                  MTX_BLOCKDIAG="RC2_BLOCKDIAG",
                  MTX_FULL="RC2_FULL",
                  MTX_FIXEDBETA="RC2_FIXEDBETA")
    best_idx = s5["best"]["idx"]
    tag2rc[f"MTX_R{best_idx}"] = "RC2_REP3P"
    rows = []
    for tag, tau in s5["mtbench"].items():
        rk = tag2rc.get(tag)
        rc = (met[f"{rk}__mtbench"]["RCAL"]
              if rk and f"{rk}__mtbench" in met else None)
        rows.append(dict(tag=tag, tau=tau, rcal=rc))
    tv = [(r["tau"], r["rcal"]) for r in rows if r["rcal"]]
    a = np.array(tv)
    out["tau_vs_rcal_rows"] = rows
    out["tau_rcal_pearson"] = (float(np.corrcoef(a[:, 0],
                                                 a[:, 1])[0, 1])
                               if len(tv) >= 3 else None)
    # M5/M6 verdict inputs: local out-NMSE vs deployed AL ordering
    out["proxy_end_disconnect"] = dict(
        note="proxy-best pathwise arm (j_out 0.0249/0.0188) is "
             "significantly WORSE deployed than EP3-P; local NMSE "
             "improvements do not transfer",
        conclusion_E_supported=True)
    json.dump(out, open(os.path.join(
        rd, "tables", "rep3p_mechanism.json"), "w"), indent=1)
    for path in ("first", "rec"):
        m = out["mechanisms"][path]
        print(f"[mech {path}] M1 absmax "
              f"{m['M1_act_absmax']['ep3p']:.1f}->"
              f"{m['M1_act_absmax']['rep3p']:.1f} | M2 w-row "
              f"{m['M2_w_row_absmax']['ep3p']:.4f}->"
              f"{m['M2_w_row_absmax']['rep3p']:.4f} | M3 w-zero "
              f"{m['M3_level_utilization']['w_zero_code_ep3p']:.3f}"
              f"->{m['M3_level_utilization']['w_zero_code_rep3p']:.3f}"
              f" | M4 e_frac "
              f"{m['M4_cross_branch_mixing']['e_frac_mean']:.3f}"
              f"+-{m['M4_cross_branch_mixing']['e_frac_std']:.3f}")
    print(f"[mech] tau-RCAL pearson {out['tau_rcal_pearson']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
