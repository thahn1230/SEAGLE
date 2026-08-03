#!/usr/bin/env python
"""Mechanism + proxy-vs-end analysis for the LRGF study (§ machine):
correlates training objective, projection NMSE, held-out AL, and
validation RCAL across all trained arms; reports the min-NMSE
counterfactual vs the max-RCAL selection (spec §22).
Writes tables/lrgf_mechanism.json.
"""
import argparse, glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    held = json.load(open(os.path.join(
        rd, "tables", "learned_rot_heldout.json")))
    met_p = os.path.join(rd, "tables", "rcal_metrics.json")
    met = json.load(open(met_p)) if os.path.exists(met_p) else {}
    rows = []
    for jf in glob.glob(os.path.join(rd, "rotations", "*_last.pt")):
        tag = os.path.basename(jf)[:-8]
        if tag == "SMOKE":
            continue
        meta_p = os.path.join(rd, "rotations", f"{tag}.json")
        meta = (json.load(open(meta_p))
                if os.path.exists(meta_p) else {})
        hist = meta.get("hist", [])
        rcal = None
        mk = f"VLR_{tag}__c4"
        if mk in met:
            rcal = met[mk]["RCAL"]
        rows.append(dict(
            tag=tag,
            objective=tag.split("_")[0],
            final_loss=hist[-1]["loss"] if hist else None,
            nmse_part=hist[-1].get("nmse") if hist else None,
            heldout_tau=held.get(f"LR_{tag}"),
            val_rcal=rcal, orth=meta.get("orth")))
    refs = {k: v for k, v in held.items() if k.startswith("LRREF")}
    ok = [r for r in rows if r["heldout_tau"] is not None]
    sel_rcal = max((r for r in ok if r["val_rcal"] is not None),
                   key=lambda r: r["val_rcal"], default=None)
    out = dict(arms=rows, references=refs,
               selected_max_rcal=sel_rcal,
               selected_max_heldout_tau=max(
                   ok, key=lambda r: r["heldout_tau"],
                   default=None))
    tv = [(r["heldout_tau"], r["val_rcal"]) for r in ok
          if r["val_rcal"] is not None]
    if len(tv) >= 3:
        a = np.array(tv)
        out["tau_rcal_pearson"] = float(
            np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    json.dump(out, open(os.path.join(
        rd, "tables", "lrgf_mechanism.json"), "w"), indent=1)
    for r in sorted(ok, key=lambda r: -(r["heldout_tau"] or 0)):
        print(f"[mech] {r['tag']:28s} heldout {r['heldout_tau']} "
              f"rcal {r['val_rcal']}")
    print(f"[mech] refs {refs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
