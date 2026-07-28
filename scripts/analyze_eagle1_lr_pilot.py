#!/usr/bin/env python
"""LR pilot selection (study spec §17): pick the peak LR per variant by
held-out calib acceptance length (tie-break: lower validation loss from
the training jsonl). Writes tables/lr_pilot.json."""
import argparse, csv, glob, json, os, sys


def tau_of(p):
    taus = [t for r in csv.DictReader(open(p))
            for t in json.loads(r["acceptance_list"])]
    return sum(taus) / max(len(taus), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    out = {}
    for var, tgt in (("Q0", "fp16"), ("Q1", "int4")):
        rows = []
        for lr in ("1e-06", "3e-06", "1e-05"):
            sh = os.path.join(
                rd, "shards",
                f"al__PILOT_{var}p_lr{lr}__{tgt}__c4__calib.csv")
            if not os.path.exists(sh):
                continue
            vl = None
            jl = os.path.join(rd, "logs", f"train_{var}p_lr{lr}.jsonl")
            if os.path.exists(jl):
                vals = [json.loads(x) for x in open(jl)
                        if '"kind": "val"' in x]
                if vals:
                    vl = vals[-1]["vloss"]
            rows.append(dict(lr=float(lr), calib_tau=round(tau_of(sh), 4),
                             final_vloss=vl))
        if rows:
            best = max(rows, key=lambda r: (r["calib_tau"],
                                            -(r["final_vloss"] or 9)))
            out[var] = dict(grid=rows, selected_lr=best["lr"])
    path = os.path.join(rd, "tables", "lr_pilot.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v["selected_lr"] for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
