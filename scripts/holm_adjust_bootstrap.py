#!/usr/bin/env python
"""Holm-Bonferroni over bootstrap pair JSONs (this study's schema).

Reads stats/bootstrap_pairs_<dataset>.json files, groups comparisons
into pre-registered families (one family per dataset by default, or an
explicit --family name=pair1,pair2,...), applies Holm step-down on the
two-sided bootstrap p-values, writes stats/holm_adjusted.json.

p-values equal to 0.0 from the bootstrap are floored to 1/reps before
adjustment and reported as "p < 1/reps".
"""
import argparse, glob, json, os


def holm(items):
    """items: list of (name, p). Returns dict name -> (p_adj, reject05)."""
    m = len(items)
    s = sorted(items, key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (name, p) in enumerate(s):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # monotonicity
        out[name] = (running, running < 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--family", action="append", default=[],
                    help="name=comp1,comp2 (comparison names as in the "
                         "bootstrap json 'name' field, prefixed "
                         "<dataset>:) — default one family per dataset")
    args = ap.parse_args()
    floor = 1.0 / args.reps
    comps = {}
    for p in glob.glob(os.path.join(args.run_dir, "stats",
                                    "bootstrap_pairs_*.json")):
        ds = os.path.basename(p)[len("bootstrap_pairs_"):-len(".json")]
        for r in json.load(open(p)):
            if r.get("status") != "ok":
                continue
            pv = max(float(r["p_two_sided"]), floor)
            comps[f"{ds}:{r['name']}"] = dict(
                p_raw=pv, floored=r["p_two_sided"] < floor,
                delta=r["delta_b_minus_a"], ci=r["ci_delta"], dataset=ds)
    if args.family:
        fams = {}
        for f in args.family:
            name, _, rest = f.partition("=")
            fams[name] = [c for c in rest.split(",") if c in comps]
    else:
        fams = {}
        for k, v in comps.items():
            fams.setdefault(v["dataset"], []).append(k)
    out = {}
    for fname, members in fams.items():
        adj = holm([(k, comps[k]["p_raw"]) for k in members])
        for k, (pa, rej) in adj.items():
            out[k] = dict(comps[k], family=fname, p_holm=round(pa, 6),
                          reject_05=bool(rej),
                          p_display=(f"< {len(members)}/{args.reps}"
                                     if comps[k]["floored"] else
                                     f"{comps[k]['p_raw']:.4g} (holm "
                                     f"{pa:.4g})"))
    path = os.path.join(args.run_dir, "stats", "holm_adjusted.json")
    json.dump(out, open(path, "w"), indent=1)
    n_rej = sum(1 for v in out.values() if v["reject_05"])
    print(f"[holm] {len(out)} comparisons, {len(fams)} families, "
          f"{n_rej} rejected at 0.05 -> {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
