#!/usr/bin/env python
"""Paired prompt-cluster bootstrap over cycle records (study §22).

Resamples PROMPTS (keeping all cycles of a prompt together), >=3000
reps; CIs for AL_q, AL_0, RCAL, SAL, LAL, AFS per method and paired
deltas (delta AL_q AND delta RCAL) for method pairs. Flags the
"deceptive / verifier-drift-driven AL gain" case only when CIs support
delta AL_q > 0 with delta RCAL <= 0.
"""
import argparse, glob, json, math, os, sys
import numpy as np


def load_by_prompt(rd, tag, ds="mtbench"):
    p = os.path.join(rd, "cycles", f"cyc__{tag}__{ds}.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for x in open(p):
        r = json.loads(x)
        out.setdefault(r["prompt_id"], []).append(
            (r["R_q"], r["R_0"], r["R_RC"]))
    return out


def agg(cl, keys):
    rq = r0 = rc = n = 0.0
    for k in keys:
        for (a, b, c) in cl[k]:
            rq += a; r0 += b; rc += c; n += 1
    if n == 0:
        return None
    alq, al0, rcal = rq / n, r0 / n, rc / n
    return dict(AL_q=alq, AL_0=al0, RCAL=rcal, SAL=alq - rcal,
                LAL=al0 - rcal,
                AFS=(2 * rcal / (alq + al0)) if alq + al0 > 0 else 0.0)


def ci(vals):
    return [float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tags", required=True,
                    help="comma list of cycle tags")
    ap.add_argument("--pairs", default="",
                    help="comma list A:B (delta = B - A)")
    ap.add_argument("--dataset", default="mtbench")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260731)
    rd = args.run_dir
    data = {}
    for t in args.tags.split(","):
        d = load_by_prompt(rd, t, args.dataset)
        if d:
            data[t] = d
    out = dict(methods={}, pairs={})
    for t, cl in data.items():
        keys = sorted(cl)
        point = agg(cl, keys)
        boots = {m: [] for m in point}
        idx = rng.integers(0, len(keys), size=(args.reps, len(keys)))
        for i in range(args.reps):
            s = agg(cl, [keys[j] for j in idx[i]])
            for m in point:
                boots[m].append(s[m])
        out["methods"][t] = {m: dict(point=point[m], ci=ci(boots[m]))
                             for m in point}
        print(f"[rcal-boot] {t}: RCAL {point['RCAL']:.4f} "
              f"{out['methods'][t]['RCAL']['ci']}")
    for pair in [p for p in args.pairs.split(",") if p]:
        a, _, b = pair.partition(":")
        if a not in data or b not in data:
            out["pairs"][pair] = dict(status="missing")
            continue
        common = sorted(set(data[a]) & set(data[b]))
        d_alq, d_rcal = [], []
        idx = rng.integers(0, len(common), size=(args.reps, len(common)))
        pa = agg(data[a], common)
        pb = agg(data[b], common)
        for i in range(args.reps):
            ks = [common[j] for j in idx[i]]
            sa, sb = agg(data[a], ks), agg(data[b], ks)
            d_alq.append(sb["AL_q"] - sa["AL_q"])
            d_rcal.append(sb["RCAL"] - sa["RCAL"])
        calq, crc = ci(d_alq), ci(d_rcal)
        deceptive = bool(calq[0] > 0 and crc[1] <= 0)
        out["pairs"][pair] = dict(
            n_prompts=len(common),
            delta_AL_q=pb["AL_q"] - pa["AL_q"], ci_delta_AL_q=calq,
            delta_RCAL=pb["RCAL"] - pa["RCAL"], ci_delta_RCAL=crc,
            deceptive_al_gain=deceptive)
        print(f"[rcal-boot] {pair}: dAL_q "
              f"{out['pairs'][pair]['delta_AL_q']:.4f} {calq} | dRCAL "
              f"{out['pairs'][pair]['delta_RCAL']:.4f} {crc}"
              + ("  [DECEPTIVE]" if deceptive else ""))
    os.makedirs(os.path.join(rd, "stats"), exist_ok=True)
    path = os.path.join(rd, "stats",
                        f"rcal_bootstrap_{args.dataset}.json")
    old = json.load(open(path)) if os.path.exists(path) else \
        dict(methods={}, pairs={})
    old["methods"].update(out["methods"])
    old["pairs"].update(out["pairs"])
    json.dump(old, open(path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
