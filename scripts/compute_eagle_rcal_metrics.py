#!/usr/bin/env python
"""RCAL metrics from captured cycle records (study §17-§18).

Proposal-only accepted lengths (no verifier bonus token; convention
audited in docs/EAGLE_ACCEPTANCE_LENGTH_CONTRACT.md). FP64 aggregation;
identities checked to 1e-12:

  AL_q = mean R_q     AL_0 = mean R_0     RCAL = mean R_RC
  SAL = AL_q - RCAL   LAL = AL_0 - RCAL
  P_A = RCAL/AL_q     R_A = RCAL/AL_0     AFS = 2*RCAL/(AL_q+AL_0)
  (ratios of aggregates; zero denominators -> metric = 0, recorded)

Depth-wise: S_q(k), S_0(k), S_RC(k) survival + spurious/lost curves,
first-divergence-depth histogram, branch-divergence rate.
"""
import argparse, glob, json, math, os, sys


def load(path):
    return [json.loads(x) for x in open(path)]


def metrics(recs, kmax=8):
    n = len(recs)
    if n == 0:
        return None
    Rq = [r["R_q"] for r in recs]
    R0 = [r["R_0"] for r in recs]
    RC = [r["R_RC"] for r in recs]
    alq = math.fsum(Rq) / n
    al0 = math.fsum(R0) / n
    rc = math.fsum(RC) / n
    sal, lal = alq - rc, al0 - rc
    pa = rc / alq if alq > 0 else 0.0
    ra = rc / al0 if al0 > 0 else 0.0
    afs = (2 * rc / (alq + al0)) if (alq + al0) > 0 else 0.0
    assert abs(alq - (rc + sal)) < 1e-12
    assert abs(al0 - (rc + lal)) < 1e-12
    assert abs((alq - al0) - (sal - lal)) < 1e-12
    surv = {k: dict(
        S_q=sum(x >= k for x in Rq) / n,
        S_0=sum(x >= k for x in R0) / n,
        S_RC=sum(x >= k for x in RC) / n) for k in range(1, kmax + 1)}
    for k in surv:
        surv[k]["spurious"] = surv[k]["S_q"] - surv[k]["S_RC"]
        surv[k]["lost"] = surv[k]["S_0"] - surv[k]["S_RC"]
    div = [r for r in recs if not r["same_branch"]]
    dd = {}
    for r in div:
        dd[r["div_depth"]] = dd.get(r["div_depth"], 0) + 1
    min_rel = sum(1 for r in recs
                  if r["R_RC"] == min(r["R_q"], r["R_0"])) / n
    return dict(n_cycles=n, AL_q=alq, AL_0=al0, RCAL=rc, SAL=sal,
                LAL=lal, P_A=pa, R_A=ra, AFS=afs,
                branch_divergence_rate=len(div) / n,
                min_relation_rate=min_rel,
                first_divergence_depth_hist=dd, depth_survival=surv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None,
                    help="single tag; default: all cycle files")
    args = ap.parse_args()
    rd = args.run_dir
    pats = ([os.path.join(rd, "cycles", f"cyc__{args.tag}__*.jsonl")]
            if args.tag else
            [os.path.join(rd, "cycles", "cyc__*.jsonl")])
    out = {}
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            b = os.path.basename(p)[5:-6]
            m = metrics(load(p))
            if m:
                out[b] = m
                print(f"[rcal] {b}: AL_q={m['AL_q']:.4f} "
                      f"AL_0={m['AL_0']:.4f} RCAL={m['RCAL']:.4f} "
                      f"SAL={m['SAL']:.4f} LAL={m['LAL']:.4f} "
                      f"AFS={m['AFS']:.4f}")
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    path = os.path.join(rd, "tables", "rcal_metrics.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    json.dump(old, open(path, "w"), indent=1)
    print(f"[rcal] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
