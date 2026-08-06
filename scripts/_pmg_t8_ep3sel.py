#!/usr/bin/env python
"""T8 EP3 selection: p3exp_search_w8a8.json (pairs25 by NMSE) ->
top-5 -> c4 calib-20 AL -> argmax. EP3-G = diagonal (beta_f==beta_r)
NMSE-sum argmin over the 25-pair grid's per-path grids.
Writes tables/ep3_selection_w8a8.json."""
import csv, json, os, subprocess, sys

rd = sys.argv[1]
D = 4096
out_s = json.load(open(os.path.join(rd, "tables",
                                    "p3exp_search_w8a8.json")))
pairs = out_s["pairs25"][:5]
res = []
for i, p in enumerate(pairs):
    mf, mr = D ** p["beta_first"], D ** p["beta_rec"]
    tag = f"T8SEL_{i}"
    subprocess.run(
        ["python", "scripts/eval_eagle_acceptance_length.py",
         "--target", "w8a8", "--draft-cfg", "d4p3_deploy",
         "--alpha", str(mf), "--alpha-rec", str(mr), "--tag", tag,
         "--datasets", "c4", "--pool", "calib", "--n-prompts", "20",
         "--run-dir", rd], check=True)
    sh = os.path.join(rd, "shards", f"al__{tag}__w8a8__c4__calib.csv")
    ts = [t for r in csv.DictReader(open(sh))
          for t in json.loads(r["acceptance_list"])]
    tau = sum(ts) / max(len(ts), 1)
    res.append(dict(p, m_f=mf, m_r=mr, calib_tau=round(tau, 4)))
    print(f"[t8sel] {p['beta_first']}/{p['beta_rec']} tau={tau:.4f}",
          flush=True)
best = max(res, key=lambda r: r["calib_tau"])
# EP3-G: common beta minimizing first+rec NMSE sum
pf = {r["beta"]: r["nmse"] for r in out_s["paths"]["first"]["grid"]}
pr = {r["beta"]: r["nmse"] for r in out_s["paths"]["rec"]["grid"]}
common = sorted(set(pf) & set(pr))
gbeta = min(common, key=lambda b: pf[b] + pr[b])
out = dict(ep3p=dict(beta_f=best["beta_first"],
                     beta_r=best["beta_rec"], m_f=best["m_f"],
                     m_r=best["m_r"], calib_tau=best["calib_tau"]),
           ep3g=dict(beta=gbeta, m=D ** gbeta,
                     nmse_sum=pf[gbeta] + pr[gbeta]),
           candidates=res,
           protocol="pairs25 NMSE top-5 -> c4 calib-20 AL argmax "
                    "(validation only); EP3-G = common-beta NMSE-sum "
                    "argmin (int4-protocol analogue)")
json.dump(out, open(os.path.join(rd, "tables",
                                 "ep3_selection_w8a8.json"), "w"),
          indent=1)
print(f"[t8sel] EP3-P ({best['beta_first']},{best['beta_rec']}) "
      f"tau {best['calib_tau']}; EP3-G beta {gbeta}")
