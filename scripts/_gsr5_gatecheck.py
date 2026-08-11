#!/usr/bin/env python
"""GS+R5 campaign gate verdicts.

Usage: _gsr5_gatecheck.py <run_dir> g     # gateG (fold parity) only
       _gsr5_gatecheck.py <run_dir> full  # gateG + B3R/B7R recon tolerance

Exits 1 (and does NOT write the PASS artifact) on any failure, so
scheduler deps on this job block downstream work. `full` writes
audit/parity_results.json used by the final report.
"""
import csv, json, os, sys

rd, mode = sys.argv[1], sys.argv[2]

CANON = {
    "B3R_T4": {"mtbench": 2.9955, "gsm8k": 3.4490, "sharegpt": 3.1092,
               "humaneval": 3.7490},
    "B7R_T4": {"mtbench": 3.2004, "gsm8k": 3.7390, "sharegpt": 3.3115,
               "humaneval": 3.8557},
}
TOL_DS, TOL_MEAN = 0.04, 0.02


def pooled_tau(shard):
    ts = [t for r in csv.DictReader(open(shard))
          for t in json.loads(r["acceptance_list"])]
    return sum(ts) / max(len(ts), 1)


fails, out = [], {}
gg = os.path.join(rd, "gradchecks", "gateG_gs_r5_parity.json")
if not os.path.exists(gg):
    fails.append("gateG json missing")
else:
    j = json.load(open(gg))
    out["gateG"] = dict(verdict=j["verdict"], fails=j["fails"])
    if j["verdict"] != "PASS":
        fails.append(f"gateG verdict {j['verdict']}: {j['fails']}")

if mode == "full":
    for tag, canon in CANON.items():
        rec = {}
        for ds, cv in canon.items():
            sh = os.path.join(rd, "shards",
                              f"al__{tag}__int4__{ds}.csv")
            if not os.path.exists(sh):
                fails.append(f"missing shard {sh}")
                continue
            tau = pooled_tau(sh)
            rec[ds] = dict(tau=round(tau, 4), canonical=cv,
                           delta=round(tau - cv, 4))
            if abs(tau - cv) > TOL_DS:
                fails.append(f"{tag}/{ds} tau {tau:.4f} vs canonical "
                             f"{cv:.4f} exceeds tol {TOL_DS}")
        if len(rec) == len(canon):
            m = sum(v["tau"] for v in rec.values()) / len(rec)
            cm = sum(canon.values()) / len(canon)
            rec["mean"] = dict(tau=round(m, 4), canonical=round(cm, 4),
                               delta=round(m - cm, 4))
            if abs(m - cm) > TOL_MEAN:
                fails.append(f"{tag} mean {m:.4f} vs {cm:.4f} exceeds "
                             f"tol {TOL_MEAN}")
        out[tag] = rec

out["fails"] = fails
out["verdict"] = "PASS" if not fails else "FAIL"
dst = os.path.join(rd, "audit",
                   "parity_results.json" if mode == "full"
                   else "gatecheck_g.json")
if fails:
    json.dump(out, open(dst + ".FAIL", "w"), indent=1)
    print(f"[gatecheck:{mode}] FAIL -> {dst}.FAIL")
    for f in fails:
        print(f"[gatecheck:{mode}] {f}")
    sys.exit(1)
json.dump(out, open(dst, "w"), indent=1)
print(f"[gatecheck:{mode}] PASS -> {dst}")
