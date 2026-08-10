#!/usr/bin/env python
"""Tables for the GS R1/R2 factorial study (spec sections 13/14/21/24).

Reads eval shards + bootstrap/holm JSONs + rcal/runtime/geometry outputs
and writes:

  tables/al_by_dataset.csv       every evaluated tag x dataset micro-tau
  tables/final_summary.json      2x2 arms, deltas, interaction, mean4
  tables/bootstrap_summary.csv   all paired contrasts + CI + p + Holm
  tables/rcal_summary.csv        AL_q/AL_0/RCAL/SAL/LAL/AFS per arm
  tables/rotation_geometry.csv   R1/R2 geometry per checkpoint
  tables/runtime_foldability.csv per-arm timing + operator audit

--arm-map maps canonical arm names to eval tags (median-val seeds), e.g.
  A0=A0_BASE,A1=A1_s1002,A2=A2_s1001,A3=A3_s1003
"""
import argparse, csv, glob, json, os, sys

import torch

DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]


def shard_tau(path):
    taus = []
    with open(path) as f:
        for r in csv.DictReader(f):
            taus += json.loads(r["acceptance_list"])
    return (sum(taus) / len(taus) if taus else None), len(taus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm-map", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    tdir = os.path.join(rd, "tables")
    os.makedirs(tdir, exist_ok=True)
    amap = dict(kv.split("=") for kv in args.arm_map.split(","))

    # ---- al_by_dataset.csv (ALL tags, canonical arms resolved) -----------
    rows = []
    for p in sorted(glob.glob(os.path.join(rd, "shards",
                                           "al__*__int4__*.csv"))):
        base = os.path.basename(p)[4:-4]
        parts = base.split("__")
        tag, ds = parts[0], parts[2]
        if len(parts) > 3:              # calib-pool shard
            continue
        tau, ncyc = shard_tau(p)
        arm = next((a for a, t in amap.items() if t == tag), tag)
        rows.append([arm, tag, ds, round(tau, 5) if tau else None, ncyc])
    with open(os.path.join(tdir, "al_by_dataset.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "tag", "dataset", "tau", "n_cycles"])
        w.writerows(rows)

    tau_of = {}
    for arm, tag, ds, tau, _n in rows:
        if tau is not None:
            tau_of[(tag, ds)] = tau

    def arm_tau(a, ds):
        return tau_of.get((amap[a], ds))

    # ---- 2x2 + deltas + interaction --------------------------------------
    table = {}
    for a in ("A0", "A1", "A2", "A3"):
        per = {ds: arm_tau(a, ds) for ds in DS}
        vals = [v for v in per.values() if v is not None]
        per["mean4"] = (round(sum(vals) / len(vals), 5)
                        if len(vals) == len(DS) else None)
        table[a] = per

    def delta(a, b):        # a - b per dataset (+mean4)
        out = {}
        for ds in DS + ["mean4"]:
            va, vb = table[a].get(ds), table[b].get(ds)
            out[ds] = (round(va - vb, 5)
                       if va is not None and vb is not None else None)
        return out

    deltas = {
        "R1_effect (A1-A0)": delta("A1", "A0"),
        "R2_effect (A2-A0)": delta("A2", "A0"),
        "R2_after_R1 (A3-A1)": delta("A3", "A1"),
        "R1_after_R2 (A3-A2)": delta("A3", "A2"),
    }
    inter = {}
    for ds in DS + ["mean4"]:
        vs = [table[a].get(ds) for a in ("A0", "A1", "A2", "A3")]
        inter[ds] = (round(vs[3] - vs[1] - vs[2] + vs[0], 5)
                     if all(v is not None for v in vs) else None)
    deltas["interaction (A3-A1-A2+A0)"] = inter

    # ---- bootstrap + holm summary ----------------------------------------
    brows = []
    holm_p = os.path.join(rd, "stats", "holm_adjusted.json")
    holm = json.load(open(holm_p)) if os.path.exists(holm_p) else {}
    if isinstance(holm, list):        # list-of-rows schema
        holm = {f"{r['dataset']}:{r['name']}": r for r in holm}
    for p in sorted(glob.glob(os.path.join(rd, "stats",
                                           "bootstrap_pairs_*.json"))):
        ds = os.path.basename(p)[16:-5]
        d = json.load(open(p))
        rows = d if isinstance(d, list) else [
            dict(r, name=k) for k, r in d.items() if isinstance(r, dict)]
        for r in rows:
            if r.get("status") != "ok":
                continue
            h = holm.get(f"{ds}:{r.get('name')}", {})
            ci = r.get("ci_delta", [None, None])
            brows.append([r.get("name"), ds, r.get("tau_a"),
                          r.get("tau_b"),
                          r.get("delta_b_minus_a", r.get("delta")),
                          ci[0], ci[1],
                          r.get("p_two_sided"),
                          h.get("p_display", ""),
                          h.get("reject_05", "")])
    with open(os.path.join(tdir, "bootstrap_summary.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "dataset", "tau_a", "tau_b", "delta",
                    "ci_lo", "ci_hi", "p_raw", "p_holm_display",
                    "reject_05"])
        w.writerows(brows)

    # ---- rcal ------------------------------------------------------------
    rcal_p = os.path.join(rd, "tables", "rcal_metrics.json")
    if os.path.exists(rcal_p):
        rc = json.load(open(rcal_p))
        with open(os.path.join(tdir, "rcal_summary.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "AL_q", "AL_0", "RCAL", "SAL", "LAL",
                        "P_A", "R_A", "AFS", "n_cycles"])
            for tag, m in sorted(rc.items()):
                if not isinstance(m, dict) or "AL_q" not in m:
                    continue
                w.writerow([tag] + [round(m.get(k, 0), 5) for k in
                                    ("AL_q", "AL_0", "RCAL", "SAL",
                                     "LAL", "P_A", "R_A", "AFS")]
                           + [m.get("n")])

    # ---- rotation geometry ----------------------------------------------
    grows = []
    for ck in sorted(glob.glob(os.path.join(rd, "rotations",
                                            "*.pt"))):
        if ck.endswith(".final.pt"):
            continue
        try:
            d = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(d, dict) or "meta" not in d:
            continue
        m = d["meta"]
        g1 = m.get("geometry", {})
        g2 = m.get("r2_geometry") or {}
        cp = m.get("checkpoint_policy", {})
        grows.append([os.path.basename(ck)[:-3],
                      cp.get("policy"), cp.get("saved_step"),
                      g1.get("geodesic_dist"), g1.get("frob_dist"),
                      g1.get("orth_error"),
                      g2.get("geodesic_dist"), g2.get("frob_dist"),
                      g2.get("max_elem_change"), g2.get("orth_error"),
                      g2.get("generator_frob_norm")])
    with open(os.path.join(tdir, "rotation_geometry.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "policy", "saved_step",
                    "r1_geodesic", "r1_frob", "r1_orth",
                    "r2_geodesic", "r2_frob", "r2_max_elem", "r2_orth",
                    "r2_generator_frob"])
        w.writerows(grows)

    # ---- runtime/foldability --------------------------------------------
    rrows = []
    for p in sorted(glob.glob(os.path.join(rd, "tables",
                                           "runtime__*.json"))):
        d = json.load(open(p))
        rrows.append([os.path.basename(p)[9:-5],
                      d.get("draft_ms_per_cycle"),
                      d.get("verify_ms_per_cycle"),
                      d.get("postproj_ms_per_cycle"),
                      d.get("ms_per_token"),
                      d.get("cycles"), d.get("tokens")])
    if rrows:
        with open(os.path.join(tdir, "runtime_foldability.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "draft_ms_cycle", "verify_ms_cycle",
                        "post_r1_ms_call", "ms_per_token", "cycles",
                        "tokens"])
            w.writerows(rrows)

    summary = dict(arm_map=amap, table_2x2=table, deltas=deltas,
                   n_eval_rows=len(rows))
    json.dump(summary, open(os.path.join(tdir, "final_summary.json"),
                            "w"), indent=1)
    print(json.dumps(dict(table_2x2=table, deltas=deltas), indent=1))
    print(f"[tables] -> {tdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
