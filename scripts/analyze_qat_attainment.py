#!/usr/bin/env python
"""Assemble the PTQ-vs-QAT study results: primary C1-C14/N1-N5 table,
attainment metrics (spec section 13), pre-registered non-inferiority
criteria (section 14), ceilings (section 12), decision logic (section 22)
and the secondary tables that derive from acceptance shards.

Robust to missing cells (reports status per cell); rerun any time.
Writes tables/*.csv|json and stats/verdicts.json.
"""
import argparse, glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_micro_al import read_shard, pooled_mal

# study-ID -> (tag, target). N4/N5 are definitional aliases of the P3
# shared-R_T and P3 local-R_D cells (recorded as such in the table).
CELLS = {
    "C1": ("C1_fp16_stock", "fp16", "stock FP16 draft", "none", "none"),
    "C2": ("C2_fp16_d4p3", "fp16", "strict TF PTQ D4P3", "none", "none"),
    "C3": ("C3_s{s}_qat", "fp16", "INT4 QAT draft", "QAT", "none"),
    "C4": ("C4_int4_stockrestored", "int4", "stock FP16 draft (restored)",
           "none", "none"),
    "C5": ("C5_int4_d4p3", "int4", "strict TF PTQ D4P3", "none", "none"),
    "C6": ("C6_tgtadapt", "int4", "INT4-target-adapted FP16 draft",
           "FP16 retrain", "none"),
    "C7": ("C7_s{s}_qat", "int4", "INT4 QAT draft", "QAT", "none"),
    "C7b": ("C7b_qat", "int4", "C6-init INT4 QAT", "QAT", "none"),
    "C8": ("C8_fp16retrain", "fp16", "FP16-retrained draft",
           "FP16 retrain", "none"),
    "C9": ("C9_c8ptq", "fp16", "C8 then D4P3 PTQ", "FP16 retrain",
           "none"),
    "C10": ("C10_c6ptq", "int4", "C6 then D4P3 PTQ", "FP16 retrain",
            "none"),
    "C11": ("C11_fp16_rot", "fp16", "rotation-optimized PTQ (R_D)",
            "none", "R_D"),
    "C12": ("C12_int4_rot", "int4", "rotation-optimized PTQ (R_D)",
            "none", "R_D"),
    "C13": ("C13_c8_at_int4", "int4", "FP16-teacher FP16 draft @ INT4",
            "FP16 retrain", "none"),
    "C14": ("C14_c3_at_int4", "int4", "FP16-teacher QAT draft @ INT4",
            "QAT", "none"),
    "N1": ("N1_fp16_naive", "fp16", "naive W4A4, no P3", "none", "none"),
    "N2": ("N2_int4_naive", "int4", "naive W4A4, no P3", "none", "none"),
    "N3": ("N3_fp16_p2", "fp16", "P2 branchwise scales, no P3", "none",
           "none"),
    "N3b": ("N3b_int4_p2", "int4", "P2 branchwise scales, no P3",
            "none", "none"),
    "N4": ("C5_int4_d4p3", "int4", "P3 with shared R_T (== C5)", "none",
           "none"),
    "N5": ("C12_int4_rot", "int4", "P3 with local R_D (== C12)", "none",
           "R_D"),
}
SEEDS = (0, 1, 2)


def shard(rd, tag, tgt, ds="mtbench"):
    p = os.path.join(rd, "shards", f"al__{tag}__{tgt}__{ds}.csv")
    return p if os.path.exists(p) else None


def clusters(p):
    return [(pid, taus) for pid, taus, _ in read_shard(p)]


def tau_of(rd, tag, tgt, ds="mtbench"):
    p = shard(rd, tag, tgt, ds)
    if p is None:
        return None, None
    c = clusters(p)
    return pooled_mal(c), c


def seed_mean_delta_ci(base_c, seed_cs, rng, reps=3000):
    """CI of mean-over-seeds paired delta (QAT - base), prompt-cluster
    bootstrap with the SAME prompt resample across seeds."""
    kb = {k: v for k, v in base_c}
    keys = sorted(set(kb) & set.intersection(
        *[set(k for k, _ in c) for c in seed_cs]))
    ks = [{k: v for k, v in c} for c in seed_cs]
    n = len(keys)
    idx = rng.integers(0, n, size=(reps, n))

    def mdelta(sel_keys):
        b = pooled_mal([(k, kb[k]) for k in sel_keys])
        return float(np.mean([pooled_mal([(k, kd[k]) for k in sel_keys])
                              for kd in ks])) - b

    d0 = mdelta(keys)
    ds = np.array([mdelta([keys[j] for j in idx[i]])
                   for i in range(reps)])
    return d0, float(np.percentile(ds, 2.5)), \
        float(np.percentile(ds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rd = args.run_dir
    rng = np.random.default_rng(20260723)
    T = os.path.join(rd, "tables")
    os.makedirs(T, exist_ok=True)
    os.makedirs(os.path.join(rd, "stats"), exist_ok=True)

    # ---- oracle + policy ceilings ----
    oracle = {}
    for tgt in ("fp16", "int4"):
        p = os.path.join(rd, "oracle", f"oracle_{tgt}_mtbench.json")
        if os.path.exists(p):
            oracle[tgt] = json.load(open(p))

    # ---- primary table ----
    rows, taus, cl = [], {}, {}
    for cid, (tag, tgt, desc, wt, rt) in CELLS.items():
        if "{s}" in tag:
            per = []
            for s in SEEDS:
                t, c = tau_of(rd, tag.format(s=s), tgt)
                if t is not None:
                    per.append((s, t, c))
            if not per:
                rows.append(dict(id=cid, target=tgt, draft=desc,
                                 weight_training=wt, rot_training=rt,
                                 tau="", status="missing"))
                continue
            ts = [t for _, t, _ in per]
            tau = float(np.mean(ts))
            taus[cid] = tau
            cl[cid] = per[0][2]           # seed-0 clusters for pairing
            cl[cid + "_all"] = [c for _, _, c in per]
            rows.append(dict(id=cid, target=tgt, draft=desc,
                             weight_training=wt, rot_training=rt,
                             tau=round(tau, 4),
                             seed_std=round(float(np.std(ts)), 4),
                             seeds=len(per), status="ok"))
        else:
            t, c = tau_of(rd, tag, tgt)
            if t is None:
                rows.append(dict(id=cid, target=tgt, draft=desc,
                                 weight_training=wt, rot_training=rt,
                                 tau="", status="missing"))
                continue
            taus[cid] = t
            cl[cid] = c
            u = ""
            if tgt in oracle:
                u = round((t - 1) / (oracle[tgt]["oracle_tau_measured"]
                                     - 1), 4)
            rows.append(dict(id=cid, target=tgt, draft=desc,
                             weight_training=wt, rot_training=rt,
                             tau=round(t, 4), oracle_util=u,
                             status="ok"))
    # oracle utilization for seed cells
    for r in rows:
        if r["status"] == "ok" and r["id"] in taus and \
                r["target"] in oracle and "oracle_util" not in r:
            r["oracle_util"] = round(
                (taus[r["id"]] - 1) /
                (oracle[r["target"]]["oracle_tau_measured"] - 1), 4)
    import csv as _csv
    with open(os.path.join(T, "primary_table.csv"), "w", newline="") as f:
        cols = ["id", "target", "draft", "weight_training",
                "rot_training", "tau", "seed_std", "seeds",
                "oracle_util", "status"]
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- attainment metrics + criteria ----
    att, verd = {}, {}
    for tgt, ptq, qat, naive, rot, ceil in (
            ("fp16", "C2", "C3", "N1", "C11", "C8"),
            ("int4", "C5", "C7", "N2", "C12", "C6")):
        a = {}
        if all(k in taus for k in (ptq, qat)):
            a["tau_strict_PTQ"] = taus[ptq]
            a["tau_QAT_seedmean"] = taus[qat]
            a["QAT_gain"] = taus[qat] - taus[ptq]
            a["TF_gap_to_QAT"] = taus[ptq] - taus[qat]
            base_c = cl[ptq]
            seed_cs = cl.get(qat + "_all", [cl[qat]])
            d0, lo, hi = seed_mean_delta_ci(base_c, seed_cs, rng,
                                            args.reps)
            a["delta_QAT_minus_PTQ"] = d0
            a["ci_delta"] = [lo, hi]
            # Criterion A: LCB of (PTQ - QAT) > -0.05  <=>  UCB(QAT-PTQ)
            # < 0.05
            a["criterionA_LCB_ptq_minus_qat"] = -hi
            a["criterionA_pass"] = bool(-hi > -0.05)
        if rot in taus and qat in taus:
            a["Frozen_rotation_gap_to_QAT"] = taus[rot] - taus[qat]
        if qat in taus and ceil in taus and taus[ceil] > 1:
            a["QAT_fraction_of_empirical_ceiling"] = \
                (taus[qat] - 1) / (taus[ceil] - 1)
        if all(k in taus for k in (ptq, qat, naive)):
            rec_gap = taus[qat] - taus[naive]
            struct = taus[ptq] - taus[naive]
            a["recoverable_gap"] = rec_gap
            a["structural_recovery"] = struct
            a["recovery_ratio"] = (struct / rec_gap if abs(rec_gap) > 1e-9
                                   else float("nan"))
            gap = taus[qat] - taus[ptq]
            a["criterionB_pass"] = bool(
                a["recovery_ratio"] >= 0.90 and gap <= 0.10)
        if tgt in oracle:
            a["oracle_tau"] = oracle[tgt]["oracle_tau_measured"]
            a["policy_theoretical_tau"] = \
                oracle[tgt]["policy_theoretical_tau"]
        att[tgt] = a

    # ---- decision logic (section 22) ----
    def g(t, k):
        return att.get(t, {}).get(k)

    concl = {}
    if all(g(t, "criterionA_pass") is not None for t in ("fp16", "int4")):
        a_fp, a_i4 = g("fp16", "criterionA_pass"), \
            g("int4", "criterionA_pass")
        b_fp, b_i4 = g("fp16", "criterionB_pass"), \
            g("int4", "criterionB_pass")
        qat_sig_fp = g("fp16", "ci_delta") and \
            g("fp16", "ci_delta")[0] > 0
        qat_sig_i4 = g("int4", "ci_delta") and \
            g("int4", "ci_delta")[0] > 0
        concl["A_qat_necessary"] = bool(
            qat_sig_fp and qat_sig_i4 and not (a_fp and a_i4)
            and not (b_fp and b_i4))
        concl["B_tf_ptq_competitive"] = bool(a_fp and a_i4 and
                                             (b_fp or b_i4))
        if all(k in taus for k in ("C4", "C6", "C7", "C10")):
            concl["C_target_adaptation_matters"] = bool(
                taus["C6"] - taus["C4"] > 0.05 and
                taus["C10"] >= taus["C7"] - 0.05)
        if all(k in taus for k in ("C3", "C9", "C7", "C10")):
            concl["D_quant_exposure_necessary"] = bool(
                taus["C3"] > taus["C9"] + 0.02 and
                taus["C7"] > taus["C10"] + 0.02)
        if all(k in taus for k in ("C11", "C3", "C12", "C7")):
            concl["E_rotation_closes_gap"] = bool(
                taus["C3"] - taus["C11"] <= 0.05 and
                taus["C7"] - taus["C12"] <= 0.05)
        qg = [g(t, "QAT_gain") for t in ("fp16", "int4")
              if g(t, "QAT_gain") is not None]
        concl["F_qat_gain_small"] = bool(
            qg and all(x < 0.10 for x in qg) and
            (qat_sig_fp or qat_sig_i4))

    out = dict(attainment=att, conclusions=concl,
               taus={k: round(v, 4) for k, v in taus.items()})
    json.dump(out, open(os.path.join(rd, "stats", "verdicts.json"), "w"),
              indent=1, default=float)

    # ---- secondary tables from shards ----
    # acceptance by depth + first-rejection distribution
    for name, ids in (("acceptance_by_depth",
                       ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C11",
                        "C12", "N1", "N2"]),):
        dep = {}
        for cid in ids:
            if cid not in cl:
                continue
            counts = {}
            for _, ts in cl[cid]:
                for t in ts:
                    counts[t] = counts.get(t, 0) + 1
            tot = sum(counts.values())
            dep[cid] = {str(k): round(v / tot, 4)
                        for k, v in sorted(counts.items())}
        json.dump(dep, open(os.path.join(T, f"{name}.json"), "w"),
                  indent=1)

    # dataset-specific tau
    ds_rows = []
    for cid, (tag, tgt, *_rest) in CELLS.items():
        for ds in ("mtbench", "sharegpt", "c4", "gsm8k", "humaneval"):
            tg = tag.format(s=0) if "{s}" in tag else tag
            t, _ = tau_of(rd, tg, tgt, ds)
            if t is not None:
                ds_rows.append(dict(id=cid, dataset=ds, target=tgt,
                                    tau=round(t, 4)))
    with open(os.path.join(T, "dataset_tau.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "dataset", "target",
                                           "tau"])
        w.writeheader()
        for r in ds_rows:
            w.writerow(r)

    # seed variance table
    sv = []
    for cid in ("C3", "C7"):
        tag = CELLS[cid][0]
        for s in SEEDS:
            t, _ = tau_of(rd, tag.format(s=s), CELLS[cid][1])
            if t is not None:
                sv.append(dict(id=cid, seed=s, tau=round(t, 4)))
    json.dump(sv, open(os.path.join(T, "seed_variance.json"), "w"),
              indent=1)

    # preparation cost from training manifests
    cost = {}
    for p in glob.glob(os.path.join(rd, "manifests", "train_*.json")):
        m = json.load(open(p))
        cost[m["tag"]] = dict(gpu_hours=m.get("gpu_hours"),
                              tokens=m.get("tokens"),
                              peak_mem_gib=m.get("peak_mem_gib"))
    json.dump(cost, open(os.path.join(T, "preparation_cost.json"), "w"),
              indent=1)

    # model memory table (analytic)
    D, V, FF = 4096, 32000, 11008
    draft_lin = D * 2 * D + 4 * D * D + 3 * D * FF
    mem = dict(
        draft_fp16_MiB=round((draft_lin + V * D + D * V) * 2 / 2**20, 1),
        draft_d4p3_MiB=round((draft_lin * 0.5 + (V * D + D * V) * 2)
                             / 2**20, 1),
        target_fp16_MiB=round(6.74e9 * 2 / 2**20, 1),
        target_w4_MiB=round((6.74e9 * 0.5 + 2 * V * D * 2) / 2**20, 1),
        note="INT4 packed estimate: 0.5 byte/param on quantized linears; "
             "embedding + LM head fp16 (D4P3 policy)")
    json.dump(mem, open(os.path.join(T, "model_memory.json"), "w"),
              indent=1)

    print(json.dumps(out, indent=1, default=float))
    print(f"[attain] tables -> {T}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
