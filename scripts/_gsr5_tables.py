#!/usr/bin/env python
"""GS+R5 campaign table builder + result lock.

Recomputes every dataset tau directly from raw cycle records (shards),
verifies prompt counts / no duplicate prompts, then writes all required
CSVs. Historical LS rows are carried verbatim as reference; every NEW
row is recomputed here, never copied.

Usage: _gsr5_tables.py <run_dir>
"""
import csv, json, os, sys

rd = sys.argv[1]
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]
NP = {"mtbench": 80, "gsm8k": 200, "sharegpt": 80, "humaneval": 164}


def measure(tag):
    """pooled micro-tau per dataset from raw cycle records + lock checks."""
    out, lock = {}, {}
    for ds in DS:
        p = os.path.join(rd, "shards", f"al__{tag}__int4__{ds}.csv")
        if not os.path.exists(p):
            return None, None
        rows = list(csv.DictReader(open(p)))
        ts = [t for r in rows for t in json.loads(r["acceptance_list"])]
        pids = [r["prompt_id"] for r in rows]
        out[ds] = sum(ts) / max(len(ts), 1)
        lock[ds] = dict(n_prompts=len(rows), expected=NP[ds],
                        n_prompts_ok=len(rows) == NP[ds],
                        duplicates=len(pids) - len(set(pids)),
                        n_cycles=len(ts), tau=round(out[ds], 4))
    out["mean"] = sum(out[d] for d in DS) / len(DS)
    return out, lock


# tag -> (report label, kind)
ARMS = [
    ("B3R_T4", "GS PTQ (re-run: reuse target R_T)", "new"),
    ("B9_T4", "GS + R5 PTQ", "new"),
    ("QFR_ep3g_T4", "GS + QAT (re-run: reuse target R_T)", "new"),
    ("B7R_T4", "LS + R5 PTQ (re-run)", "new"),
]
med_p = os.path.join(rd, "tables", "median_seeds.json")
QTAG = None
if os.path.exists(med_p):
    s = json.load(open(med_p))["rdg_T4"]["seed"]
    QTAG = f"QF_rdg_T4_s{s}"
    ARMS.append((QTAG, "GS + R5 + QAT", "new"))

measured, locks = {}, {}
for tag, _, _ in ARMS:
    m, l = measure(tag)
    if m:
        measured[tag], locks[tag] = m, l

# historical canonical PMG rows (LS + naive/generic), reference only
HIST = {
    "Naive PTQ": dict(mtbench=None, mean=1.267, src="PMG B1_T4"),
    "Generic QAT": dict(mean=1.801, src="PMG QF_gen_T4"),
    "GS PTQ": dict(mtbench=2.996, gsm8k=3.449, sharegpt=3.109,
                   humaneval=3.749, mean=3.326, src="PMG B3_T4"),
    "GS + QAT": dict(mtbench=3.192, gsm8k=3.633, sharegpt=3.268,
                     humaneval=3.699, mean=3.448, src="PMG QF_ep3g_T4_s2"),
    "LS PTQ": dict(mean=3.343, src="PMG B5_T4"),
    "LS + QAT": dict(mean=3.445, src="PMG QF_ep3p_T4"),
    "LS + R5 PTQ": dict(mtbench=3.200, gsm8k=3.739, sharegpt=3.312,
                        humaneval=3.856, mean=3.527, src="PMG B7_T4"),
    "LS + R5 + QAT": dict(mtbench=3.248, gsm8k=3.617, sharegpt=3.324,
                          humaneval=3.700, mean=3.472, src="PMG QF_rd_T4_s0"),
}


def f(x):
    return "" if x is None else f"{x:.4f}"


os.makedirs(os.path.join(rd, "tables"), exist_ok=True)

# --- main method table (historical rows verbatim + NEW rows measured) ---
rows = [["draft_method", "target_w4a4", "source", "provenance"]]
order = ["Naive PTQ", "Generic QAT", "GS PTQ", "GS + QAT"]
for k in order:
    rows.append([k, f"{HIST[k]['mean']:.3f}", HIST[k]["src"], "historical"])
if "B9_T4" in measured:
    rows.append(["GS + R5 PTQ", f"{measured['B9_T4']['mean']:.4f}",
                 "B9_T4 (this run)", "new"])
if QTAG and QTAG in measured:
    rows.append(["GS + R5 + QAT", f"{measured[QTAG]['mean']:.4f}",
                 f"{QTAG} (this run)", "new"])
for k in ["LS PTQ", "LS + QAT", "LS + R5 PTQ", "LS + R5 + QAT"]:
    rows.append([k, f"{HIST[k]['mean']:.3f}", HIST[k]["src"], "historical"])
with open(os.path.join(rd, "tables", "w4a4_method_table.csv"), "w",
          newline="") as fh:
    csv.writer(fh).writerows(rows)

# --- per-dataset table ---
rows = [["method"] + DS + ["mean", "source", "provenance"]]
for k in ["GS PTQ", "GS + QAT"]:
    h = HIST[k]
    rows.append([k] + [f(h.get(d)) for d in DS] + [f"{h['mean']:.3f}",
                                                   h["src"], "historical"])
for tag, label, _ in ARMS:
    if tag in measured and tag not in ("B7R_T4",):
        m = measured[tag]
        rows.append([label] + [f(m[d]) for d in DS] + [f(m["mean"]),
                                                       f"{tag} (this run)",
                                                       "new"])
for k in ["LS + R5 PTQ", "LS + R5 + QAT"]:
    h = HIST[k]
    rows.append([k] + [f(h.get(d)) for d in DS] + [f"{h['mean']:.3f}",
                                                   h["src"], "historical"])
if "B7R_T4" in measured:
    m = measured["B7R_T4"]
    rows.append(["LS + R5 PTQ (re-run parity)"] + [f(m[d]) for d in DS]
                + [f(m["mean"]), "B7R_T4 (this run)", "new"])
with open(os.path.join(rd, "tables", "w4a4_per_dataset.csv"), "w",
          newline="") as fh:
    csv.writer(fh).writerows(rows)

# --- 19.5 GS rotation-policy table (fresh paired records only) ---
if "B3R_T4" in measured and "B9_T4" in measured:
    a, b = measured["B3R_T4"], measured["B9_T4"]
    rows = [["rotation_policy_under_GS"] + DS + ["avg"]]
    rows.append(["Reuse Target Rotation (R_T)"]
                + [f(a[d]) for d in DS] + [f(a["mean"])])
    rows.append(["Draft-aware Rotation (R5)"]
                + [f(b[d]) for d in DS] + [f(b["mean"])])
    rows.append(["Delta (R5 - R_T)"]
                + [f"{b[d]-a[d]:+.4f}" for d in DS]
                + [f"{b['mean']-a['mean']:+.4f}"])
    with open(os.path.join(rd, "tables", "gs_rt_vs_r5_per_dataset.csv"),
              "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with open(os.path.join(rd, "tables", "gs_rt_vs_r5_summary.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scaling", "draft_rotation", "tag"] + DS + ["avg"])
        w.writerow(["GS", "reuse R_T", "B3R_T4"]
                   + [f(a[d]) for d in DS] + [f(a["mean"])])
        w.writerow(["GS", "learned R5", "B9_T4"]
                   + [f(b[d]) for d in DS] + [f(b["mean"])])

# --- 19.9 QAT rotation table ---
if QTAG and QTAG in measured:
    rows = [["rotation_policy_under_GS_plus_QAT"] + DS + ["avg", "source"]]
    if "QFR_ep3g_T4" in measured:
        a = measured["QFR_ep3g_T4"]
        rows.append(["Reuse R_T / GS + QAT (re-run)"]
                    + [f(a[d]) for d in DS] + [f(a["mean"]), "this run"])
    h = HIST["GS + QAT"]
    rows.append(["Reuse R_T / GS + QAT (historical)"]
                + [f(h[d]) for d in DS] + [f"{h['mean']:.3f}", h["src"]])
    b = measured[QTAG]
    rows.append(["Learned R5 / GS + R5 + QAT"]
                + [f(b[d]) for d in DS] + [f(b["mean"]), "this run"])
    ref = measured.get("QFR_ep3g_T4") or {d: HIST["GS + QAT"][d] for d in DS}
    if "mean" not in ref:
        ref = dict(ref, mean=HIST["GS + QAT"]["mean"])
    rows.append(["Delta (R5 - R_T)"]
                + [f"{b[d]-ref[d]:+.4f}" for d in DS]
                + [f"{b['mean']-ref['mean']:+.4f}", ""])
    with open(os.path.join(rd, "tables",
                           "gs_rt_vs_r5_qat_per_dataset.csv"), "w",
              newline="") as fh:
        csv.writer(fh).writerows(rows)

# --- QAT seed summary ---
seed_rows = [["seed", "final_train_loss", "final_vloss", "final_top3",
              "selected_ckpt", "calib_tau", "qf_mtbench_tau",
              "gpu_hours", "ckpt_sha256"]]
for s in (0, 1, 2):
    tag = f"Q_rdg_T4_s{s}"
    lg = os.path.join(rd, "logs", f"train_{tag}.jsonl")
    tr = vl = t3 = gh = None
    if os.path.exists(lg):
        recs = [json.loads(l) for l in open(lg)]
        tl = [r for r in recs if "hours" in r]
        vv = [r for r in recs if r.get("kind") == "val"]
        if tl:
            tr, gh = tl[-1].get("ploss"), tl[-1].get("hours")
        if vv:
            vl, t3 = vv[-1].get("vloss"), vv[-1].get("top3")
    selp = os.path.join(rd, "tables", f"qat_selection_{tag}.json")
    sel, ct = "", ""
    if os.path.exists(selp):
        j = json.load(open(selp))["selected"]
        sel, ct = j["ckpt"], j["calib_tau"]
    qt = ""
    sh = os.path.join(rd, "shards", f"al__QF_rdg_T4_s{s}__int4__mtbench.csv")
    if os.path.exists(sh):
        ts = [t for r in csv.DictReader(open(sh))
              for t in json.loads(r["acceptance_list"])]
        qt = round(sum(ts) / max(len(ts), 1), 4)
    shafile = os.path.join(rd, "ckpts", f"{tag}_{sel}.pt.sha256") if sel else ""
    sha = open(shafile).read().split()[0] if os.path.exists(shafile) else ""
    seed_rows.append([s, tr, vl, t3, sel, ct, qt, gh, sha])
with open(os.path.join(rd, "tables", "qat_seed_summary.csv"), "w",
          newline="") as fh:
    csv.writer(fh).writerows(seed_rows)

json.dump(dict(measured={k: {d: round(v, 4) for d, v in m.items()}
                         for k, m in measured.items()},
               result_lock=locks,
               lock_verdict=("PASS" if all(
                   l[d]["n_prompts_ok"] and l[d]["duplicates"] == 0
                   for l in locks.values() for d in DS) else "FAIL")),
          open(os.path.join(rd, "tables", "measured_taus.json"), "w"),
          indent=1)
print(json.dumps({k: {d: round(v, 4) for d, v in m.items()}
                  for k, m in measured.items()}, indent=1))
print("[tables] written ->", os.path.join(rd, "tables"))
