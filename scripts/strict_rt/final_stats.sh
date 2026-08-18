#!/bin/bash
# Strict SEAGLE-RT final statistics: bootstrap P1-P7 (10k, ALL pairs in
# ONE invocation per dataset — the overwrite trap), Holm per family
# across the 4 datasets, retention table, RCAL metrics + bootstrap.
set -u
cd /home/thahn1230/SEAGLE
RUN=runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552
PY=/home/thahn1230/anaconda3/envs/seagle/bin/python
export HF_HOME=/data/thahn1230/hf_cache

for ds in mtbench gsm8k sharegpt humaneval; do
  $PY scripts/bootstrap_eagle_tau.py --run-dir $RUN --dataset $ds \
    --reps 10000 \
    --pair "P1_ctrl_vs_fp16=CTRL_OI@int4:SRT_FP16@int4" \
    --pair "P2_fp16_vs_w8a8rtn=SRT_FP16@int4:SRT_W8A8_RTN@int4" \
    --pair "P3_w8a8rtn_vs_sq=SRT_W8A8_RTN@int4:SRT_W8A8_SQ@int4" \
    --pair "P4_fp16_vs_w4a4rtn=SRT_FP16@int4:SRT_W4A4_RTN@int4" \
    --pair "P5_rtn_vs_had=SRT_W4A4_RTN@int4:SRT_W4A4_HAD@int4" \
    --pair "P6_had_vs_sq=SRT_W4A4_HAD@int4:SRT_W4A4_SQ@int4" \
    --pair "P7_sq_vs_rescue=SRT_W4A4_SQ@int4:SRT_RESCUE_ASQ@int4"
done

$PY scripts/compute_eagle_rcal_metrics.py --run-dir $RUN
$PY scripts/bootstrap_eagle_rcal.py --run-dir $RUN \
  --tags SRT_FP16,SRT_W8A8_RTN,SRT_W4A4_RTN,SRT_W4A4_SQ,SRT_RESCUE_ASQ \
  --pairs "SRT_FP16:SRT_W8A8_RTN,SRT_FP16:SRT_W4A4_RTN,SRT_W4A4_SQ:SRT_RESCUE_ASQ" \
  --dataset mtbench --reps 3000

$PY - <<'EOF'
import csv, glob, json, math, os
RUN = "runs/eagle1_strict_seagle_rt_native_w4a4_20260816_172552"
DS = ("mtbench", "gsm8k", "sharegpt", "humaneval")

# ---- Holm per preregistered family across the 4 datasets ----
boots = {}
for ds in DS:
    for e in json.load(open(f"{RUN}/stats/bootstrap_pairs_{ds}.json")):
        if e.get("status") != "ok":
            continue
        boots.setdefault(e["name"], {})[ds] = e
holm = {}
ALPHA = 0.05
for fam, per in sorted(boots.items()):
    items = sorted(per.items(), key=lambda kv: kv[1]["p_two_sided"])
    m = len(items)
    alive, out = True, {}
    for i, (ds, e) in enumerate(items):
        p = max(e["p_two_sided"], 1 / 10000)
        thr = ALPHA / (m - i)
        sig = alive and p <= thr
        if not sig:
            alive = False
        out[ds] = dict(delta=round(e["delta_b_minus_a"], 4),
                       lo=round(e["ci_delta"][0], 4), hi=round(e["ci_delta"][1], 4),
                       p=p, sig=bool(sig))
    holm[fam] = out
json.dump(holm, open(f"{RUN}/stats/holm.json", "w"), indent=1)
with open(f"{RUN}/tables/holm.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "dataset", "delta", "ci_lo", "ci_hi", "p", "holm_sig"])
    for fam, per in holm.items():
        for ds in DS:
            e = per.get(ds)
            if e:
                w.writerow([fam, ds, e["delta"], e["lo"], e["hi"], e["p"], e["sig"]])
    wins = {f: sum(1 for d in per.values() if d["sig"]) for f, per in holm.items()}
print("[holm]", json.dumps(wins))

# ---- bootstrap.csv (flat export) ----
with open(f"{RUN}/tables/bootstrap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "dataset", "tau_a", "tau_b", "delta", "ci_lo", "ci_hi", "p"])
    for fam, per in boots.items():
        for ds, e in per.items():
            w.writerow([fam, ds, round(e["tau_a"], 4), round(e["tau_b"], 4),
                        round(e["delta_b_minus_a"], 4), round(e["ci_delta"][0], 4),
                        round(e["ci_delta"][1], 4), e["p_two_sided"]])

# ---- final al_4dataset + retention ----
def tau(p):
    t = c = 0
    for r in csv.DictReader(open(p)):
        al = json.loads(r["acceptance_list"]); t += sum(al); c += len(al)
    return t / max(c, 1)
tags = ["CTRL_OI", "SRT_FP16", "SRT_W8A8_RTN", "SRT_W8A8_SQ",
        "SRT_W4A4_RTN", "SRT_W4A4_HAD", "SRT_W4A4_SQ",
        "SRT_RESCUE", "SRT_RESCUE_ASQ"]
rows = {}
for t_ in tags:
    d = {}
    for ds in DS:
        p = f"{RUN}/shards/al__{t_}__int4__{ds}.csv"
        if os.path.exists(p):
            d[ds] = round(tau(p), 4)
    if len(d) == 4:
        d["mean4"] = round(sum(d.values()) / 4, 4)
    rows[t_] = d
ref = rows["SRT_FP16"]["mean4"]
with open(f"{RUN}/tables/al_4dataset.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag"] + list(DS) + ["mean4", "retention"])
    for t_ in tags:
        d = rows.get(t_, {})
        ret = round(d["mean4"] / ref, 4) if "mean4" in d and t_ != "CTRL_OI" else ""
        w.writerow([t_] + [d.get(k, "") for k in DS] + [d.get("mean4", ""), ret])
with open(f"{RUN}/tables/al_retention.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "mean4", "retention", "drop"])
    for t_ in tags[1:]:
        d = rows.get(t_, {})
        if "mean4" in d:
            w.writerow([t_, d["mean4"], round(d["mean4"] / ref, 4),
                        round(ref - d["mean4"], 4)])
# ---- compute-quality tradeoff ----
with open(f"{RUN}/tables/compute_quality_tradeoff.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "core_adaptation_gpu_h", "draft_rotation_gpu_h",
                "provenance", "mean4_w4a4_deploy", "mean4_fp16_draft"])
    w.writerow(["SEAGLE-PTQ", 3.0, 1.8, "existing canonical (est/measured)", 3.4788, ""])
    w.writerow(["SEAGLE-QAT", 8.03, 1.8, "existing canonical (measured)", 3.6108, ""])
    w.writerow(["SEAGLE-RT (strict)", 267.9, 0.5,
                "THIS RUN measured (train timestamps; +4.66 cache gen)",
                rows["SRT_RESCUE_ASQ"].get("mean4", ""), ref])
print("[tables] al_4dataset / retention / tradeoff written")
print(json.dumps(rows, indent=1))
EOF
echo "FINAL_STATS_DONE"
