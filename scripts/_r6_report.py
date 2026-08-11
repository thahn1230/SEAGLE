#!/usr/bin/env python
"""R6 study: tables A/B/C + final report from raw cycle records.

Usage: _r6_report.py <run_dir>
Writes tables/{al_by_dataset.csv,final_summary.json,bootstrap_summary.csv}
and reports/R6_DRAFT_AWARE_R2_FINAL_REPORT.md (+ KOREAN_SUMMARY.md).
"""
import csv, glob, hashlib, json, os, subprocess, sys

rd = sys.argv[1]
DS = ["mtbench", "gsm8k", "sharegpt", "humaneval"]
NP = {"mtbench": 80, "gsm8k": 200, "sharegpt": 80, "humaneval": 164}


def taus(tag):
    out, lock = {}, {}
    for ds in DS:
        p = os.path.join(rd, "shards", f"al__{tag}__int4__{ds}.csv")
        if not os.path.exists(p):
            return None, None
        rows = list(csv.DictReader(open(p)))
        ts = [t for r in rows for t in json.loads(r["acceptance_list"])]
        pids = [r["prompt_id"] for r in rows]
        out[ds] = sum(ts) / max(len(ts), 1)
        lock[ds] = dict(prompts=len(rows), expected=NP[ds],
                        dup=len(pids) - len(set(pids)), cycles=len(ts))
    out["mean"] = sum(out[d] for d in DS) / 4
    return out, lock


def mt_only(tag):
    p = os.path.join(rd, "shards", f"al__{tag}__int4__mtbench.csv")
    if not os.path.exists(p):
        return None
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return sum(ts) / max(len(ts), 1)


med = json.load(open(os.path.join(rd, "tables", "median_seeds.json")))
PT = f"R6P_s{med['r6p']['seed']}"
QT = f"R6Q_s{med['r6q']['seed']}"
M, LK = {}, {}
for tag in ("B9_T4", "QF_rdg_T4_s2", PT, QT):
    t, l = taus(tag)
    if t:
        M[tag], LK[tag] = t, l

boot, holm = {}, {}
for ds in DS:
    p = os.path.join(rd, "stats", f"bootstrap_pairs_{ds}.json")
    if os.path.exists(p):
        boot[ds] = {r["name"]: r for r in json.load(open(p))
                    if r.get("status") == "ok"}
hp = os.path.join(rd, "stats", "holm_adjusted.json")
if os.path.exists(hp):
    holm = json.load(open(hp))


def hrow(ds, name):
    return holm.get(f"{ds}:{name}", {})


def pfmt(p):
    return "< 1/3000" if p is not None and p <= 1 / 3000 else \
        (f"{p:.4f}" if p is not None else "—")


os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
os.makedirs(os.path.join(rd, "reports"), exist_ok=True)

rows = [["arm", "method"] + DS + ["mean4"]]
LBL = [("B9_T4", "PTQ", "R5 (reused baseline)"),
       (PT, "PTQ", "R5 + R6 (learned)"),
       ("QF_rdg_T4_s2", "QAT", "R5 + QAT (reused baseline)"),
       (QT, "QAT", "R5 + R6 + QAT (stage-2 R6)")]
for tag, kind, lbl in LBL:
    if tag in M:
        rows.append([lbl, kind] + [f"{M[tag][d]:.4f}" for d in DS]
                    + [f"{M[tag]['mean']:.4f}"])
with open(os.path.join(rd, "tables", "al_by_dataset.csv"), "w",
          newline="") as f:
    csv.writer(f).writerows(rows)

bs = [["pair", "dataset", "delta", "ci_lo", "ci_hi", "p_raw", "p_holm"]]
for ds in DS:
    for name, r in boot.get(ds, {}).items():
        h = hrow(ds, name)
        bs.append([name, ds, r["delta_b_minus_a"], r["ci_delta"][0],
                   r["ci_delta"][1], r["p_two_sided"],
                   h.get("p_holm", "")])
with open(os.path.join(rd, "tables", "bootstrap_summary.csv"), "w",
          newline="") as f:
    csv.writer(f).writerows(bs)

git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip()
dw = {}
dwp = os.path.join(rd, "geometry", "r6_depthwise.json")
if os.path.exists(dwp):
    dw = json.load(open(dwp))
rc = {}
rcp = os.path.join(rd, "tables", "rcal_metrics.json")
if os.path.exists(rcp):
    rc = json.load(open(rcp))
rndg = {}
rgp = os.path.join(rd, "geometry", "r6_randctl.json")
if os.path.exists(rgp):
    rndg = json.load(open(rgp))
gate = {}
gp = os.path.join(rd, "gradchecks", "gateR6_parity.json")
if os.path.exists(gp):
    gate = json.load(open(gp))
rt = {}
for f in glob.glob(os.path.join(rd, "tables", "runtime__*.json")):
    j = json.load(open(f))
    rt[j["tag"]] = j
prox = {}
pxp = os.path.join(rd, "geometry", "r6_proxy_diag.json")
if os.path.exists(pxp):
    prox = json.load(open(pxp))


def dline(name, label):
    parts = []
    for ds in DS:
        r = boot.get(ds, {}).get(name)
        if not r:
            continue
        h = hrow(ds, name)
        sig = "SIG" if h.get("reject_05") else "n.s."
        parts.append(f"| {ds} | {r['delta_b_minus_a']:+.4f} "
                     f"[{r['ci_delta'][0]:+.4f}, {r['ci_delta'][1]:+.4f}]"
                     f" | {pfmt(r['p_two_sided'])} | "
                     f"{pfmt(h.get('p_holm'))} | {sig} |")
    if not parts:
        return []
    return [f"**{label}**", "",
            "| Dataset | delta tau [95% CI] | raw p | Holm p | |",
            "|---|---|---:|---:|---|"] + parts + [""]


Lm = []
w = Lm.append
w("# Draft-aware attention V/O rotation R6 (SpinQuant R2 counterpart) — "
  "final report")
w("")
w(f"Run: `{rd}` · git `{git}` · W4A4 fake-quantized (no real-INT4 "
  f"latency claims) · GS scaling (m = 4096^0.42) · R5 frozen/reused · "
  f"objective = acceptance-aware LK surrogate (not a direct greedy-AL "
  f"loss).")
w("")
if M.get("B9_T4") and M.get(PT):
    a, b = M["B9_T4"], M[PT]
    w("## Table A — PTQ (target W4A4)")
    w("")
    w("| Method | MT-Bench | GSM8K | ShareGPT | HumanEval | mean4 |")
    w("|---|---:|---:|---:|---:|---:|")
    w("| R5 | " + " | ".join(f"{a[d]:.4f}" for d in DS)
      + f" | {a['mean']:.4f} |")
    w("| R5 + R6 | " + " | ".join(f"{b[d]:.4f}" for d in DS)
      + f" | {b['mean']:.4f} |")
    w("| Delta | " + " | ".join(f"{b[d]-a[d]:+.4f}" for d in DS)
      + f" | **{b['mean']-a['mean']:+.4f}** |")
    w("")
if M.get("QF_rdg_T4_s2") and M.get(QT):
    a, b = M["QF_rdg_T4_s2"], M[QT]
    w("## Table B — QAT (stage-2 R6 on frozen QAT weights)")
    w("")
    w("| Method | MT-Bench | GSM8K | ShareGPT | HumanEval | mean4 |")
    w("|---|---:|---:|---:|---:|---:|")
    w("| R5 + QAT | " + " | ".join(f"{a[d]:.4f}" for d in DS)
      + f" | {a['mean']:.4f} |")
    w("| R5 + R6 + QAT | " + " | ".join(f"{b[d]:.4f}" for d in DS)
      + f" | {b['mean']:.4f} |")
    w("| Delta | " + " | ".join(f"{b[d]-a[d]:+.4f}" for d in DS)
      + f" | **{b['mean']-a['mean']:+.4f}** |")
    w("")
if all(t in M for t in ("B9_T4", PT, "QF_rdg_T4_s2", QT)):
    w("## Table C — overall (target W4A4)")
    w("")
    w("| R5 PTQ | R5+R6 PTQ | dR6 PTQ | R5 QAT | R5+R6 QAT | dR6 QAT |")
    w("|---:|---:|---:|---:|---:|---:|")
    w(f"| {M['B9_T4']['mean']:.4f} | {M[PT]['mean']:.4f} | "
      f"{M[PT]['mean']-M['B9_T4']['mean']:+.4f} | "
      f"{M['QF_rdg_T4_s2']['mean']:.4f} | {M[QT]['mean']:.4f} | "
      f"{M[QT]['mean']-M['QF_rdg_T4_s2']['mean']:+.4f} |")
    w("")
w("## Statistics (paired prompt-cluster bootstrap, 3000 resamples; "
  "Holm within family F1 = PTQ, F2 = QAT)")
w("")
for nm, lb in (("r6_gain_ptq", "F1: R5+R6 PTQ vs R5 PTQ"),
               ("r6_gain_qat", "F2: R5+R6+QAT vs R5+QAT"),
               ("qat_gain_over_r6ptq",
                "exploratory: R5+R6+QAT vs R5+R6 PTQ")):
    Lm += dline(nm, lb)
rand_lines = []
for s in (101, 102, 103):
    r = boot.get("mtbench", {}).get(f"rand_r6_s{s}")
    if r:
        rand_lines.append(
            f"| seed {s} | {r['delta_b_minus_a']:+.4f} "
            f"[{r['ci_delta'][0]:+.4f}, {r['ci_delta'][1]:+.4f}] | "
            f"{pfmt(r['p_two_sided'])} |")
if rand_lines:
    w("**Random-R6 control (matched GENERATOR Frobenius norm; "
      "eigenangle magnitudes reported, not matched — mtbench only, "
      "exploratory)**")
    w("")
    w("| control | delta vs R5-only | raw p |")
    w("|---|---|---:|")
    Lm.extend(rand_lines)
    w("")
if med:
    w("## Seeds")
    w("")
    w(f"PTQ arm taus (mtbench): {med['r6p']['taus']} -> median seed "
      f"{med['r6p']['seed']}; QAT arm: {med['r6q']['taus']} -> median "
      f"seed {med['r6q']['seed']} (pre-registered median rule, never "
      f"best; TRUE best-validation checkpoints evaluated).")
    w("")
if dw:
    w("## Depth-wise analysis (k=1..4, held-out val windows, "
      "teacher-forced)")
    w("")
    w("| arm | " + " | ".join(f"alpha_k{k}" for k in range(1, 5))
      + " | val E[tau] |")
    w("|---|" + "---:|" * 5)
    for nm, r in dw.get("arms", {}).items():
        w(f"| {nm} | " + " | ".join(
            f"{r[f'depth_{k}']['alpha']:.4f}" for k in range(1, 5))
          + f" | {r['val_expected_tau']:.4f} |")
    w("")
if prox:
    w("## Quantization proxies (diagnostic only — not used for "
      "selection)")
    w("")
    a, b = prox["legs"]["R5_baselineR2"], prox["legs"]["R5_R6"]
    w("| site | W4 NMSE base | W4 NMSE R6 | act absmax base | act "
      "absmax R6 | act kurt base | act kurt R6 |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for s in ("v", "o"):
        aa, bb = a[s], b[s]
        w(f"| {s} | {aa['w4_nmse']:.6f} | {bb['w4_nmse']:.6f} | "
          f"{(aa.get('act') or {}).get('absmax', float('nan')):.2f} | "
          f"{(bb.get('act') or {}).get('absmax', float('nan')):.2f} | "
          f"{(aa.get('act') or {}).get('excess_kurtosis', float('nan')):.2f} | "
          f"{(bb.get('act') or {}).get('excess_kurtosis', float('nan')):.2f} |")
    w("")
if rc:
    w("## RCAL (MT-Bench, proposal-only — never mixed with official "
      "tau)")
    w("")
    w("| arm | AL_q | AL_0 | RCAL | SAL | LAL | AFS |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    lbl = {"VB9_T4": "R5 PTQ", "VR6P_T4": "R5+R6 PTQ",
           "VQF_rdg_T4": "R5 QAT", "VR6Q_T4": "R5+R6 QAT"}
    for k, v in rc.items():
        t = k.split("__")[0]
        if t in lbl:
            w(f"| {lbl[t]} | {v['AL_q']:.4f} | {v['AL_0']:.4f} | "
              f"{v['RCAL']:.4f} | {v['SAL']:.4f} | {v['LAL']:.4f} | "
              f"{v['AFS']:.4f} |")
    w("")
    w("SAL is reported as verifier-specific acceptance (not labelled "
      "\"wrong tokens\" — no task-quality evidence here).")
    w("")
if gate:
    w("## Correctness gates")
    w("")
    w(f"gateR6 verdict: **{gate.get('verdict')}** — orthogonality "
      f"(max {max(gate['results']['test_r6_1_orthogonality_maxabs'].values()):.2e}), "
      f"identity-init bitwise parity, FP-gauge fold equivalence "
      f"(fp64 max diff "
      f"{gate['results']['test_r6_3_fp_gauge']['base_vs_perturbed']:.2e}), "
      f"fold-before-quant re-quantization match, no runtime R6 operator "
      f"(module count {gate['results']['test_r6_5_no_runtime_op']['n_fake_w4a4']}"
      f" == baseline), core/runtime chain parity, dL/dB > 0 "
      f"({[round(x,4) for x in gate['results']['gate_r2b_grad_norms']]}).")
    w("")
if rt:
    w("## Runtime / foldability")
    w("")
    w("| arm | draft ms/cyc | verify ms/cyc | postproj ms/cyc | "
      "ms/token |")
    w("|---|---:|---:|---:|---:|")
    for k, v in sorted(rt.items()):
        w(f"| {k} | {v['draft_ms_per_cycle']:.3f} | "
          f"{v['verify_ms_per_cycle']:.3f} | "
          f"{v['postproj_ms_per_cycle']:.4f} | "
          f"{v['ms_per_token']:.3f} |")
    w("")
    w("R6 folds offline into v/o; no additional inference operator "
      "(gateR6 module-count parity). Fake-quant timings — no real INT4 "
      "claims. Preparation cost: one 3000-step rotation training per "
      "arm/seed (~1.5 h on one RTX 4090).")
    w("")

verdict_p = verdict_q = "inconclusive"
if all(t in M for t in ("B9_T4", PT)):
    dp = M[PT]["mean"] - M["B9_T4"]["mean"]
    sig_p = [hrow(d, "r6_gain_ptq").get("reject_05") for d in DS]
    verdict_p = ("beneficial" if dp > 0 and any(sig_p) else
                 ("harmful" if dp < 0 and any(sig_p) else "neutral"))
if all(t in M for t in ("QF_rdg_T4_s2", QT)):
    dq = M[QT]["mean"] - M["QF_rdg_T4_s2"]["mean"]
    sig_q = [hrow(d, "r6_gain_qat").get("reject_05") for d in DS]
    verdict_q = ("beneficial" if dq > 0 and any(sig_q) else
                 ("harmful" if dq < 0 and any(sig_q) else "neutral"))
w("## Verdict")
w("")
w(f"    R6 verdict: {verdict_p} under PTQ, {verdict_q} under QAT.")
if all(t in M for t in ("B9_T4", PT, "QF_rdg_T4_s2", QT)):
    w(f"    Evidence: PTQ mean4 {M['B9_T4']['mean']:.4f} -> "
      f"{M[PT]['mean']:.4f} ({M[PT]['mean']-M['B9_T4']['mean']:+.4f}); "
      f"QAT mean4 {M['QF_rdg_T4_s2']['mean']:.4f} -> "
      f"{M[QT]['mean']:.4f} "
      f"({M[QT]['mean']-M['QF_rdg_T4_s2']['mean']:+.4f}).")
w("")
open(os.path.join(rd, "reports",
                  "R6_DRAFT_AWARE_R2_FINAL_REPORT.md"),
     "w").write("\n".join(Lm) + "\n")

json.dump(dict(measured={k: {d: round(v, 4) for d, v in m.items()}
                         for k, m in M.items()},
               median_seeds=med, verdict_ptq=verdict_p,
               verdict_qat=verdict_q, result_lock=LK),
          open(os.path.join(rd, "tables", "final_summary.json"), "w"),
          indent=1)
print("[r6-report] verdicts:", verdict_p, "/", verdict_q)
print("[r6-report] ->", os.path.join(
    rd, "reports", "R6_DRAFT_AWARE_R2_FINAL_REPORT.md"))
