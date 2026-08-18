#!/usr/bin/env python
"""SEAGLE training-cost benchmark analysis (§24 tables + report).

Inputs (produced by _autopilot_bench.sh on ONE GPU):
  logs/{PTQ_R5,PTQ_R6,QAT,QAT_valcad}/r*.log  — LK trainer logs with
      '[lk] step N: ... (Ss)' cumulative-elapsed markers every 20 steps
  logs/bench_steptimes_{cache,hybrid}_on_w1_*.json — strict-RT per-step
  tables/job_walls.jsonl — per-job wall incl. setup

Method contracts (canonical steps): R5 3000, R6 3000, QAT 3000,
RT 41,685 (actual completed strict run). Warmup drop: steps < 60.
"""
import argparse, csv, glob, json, os, re, statistics as st

import random

C_STEPS = dict(PTQ_R5=3000, PTQ_R6=3000, QAT=3000, RT=41685)
HIST = dict(QAT_secstep=19626 / 3 / 3000,     # AAQ logs, 3 runs x 3000
            RT_w8_gpuh=267.9, RT_w8_secstep=33.483 * 3600 / 41685,
            RT_w1_preflight_hybrid=0.953,
            PTQ_alpha_calib_gpuh=3.0)


def dist(samples):
    s = sorted(samples)
    n = len(s)
    return dict(n=n, mean=round(st.mean(s), 4),
                median=round(st.median(s), 4),
                std=round(st.pstdev(s), 4),
                p5=round(s[max(0, int(0.05 * n) - 1)], 4),
                p95=round(s[min(n - 1, int(0.95 * n))], 4))


def boot_ci_median(samples, reps=10000, seed=0):
    rng = random.Random(seed)
    n = len(samples)
    meds = sorted(st.median(rng.choices(samples, k=n))
                  for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps)]


def parse_lk(path, warmup=60):
    """(step, cum_elapsed_s) markers -> per-step sec samples (20-step
    granularity), measured window step>=warmup."""
    pts = []
    for line in open(path, errors="ignore"):
        m = re.match(r"\[lk\] step (\d+):.*\((\d+(?:\.\d+)?)s\)", line)
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    out = []
    for (s0, t0), (s1, t1) in zip(pts, pts[1:]):
        if s0 >= warmup and s1 > s0:
            out.append((t1 - t0) / (s1 - s0))
    setup = pts[0][1] if pts else None   # elapsed at step 0 marker
    return out, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    R = args.run_dir
    T = lambda name: os.path.join(R, "tables", name)

    per_method = {}
    rows_steps = []
    # LK-family methods
    for meth in ("PTQ_R5", "PTQ_R6", "QAT", "QAT_valcad"):
        allsamp = []
        for lg in sorted(glob.glob(f"{R}/logs/{meth}/r*.log")):
            samp, setup = parse_lk(lg)
            if not samp:
                continue
            d = dist(samp)
            d.update(method=meth, round=os.path.basename(lg)[:-4],
                     setup_s=setup)
            rows_steps.append(d)
            allsamp += samp
        if allsamp:
            per_method[meth] = allsamp
    # RT variants (per-step arrays)
    for tag, key in (("cache", "RT_A"), ("hybrid", "RT_B")):
        allsamp = []
        for jf in sorted(glob.glob(
                f"{R}/logs/bench_steptimes_{tag}_on_w1_*.json")):
            deltas = json.load(open(jf))["step_deltas"][60:]
            if not deltas:
                continue
            d = dist(deltas)
            d.update(method=key, round=os.path.basename(jf),
                     setup_s=None)
            rows_steps.append(d)
            allsamp += deltas
        if allsamp:
            per_method[key] = allsamp

    with open(T("benchmark_step_times.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "round", "n",
                                          "mean", "median", "std",
                                          "p5", "p95", "setup_s"])
        w.writeheader()
        for r in rows_steps:
            w.writerow(r)

    # normalized 1-GPU projections
    proj = {}
    for meth, steps_key in (("PTQ_R5", "PTQ_R5"), ("PTQ_R6", "PTQ_R6"),
                            ("QAT", "QAT"), ("RT_A", "RT"),
                            ("RT_B", "RT")):
        if meth not in per_method:
            continue
        s = per_method[meth]
        med = st.median(s)
        lo, hi = boot_ci_median(s)
        n_steps = C_STEPS[steps_key]
        proj[meth] = dict(
            method=meth, canonical_steps=n_steps,
            sec_per_step_median=round(med, 4),
            sec_per_step_ci95=f"[{lo:.4f},{hi:.4f}]",
            wall_h_1gpu=round(med * n_steps / 3600, 3),
            wall_h_ci95=f"[{lo*n_steps/3600:.3f},{hi*n_steps/3600:.3f}]",
            gpu_h_1gpu=round(med * n_steps / 3600, 3))
    with open(T("normalized_training_cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(
            next(iter(proj.values())).keys()))
        w.writeheader()
        for v in proj.values():
            w.writerow(v)

    # cost views (A+B primary; RT primary = RT_A per amendment §6)
    r5 = proj["PTQ_R5"]["gpu_h_1gpu"]
    r6 = proj.get("PTQ_R6", {}).get("gpu_h_1gpu")
    qat = proj["QAT"]["gpu_h_1gpu"]
    rt = proj["RT_A"]["gpu_h_1gpu"]
    rt_b = proj.get("RT_B", {}).get("gpu_h_1gpu")
    ptq_total = r5                              # A=0 training; B=R5
    qat_incr = qat
    qat_total = r5 + qat                        # R5 counted ONCE
    with open(T("rotation_training_cost.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "rotation", "canonical_steps",
                    "sec_per_step", "wall_h", "gpu_h", "note"])
        w.writerow(["SEAGLE-PTQ", "R5", 3000,
                    proj["PTQ_R5"]["sec_per_step_median"],
                    proj["PTQ_R5"]["wall_h_1gpu"], r5, "REQUIRED"])
        if r6 is not None:
            w.writerow(["SEAGLE-PTQ", "R6", 3000,
                        proj["PTQ_R6"]["sec_per_step_median"],
                        proj["PTQ_R6"]["wall_h_1gpu"], r6,
                        "OPTIONAL/ablation — NOT in headline"])
        w.writerow(["SEAGLE-QAT", "R5 (reused)", 0, "-", 0, 0,
                    "reuses PTQ's R5; counted once in QAT-total"])
        w.writerow(["SEAGLE-RT", "none in core training", 0, "-", 0, 0,
                    "strict RT trains no draft rotation"])
    with open(T("calibration_cost.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "gpu_h", "provenance"])
        w.writerow(["alpha/GS calibration (PTQ & QAT shared)",
                    HIST["PTQ_alpha_calib_gpuh"],
                    "historical DOCUMENTED (2x9 grid evals); "
                    "calibration, not training"])
        w.writerow(["strict-RT rescue alpha grid (post-hoc arm)", 1.4,
                    "measured this-study eval jobs; not core RT"])
    # ratios
    with open(T("cost_ratios.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "gpu_h_ratio", "normalized_wall_ratio"])
        for nm, den in (("SEAGLE-RT / SEAGLE-PTQ", ptq_total),
                        ("SEAGLE-RT / SEAGLE-QAT incremental",
                         qat_incr),
                        ("SEAGLE-RT / SEAGLE-QAT total", qat_total)):
            w.writerow([nm, round(rt / den, 1), round(rt / den, 1)])
    # practical view B
    with open(T("practical_wallclock.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "gpus", "prep_wall_h", "gpu_h", "basis"])
        w.writerow(["SEAGLE-PTQ (R5 + calib)", 1,
                    round(proj["PTQ_R5"]["wall_h_1gpu"] + 3.0, 2),
                    round(r5 + 3.0, 2), "bench-projected + hist calib"])
        w.writerow(["SEAGLE-QAT incremental", 1,
                    proj["QAT"]["wall_h_1gpu"], qat_incr,
                    "bench-projected"])
        w.writerow(["SEAGLE-RT", 8, 33.48, HIST["RT_w8_gpuh"],
                    "ACTUAL completed strict run"])
    # historical cross-check
    with open(T("historical_crosscheck.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "bench_projection", "historical_actual",
                    "rel_err_pct", "note"])
        q = proj["QAT"]["sec_per_step_median"]
        w.writerow(["QAT sec/step", q, round(HIST["QAT_secstep"], 3),
                    round(100 * (q - HIST["QAT_secstep"])
                          / HIST["QAT_secstep"], 1),
                    "AAQ canonical logs 19626s/3/3000 (same GPU class)"])
        if rt_b is not None:
            b = proj["RT_B"]["sec_per_step_median"]
            w.writerow(["RT hybrid w1 sec/step", b,
                        HIST["RT_w1_preflight_hybrid"],
                        round(100 * (b - HIST["RT_w1_preflight_hybrid"])
                              / HIST["RT_w1_preflight_hybrid"], 1),
                        "strict-RT preflight w1 bench"])
            w.writerow(["RT practical GPU-h (w8 actual vs w1-hybrid"
                        " normalized)", round(rt_b, 1),
                        HIST["RT_w8_gpuh"],
                        round(100 * (rt_b - HIST["RT_w8_gpuh"])
                              / HIST["RT_w8_gpuh"], 1),
                        "difference = DDP/comm overhead of world-8 "
                        "(no-P2P 4090s), not benchmark error"])
    # secondary same-batch normalization (§17)
    with open(T("throughput_normalized.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "examples_per_step", "approx_tokens_per_step",
                    "sec_per_step", "tokens_per_s", "opt_steps_per_s"])
        defs = dict(RT_A=(4, 4 * 1485), RT_B=(4, 4 * 1485),
                    QAT=(32, 32 * 48), PTQ_R5=(32, 32 * 48),
                    PTQ_R6=(32, 32 * 48))
        for m, (ex, tok) in defs.items():
            if m in proj:
                sps = proj[m]["sec_per_step_median"]
                w.writerow([m, ex, tok, sps, round(tok / sps, 1),
                            round(1 / sps, 3)])
    # headline summary json for the report writer
    json.dump(dict(proj=proj, ptq_total=ptq_total, qat_incr=qat_incr,
                   qat_total=qat_total, rt=rt, rt_b=rt_b,
                   ratios=dict(rt_over_ptq=round(rt / ptq_total, 1),
                               rt_over_qat_incr=round(rt / qat_incr, 1),
                               rt_over_qat_total=round(rt / qat_total, 1)),
                   hist=HIST),
              open(T("headline_summary.json"), "w"), indent=1)
    print("[analyze] done:", json.dumps(
        {k: v["gpu_h_1gpu"] for k, v in proj.items()}))


if __name__ == "__main__":
    main()
