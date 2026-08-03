#!/usr/bin/env python
"""Stage-5 driver: top-9 calib AL -> top-3 MT-Bench + ablation arms ->
RCAL captures -> bootstrap. GPU scheduler as in the GQ study."""
import argparse, csv, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
D = 4096


def sched(jobs, gpus, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    free = list(gpus); running = []; queue = list(jobs)
    while queue or running:
        for i in range(len(running) - 1, -1, -1):
            pr, name, asg = running[i]
            if pr.poll() is not None:
                print(f"[sched] {'done' if pr.returncode == 0 else 'FAIL'} {name}", flush=True)
                free.extend(asg); running.pop(i)
        while queue and len(free) >= queue[0][2]:
            name, cmd, ng = queue.pop(0)
            asg, free = free[:ng], free[ng:]
            env = dict(os.environ,
                       CUDA_VISIBLE_DEVICES=",".join(map(str, asg)),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
            lf = open(os.path.join(log_dir, f"{name}.log"), "a")
            running.append((subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                cwd=ROOT), name, asg))
            print(f"[sched] start {name} on {asg}", flush=True)
        time.sleep(10)


def tau_of(p):
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return sum(ts) / max(len(ts), 1)


def rot_args(cfg_first, cfg_rec):
    a = []
    if cfg_first and cfg_first.get("family") != "identity":
        a += ["--proj-rot-first", json.dumps(
            {k: cfg_first[k] for k in
             ("family", "block", "seed", "interleave_chunk")
             if k in cfg_first})]
    if cfg_rec and cfg_rec.get("family") != "identity":
        a += ["--proj-rot-rec", json.dumps(
            {k: cfg_rec[k] for k in
             ("family", "block", "seed", "interleave_chunk")
             if k in cfg_rec})]
    return a


def ev(rd, tag, mf, mr, cf, cr, ds="mtbench", pool="eval", cap=False,
       n=40):
    script = ("capture_eagle_proposal_cycles.py" if cap
              else "eval_eagle_acceptance_length.py")
    c = [PY, f"scripts/{script}", "--run-dir", rd, "--target",
         "int4", "--draft-cfg", "d4p3_deploy", "--tag", tag,
         "--alpha", str(mf), "--alpha-rec", str(mr),
         "--datasets", ds] + rot_args(cf, cr)
    if cap:
        c += ["--n-prompts", str(n), "--ref-device", "cuda:1",
              "--max-new-tokens", "128"]
    else:
        c += ["--pool", pool]
    return c


def load_pairs(rd):
    return json.load(open(os.path.join(
        rd, "candidates", "s4_int4_pairs.json")))


def m_of(b):
    return float(D ** b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["calib9", "mtbench", "rcal",
                             "bootstrap"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    gpus = list(range(8))
    log_dir = os.path.join(rd, "logs")
    s4 = load_pairs(rd)

    if args.phase == "calib9":
        jobs = []
        for i, p in enumerate(s4["top9"]):
            jobs.append((f"R9_{i}", ev(
                rd, f"R9_{i}", m_of(p["first"]["beta"]),
                m_of(p["rec"]["beta"]), p["first"], p["rec"],
                ds="c4:20", pool="calib"), 1))
        sched(jobs, gpus, log_dir)
        scored = []
        for i, p in enumerate(s4["top9"]):
            sh = os.path.join(rd, "shards",
                              f"al__R9_{i}__int4__c4__calib.csv")
            if os.path.exists(sh):
                scored.append(dict(idx=i, calib_tau=round(
                    tau_of(sh), 4), **p))
        scored.sort(key=lambda r: -r["calib_tau"])
        json.dump(scored, open(os.path.join(
            rd, "candidates", "s5_calib9.json"), "w"), indent=1)
        print("[calib9]", [(r["idx"], r["calib_tau"])
                           for r in scored])
        return 0

    if args.phase == "mtbench":
        scored = json.load(open(os.path.join(
            rd, "candidates", "s5_calib9.json")))
        top3 = scored[:3]
        best = top3[0]
        cay = {}
        for path in ("first", "rec"):
            rows = [json.load(open(os.path.join(
                rd, "candidates", f"cayley_{path}_s{s}.json")))
                for s in (0, 1, 2)
                if os.path.exists(os.path.join(
                    rd, "candidates", f"cayley_{path}_s{s}.json"))]
            cay[path] = min(rows, key=lambda r: r["j_final"]) \
                if rows else None
        json.dump(cay, open(os.path.join(
            rd, "candidates", "cayley_best.json"), "w"), indent=1)
        jobs = []
        for r in top3:
            jobs.append((f"MTX_R{r['idx']}", ev(
                rd, f"MTX_R{r['idx']}", m_of(r["first"]["beta"]),
                m_of(r["rec"]["beta"]), r["first"], r["rec"]), 1))
        # baselines under THIS run dir
        jobs.append(("MTX_EP3P", ev(rd, "MTX_EP3P", m_of(0.40),
                                    m_of(0.45), None, None), 1))
        jobs.append(("MTX_EP3G", ev(rd, "MTX_EP3G", m_of(0.42),
                                    m_of(0.42), None, None), 1))
        jobs.append(("MTX_NPTQ", [
            PY, "scripts/eval_eagle_acceptance_length.py",
            "--run-dir", rd, "--target", "int4", "--draft-cfg",
            "naive_w4a4", "--tag", "MTX_NPTQ",
            "--datasets", "mtbench", "--pool", "eval"], 1))
        # ablation arms at best betas
        bf, br = best["first"], best["rec"]
        mf, mr = m_of(bf["beta"]), m_of(br["beta"])
        shared = dict(bf)                     # shared Q both paths
        jobs.append(("MTX_SHAREDQ", ev(rd, "MTX_SHAREDQ", mf, mr,
                                       shared, shared), 1))
        dual = dict(family="dual", block=256, seed=14,
                    interleave_chunk=1)
        jobs.append(("MTX_BLOCKDIAG", ev(rd, "MTX_BLOCKDIAG", mf, mr,
                                         dual, dual), 1))
        fullr_f = dict(family="full", block=8192, seed=7)
        fullr_r = dict(family="full", block=8192, seed=10)
        jobs.append(("MTX_FULL", ev(rd, "MTX_FULL", mf, mr, fullr_f,
                                    fullr_r), 1))
        # fixed original betas + best rotation (recal-vs-fixed arm)
        jobs.append(("MTX_FIXEDBETA", ev(rd, "MTX_FIXEDBETA",
                                         m_of(0.40), m_of(0.45),
                                         bf, br), 1))
        sched(jobs, gpus, log_dir)
        res = {}
        for tag in ([f"MTX_R{r['idx']}" for r in top3]
                    + ["MTX_EP3P", "MTX_EP3G", "MTX_NPTQ",
                       "MTX_SHAREDQ", "MTX_BLOCKDIAG", "MTX_FULL",
                       "MTX_FIXEDBETA"]):
            sh = os.path.join(rd, "shards",
                              f"al__{tag}__int4__mtbench.csv")
            if os.path.exists(sh):
                res[tag] = round(tau_of(sh), 4)
        json.dump(dict(top3=top3, mtbench=res, best=best),
                  open(os.path.join(rd, "candidates",
                                    "s5_mtbench.json"), "w"),
                  indent=1)
        print("[mtbench]", res)
        return 0

    if args.phase == "rcal":
        s5 = json.load(open(os.path.join(
            rd, "candidates", "s5_mtbench.json")))
        best = s5["best"]
        bf, br = best["first"], best["rec"]
        mf, mr = m_of(bf["beta"]), m_of(br["beta"])
        jobs = [
            ("cap_RC2_EP3P", ev(rd, "RC2_EP3P", m_of(0.40),
                                m_of(0.45), None, None, cap=True), 2),
            ("cap_RC2_REP3P", ev(rd, "RC2_REP3P", mf, mr, bf, br,
                                 cap=True), 2),
            ("cap_RC2_SHAREDQ", ev(rd, "RC2_SHAREDQ", mf, mr,
                                   dict(bf), dict(bf), cap=True), 2),
            ("cap_RC2_BLOCKDIAG", ev(
                rd, "RC2_BLOCKDIAG", mf, mr,
                dict(family="dual", block=256, seed=14),
                dict(family="dual", block=256, seed=14),
                cap=True), 2),
            ("cap_RC2_FULL", ev(rd, "RC2_FULL", mf, mr,
                                dict(family="full", block=8192,
                                     seed=7),
                                dict(family="full", block=8192,
                                     seed=10), cap=True), 2),
            ("cap_RC2_FIXEDBETA", ev(rd, "RC2_FIXEDBETA",
                                     m_of(0.40), m_of(0.45), bf, br,
                                     cap=True), 2),
        ]
        sched(jobs, gpus, log_dir)
        return 0

    if args.phase == "bootstrap":
        tags = ("RC2_EP3P,RC2_REP3P,RC2_SHAREDQ,RC2_BLOCKDIAG,"
                "RC2_FULL,RC2_FIXEDBETA,RC_EP3P_T1,RC_EP3G_T1,"
                "RC_NPTQ_T1,RC_LP3RD_T1,RC_LP3QAT_T1")
        pairs = ("RC2_EP3P:RC2_REP3P,RC2_BLOCKDIAG:RC2_REP3P,"
                 "RC2_REP3P:RC2_FULL,RC2_SHAREDQ:RC2_REP3P,"
                 "RC2_FIXEDBETA:RC2_REP3P,RC_LP3RD_T1:RC2_REP3P,"
                 "RC2_REP3P:RC_LP3QAT_T1,RC2_EP3P:RC_EP3P_T1")
        subprocess.run([PY, "scripts/bootstrap_eagle_rcal.py",
                        "--run-dir", rd, "--tags", tags,
                        "--pairs", pairs, "--dataset", "mtbench"],
                       cwd=ROOT)
        return 0


if __name__ == "__main__":
    sys.exit(main())
