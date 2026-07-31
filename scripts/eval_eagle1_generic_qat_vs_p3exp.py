#!/usr/bin/env python
"""Method-matrix driver for the Generic-QAT / EP3 / RCAL study.

Phases (run in order; each is idempotent — existing shards/cycles skip):

  ep3g    derive EP3-G global beta per target from the 1-D path grids
          (argmin over common betas of first_nmse + rec_nmse), write
          tables/ep3_selection_<tgt>.json (partial).
  ep3sel  stage-2/3 selection: top-9 (bf,br) pairs on 20-prompt held-out
          calib AL -> top-3 -> full MT-Bench; append to selection json.
  matrix  full-method MT-Bench AL evals (F16/NPTQ/LP3/EP3-G/EP3-P/
          GQAT seeds/LP3-RD/LP3-QAT) under THIS run dir.
  rcal    proposal-cycle captures (2 GPUs each) for the method matrix on
          T1 + T0 control + F16 identity, then metrics + replay-equiv.

Jobs are scheduled over --gpus with per-job CUDA_VISIBLE_DEVICES.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
D = 4096
LEGACY = {"fp16": 45.254834, "int4": 32.0}
LK_RD = os.path.join(ROOT, "runs",
                     "eagle_lk_exactpath_draft_rotation_20260721_151703",
                     "rotations", "LK2_AUXG_s0.pt")
QAT_RUN = os.path.join(ROOT, "runs",
                       "eagle1_official_fromscratch_ptq_vs_qat_"
                       "20260724_213243", "ckpts")


def sched(jobs, gpus, log_dir):
    """jobs: list of (name, cmd(list), n_gpus). Round-robin over gpus."""
    os.makedirs(log_dir, exist_ok=True)
    free = list(gpus)
    running = []   # (proc, name, assigned)
    queue = list(jobs)
    while queue or running:
        for i in range(len(running) - 1, -1, -1):
            pr, name, asg = running[i]
            if pr.poll() is not None:
                if pr.returncode != 0:
                    print(f"[sched] FAIL {name} rc={pr.returncode}",
                          flush=True)
                else:
                    print(f"[sched] done {name}", flush=True)
                free.extend(asg)
                running.pop(i)
        while queue and len(free) >= queue[0][2]:
            name, cmd, ng = queue.pop(0)
            asg, free = free[:ng], free[ng:]
            env = dict(os.environ,
                       CUDA_VISIBLE_DEVICES=",".join(map(str, asg)),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
            lf = open(os.path.join(log_dir, f"{name}.log"), "a")
            pr = subprocess.Popen(cmd, env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((pr, name, asg))
            print(f"[sched] start {name} on {asg}", flush=True)
        time.sleep(10)


def tau_of(path):
    import csv
    taus = [t for r in csv.DictReader(open(path))
            for t in json.loads(r["acceptance_list"])]
    return sum(taus) / max(len(taus), 1)


def ep3g_beta(search):
    g = {}
    for pk in ("first", "rec"):
        for e in search["paths"][pk]["grid"]:
            g.setdefault(round(e["beta"], 4), {})[pk] = e["nmse"]
    common = {b: v for b, v in g.items() if len(v) == 2}
    best = min(common, key=lambda b: common[b]["first"]
               + common[b]["rec"])
    return best, common[best]


def ev_cmd(rd, tgt, cfg, tag, ds="mtbench", pool="eval", alpha=None,
           alpha_rec=None, sd=None, ckpt=None):
    c = [PY, "scripts/eval_eagle_acceptance_length.py", "--run-dir", rd,
         "--target", tgt, "--draft-cfg", cfg, "--tag", tag,
         "--datasets", ds, "--pool", pool]
    if alpha is not None:
        c += ["--alpha", str(alpha)]
    if alpha_rec is not None:
        c += ["--alpha-rec", str(alpha_rec)]
    if sd:
        c += ["--draft-sd", sd]
    if ckpt:
        c += ["--ckpt", ckpt]
    return c


def cap_cmd(rd, tgt, cfg, tag, ds="mtbench", n=40, alpha=None,
            alpha_rec=None, sd=None, ckpt=None, mnt=128):
    c = [PY, "scripts/capture_eagle_proposal_cycles.py", "--run-dir",
         rd, "--target", tgt, "--draft-cfg", cfg, "--tag", tag,
         "--datasets", ds, "--n-prompts", str(n), "--ref-device",
         "cuda:1", "--max-new-tokens", str(mnt)]
    if alpha is not None:
        c += ["--alpha", str(alpha)]
    if alpha_rec is not None:
        c += ["--alpha-rec", str(alpha_rec)]
    if sd:
        c += ["--draft-sd", sd]
    if ckpt:
        c += ["--ckpt", ckpt]
    return c


def load_sel(rd, tgt):
    p = os.path.join(rd, "tables", f"ep3_selection_{tgt}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_sel(rd, tgt, d):
    json.dump(d, open(os.path.join(
        rd, "tables", f"ep3_selection_{tgt}.json"), "w"), indent=1)


def gqat_ckpts(rd, variant):
    """seed ckpts: bestval + final from ckpt_selection tables."""
    out = {}
    for s in range(3):
        tag = f"GQAT_{variant}_s{s}"
        p = os.path.join(rd, "tables", f"ckpt_selection_{tag}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            out[s] = dict(
                best=d["best"]["ckpt"] if d.get("best") else None,
                last=os.path.join(rd, "ckpts", f"{tag}_last.pt"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["ep3g", "ep3sel", "matrix", "rcal",
                             "xds"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    rd = args.run_dir
    gpus = [int(x) for x in args.gpus.split(",")]
    log_dir = os.path.join(rd, "logs")

    if args.phase == "ep3g":
        for tgt in ("fp16", "int4"):
            sp = os.path.join(rd, "tables", f"p3exp_search_{tgt}.json")
            if not os.path.exists(sp):
                print(f"[ep3g] no search for {tgt}, skip")
                continue
            search = json.load(open(sp))
            b, nm = ep3g_beta(search)
            sel = load_sel(rd, tgt)
            sel["ep3g"] = dict(beta=b, m=float(D ** b), nmse=nm)
            sel["top9"] = search["top9"]
            save_sel(rd, tgt, sel)
            print(f"[ep3g] {tgt}: beta={b} m={D**b:.4f} {nm}")
        return 0

    if args.phase == "ep3sel":
        jobs = []
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            for i, pr in enumerate(sel.get("top9", [])):
                tag = f"EP3PSEL_{tgt}_p{i}"
                jobs.append((tag, ev_cmd(
                    rd, tgt, "d4p3_deploy", tag, ds="c4:20",
                    pool="calib", alpha=pr["m_first"],
                    alpha_rec=pr["m_rec"]), 1))
        sched(jobs, gpus, log_dir)
        jobs = []
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            scored = []
            for i, pr in enumerate(sel.get("top9", [])):
                sh = os.path.join(
                    rd, "shards",
                    f"al__EP3PSEL_{tgt}_p{i}__{tgt}__c4__calib.csv")
                if os.path.exists(sh):
                    scored.append(dict(idx=i, **pr,
                                       calib_tau=round(tau_of(sh), 4)))
            scored.sort(key=lambda r: -r["calib_tau"])
            sel["top9_calib"] = scored
            sel["top3"] = scored[:3]
            save_sel(rd, tgt, sel)
            for r in sel["top3"]:
                tag = f"EP3P_{tgt}_p{r['idx']}"
                jobs.append((tag, ev_cmd(
                    rd, tgt, "d4p3_deploy", tag,
                    alpha=r["m_first"], alpha_rec=r["m_rec"]), 1))
        sched(jobs, gpus, log_dir)
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            fin = []
            for r in sel.get("top3", []):
                sh = os.path.join(
                    rd, "shards",
                    f"al__EP3P_{tgt}_p{r['idx']}__{tgt}__mtbench.csv")
                if os.path.exists(sh):
                    fin.append(dict(**r,
                                    mtbench_tau=round(tau_of(sh), 4)))
            fin.sort(key=lambda r: -r["mtbench_tau"])
            sel["top3_mtbench"] = fin
            # provisional primary by AL; final primary applies
            # pre-registered Rule 3 (max AL_q s.t. AFS>=0.95) after RCAL
            sel["primary_by_al"] = fin[0] if fin else None
            save_sel(rd, tgt, sel)
            print(f"[ep3sel] {tgt}: {fin}")
        return 0

    if args.phase == "matrix":
        jobs = []
        # F16 upper anchor + NPTQ / LP3 / EP3-G on both targets
        jobs.append(("MTX_F16", ev_cmd(rd, "fp16", "stock",
                                       "MTX_F16"), 1))
        jobs.append(("MTX_F16D_int4", ev_cmd(rd, "int4", "stock",
                                             "MTX_F16D"), 1))
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            jobs.append((f"MTX_NPTQ_{tgt}", ev_cmd(
                rd, tgt, "naive_w4a4", "MTX_NPTQ"), 1))
            jobs.append((f"MTX_LP3_{tgt}", ev_cmd(
                rd, tgt, "d4p3", "MTX_LP3", alpha=LEGACY[tgt]), 1))
            if sel.get("ep3g"):
                jobs.append((f"MTX_EP3G_{tgt}", ev_cmd(
                    rd, tgt, "d4p3", "MTX_EP3G",
                    alpha=sel["ep3g"]["m"]), 1))
            variant = "T0" if tgt == "fp16" else "T1"
            for s, cks in gqat_ckpts(rd, variant).items():
                for kind in ("best", "last"):
                    if cks.get(kind):
                        jobs.append((
                            f"MTX_GQAT_{variant}_s{s}_{kind}",
                            ev_cmd(rd, tgt, "naive_w4a4",
                                   f"MTX_GQAT_s{s}_{kind}",
                                   sd=cks[kind]), 1))
            qv = "Q0" if tgt == "fp16" else "Q1"
            for s in range(3):
                for kind in ("best", "last"):
                    ck = os.path.join(QAT_RUN, f"{qv}_s{s}_{kind}.pt")
                    if os.path.exists(ck):
                        jobs.append((
                            f"MTX_LP3QAT_{qv}_s{s}_{kind}",
                            ev_cmd(rd, tgt, "d4p3_deploy",
                                   f"MTX_LP3QAT_s{s}_{kind}",
                                   alpha=LEGACY[tgt], sd=ck), 1))
        if os.path.exists(LK_RD):
            jobs.append(("MTX_LP3RD_int4", ev_cmd(
                rd, "int4", "rot", "MTX_LP3RD", ckpt=LK_RD), 1))
        # EP3-P primary-by-AL already evaluated in ep3sel (EP3P_* tags)
        sched(jobs, gpus, log_dir)
        return 0

    if args.phase == "xds":
        # cross-dataset finalists, seed 0, SINGLE-SEED (labeled)
        XDS = "sharegpt:80,c4:80,gsm8k:80,humaneval:80"
        jobs = [("XDS_F16", ev_cmd(rd, "fp16", "stock", "XDS_F16",
                                   ds=XDS), 1)]
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            jobs.append((f"XDS_NPTQ_{tgt}", ev_cmd(
                rd, tgt, "naive_w4a4", "XDS_NPTQ", ds=XDS), 1))
            jobs.append((f"XDS_LP3_{tgt}", ev_cmd(
                rd, tgt, "d4p3", "XDS_LP3", ds=XDS,
                alpha=LEGACY[tgt]), 1))
            if sel.get("ep3g"):
                jobs.append((f"XDS_EP3G_{tgt}", ev_cmd(
                    rd, tgt, "d4p3", "XDS_EP3G", ds=XDS,
                    alpha=sel["ep3g"]["m"]), 1))
            pb = sel.get("primary_rule3") or sel.get("primary_by_al")
            if pb:
                jobs.append((f"XDS_EP3P_{tgt}", ev_cmd(
                    rd, tgt, "d4p3_deploy", "XDS_EP3P", ds=XDS,
                    alpha=pb["m_first"], alpha_rec=pb["m_rec"]), 1))
            variant = "T0" if tgt == "fp16" else "T1"
            cks = gqat_ckpts(rd, variant)
            if 0 in cks and cks[0].get("best"):
                jobs.append((f"XDS_GQAT_{tgt}", ev_cmd(
                    rd, tgt, "naive_w4a4", "XDS_GQAT", ds=XDS,
                    sd=cks[0]["best"]), 1))
            qv = "Q0" if tgt == "fp16" else "Q1"
            ck = os.path.join(QAT_RUN, f"{qv}_s0_best.pt")
            if os.path.exists(ck):
                jobs.append((f"XDS_LP3QAT_{tgt}", ev_cmd(
                    rd, tgt, "d4p3_deploy", "XDS_LP3QAT", ds=XDS,
                    alpha=LEGACY[tgt], sd=ck), 1))
        sched(jobs, gpus, log_dir)
        return 0

    if args.phase == "rcal":
        jobs = []
        n, mnt = 40, 128

        def add(tag, tgt, cfg, **kw):
            jobs.append((f"cap_{tag}", cap_cmd(
                rd, tgt, cfg, tag, n=n, mnt=mnt, **kw), 2))
        # identity + T0 controls
        add("RC_F16", "fp16", "stock")
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            sfx = "T0" if tgt == "fp16" else "T1"
            add(f"RC_NPTQ_{sfx}", tgt, "naive_w4a4")
            add(f"RC_LP3_{sfx}", tgt, "d4p3", alpha=LEGACY[tgt])
            if sel.get("ep3g"):
                add(f"RC_EP3G_{sfx}", tgt, "d4p3",
                    alpha=sel["ep3g"]["m"])
            for r in sel.get("top3_mtbench", []):
                add(f"RC_EP3P_{sfx}_p{r['idx']}", tgt, "d4p3_deploy",
                    alpha=r["m_first"], alpha_rec=r["m_rec"])
            variant = "T0" if tgt == "fp16" else "T1"
            cks = gqat_ckpts(rd, variant)
            if 0 in cks and cks[0].get("best"):
                add(f"RC_GQAT_{sfx}", tgt, "naive_w4a4",
                    sd=cks[0]["best"])
            qv = "Q0" if tgt == "fp16" else "Q1"
            ck = os.path.join(QAT_RUN, f"{qv}_s0_best.pt")
            if os.path.exists(ck):
                add(f"RC_LP3QAT_{sfx}", tgt, "d4p3_deploy",
                    alpha=LEGACY[tgt], sd=ck)
        if os.path.exists(LK_RD):
            add("RC_LP3RD_T1", "int4", "rot", ckpt=LK_RD)
        add("RC_F16D_T1", "int4", "stock")
        sched(jobs, gpus, log_dir)
        # metrics + offline replay equivalence on two spot tags
        subprocess.run([PY, "scripts/compute_eagle_rcal_metrics.py",
                        "--run-dir", rd], cwd=ROOT)
        # pre-registered Rule 3: EP3-P primary = max AL_q subject to
        # AFS >= 0.95 among the top-3 pairs (fallback: max AFS).
        met = json.load(open(os.path.join(rd, "tables",
                                          "rcal_metrics.json")))
        import shutil
        for tgt in ("fp16", "int4"):
            sel = load_sel(rd, tgt)
            sfx = "T0" if tgt == "fp16" else "T1"
            cand = []
            for r in sel.get("top3_mtbench", []):
                mk = f"RC_EP3P_{sfx}_p{r['idx']}__mtbench"
                if mk in met:
                    cand.append(dict(**r, AFS=met[mk]["AFS"],
                                     AL_q=met[mk]["AL_q"],
                                     RCAL=met[mk]["RCAL"]))
            ok = [c for c in cand if c["AFS"] >= 0.95]
            prim = (max(ok, key=lambda c: c["AL_q"]) if ok else
                    max(cand, key=lambda c: c["AFS"]) if cand else
                    None)
            sel["primary_rule3"] = prim
            sel["top3_rcal"] = cand
            save_sel(rd, tgt, sel)
            if prim:
                src = os.path.join(
                    rd, "cycles",
                    f"cyc__RC_EP3P_{sfx}_p{prim['idx']}"
                    f"__mtbench.jsonl")
                dst = os.path.join(rd, "cycles",
                                   f"cyc__RC_EP3P_{sfx}__mtbench.jsonl")
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copyfile(src, dst)
                print(f"[rcal] {tgt} EP3-P primary (Rule 3): "
                      f"pair {prim['idx']} bf={prim['beta_first']} "
                      f"br={prim['beta_rec']} AFS={prim['AFS']:.4f}")
        subprocess.run([PY, "scripts/compute_eagle_rcal_metrics.py",
                        "--run-dir", rd], cwd=ROOT)
        for tag in ("RC_F16", "RC_LP3_T1"):
            subprocess.run([PY,
                            "scripts/"
                            "replay_eagle_proposals_reference_target"
                            ".py", "--run-dir", rd, "--tag", tag,
                            "--max-prompts", "6"],
                           cwd=ROOT,
                           env=dict(os.environ,
                                    CUDA_VISIBLE_DEVICES=str(gpus[0])))
        return 0


if __name__ == "__main__":
    sys.exit(main())
