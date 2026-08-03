#!/usr/bin/env python
"""Deploy trained rotation arms through the validated evaluator
(--proj-rot* learned_ckpt specs): held-out c4:20 calib AL per arm,
then RCAL captures for the top arms + FIXED-R + EP3-P references.
Selection: max held-out RCAL (spec §22), min-NMSE counterfactual
reported. Never touches MT-Bench for selection.
"""
import argparse, csv, glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
D = 4096
MF, MR = str(D ** 0.40), str(D ** 0.45)


def sched(jobs, gpus, log_dir, ng=1):
    os.makedirs(log_dir, exist_ok=True)
    free = list(gpus); running = []; queue = list(jobs)
    while queue or running:
        for i in range(len(running) - 1, -1, -1):
            pr, name, asg = running[i]
            if pr.poll() is not None:
                print(f"[sched] {'done' if pr.returncode == 0 else 'FAIL'} {name}", flush=True)
                free.extend(asg); running.pop(i)
        while queue and len(free) >= queue[0][2]:
            name, cmd, need = queue.pop(0)
            asg, free = free[:need], free[need:]
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


def rot_args(ck, pathwise):
    a = ["--proj-rot-first",
         json.dumps(dict(learned_ckpt=ck, which="rot_f"))]
    a += ["--proj-rot-rec",
          json.dumps(dict(learned_ckpt=ck,
                          which="rot_r" if pathwise else "rot_f"))]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["heldout", "rcal"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    cks = sorted(glob.glob(os.path.join(rd, "rotations",
                                        "*_last.pt")))
    cks = [c for c in cks if "SMOKE" not in c]
    if args.phase == "heldout":
        jobs = []
        for ck in cks:
            tag = "LR_" + os.path.basename(ck)[:-8]
            pw = "_pw" in ck
            sh = os.path.join(rd, "shards",
                              f"al__{tag}__int4__c4__calib.csv")
            if os.path.exists(sh):
                continue
            jobs.append((tag, [
                PY, "scripts/eval_eagle_acceptance_length.py",
                "--run-dir", rd, "--target", "int4", "--draft-cfg",
                "d4p3_deploy", "--tag", tag, "--datasets", "c4:20",
                "--pool", "calib", "--alpha", MF, "--alpha-rec", MR]
                + rot_args(ck, pw), 1))
        # references on the same held-out set
        for tag, extra in (
                ("LRREF_EP3P", ["--alpha", MF, "--alpha-rec", MR]),
                ("LRREF_FIXEDR", ["--alpha", MF, "--alpha-rec", MR,
                                  "--proj-rot-first",
                                  json.dumps(dict(family="cross",
                                                  block=8192,
                                                  seed=12,
                                                  interleave_chunk=1)),
                                  "--proj-rot-rec",
                                  json.dumps(dict(family="cross",
                                                  block=8192,
                                                  seed=12,
                                                  interleave_chunk=1))
                                  ])):
            sh = os.path.join(rd, "shards",
                              f"al__{tag}__int4__c4__calib.csv")
            if not os.path.exists(sh):
                jobs.append((tag, [
                    PY, "scripts/eval_eagle_acceptance_length.py",
                    "--run-dir", rd, "--target", "int4",
                    "--draft-cfg", "d4p3_deploy", "--tag", tag,
                    "--datasets", "c4:20", "--pool", "calib"]
                    + extra, 1))
        gpus = [int(x) for x in os.environ.get(
            "LRGF_GPUS", "0,1,2,3,4,5,6,7").split(",")]
        sched(jobs, gpus, os.path.join(rd, "logs"))
        res = {}
        for p in glob.glob(os.path.join(
                rd, "shards", "al__LR*__int4__c4__calib.csv")):
            tag = os.path.basename(p).split("__")[1]
            res[tag] = round(tau_of(p), 4)
        json.dump(res, open(os.path.join(
            rd, "tables", "learned_rot_heldout.json"), "w"),
            indent=1)
        print("[heldout]", json.dumps(res, indent=1))
        return 0

    if args.phase == "rcal":
        res = json.load(open(os.path.join(
            rd, "tables", "learned_rot_heldout.json")))
        arms = sorted([(k, v) for k, v in res.items()
                       if k.startswith("LR_")],
                      key=lambda kv: -kv[1])[:3]
        jobs = []

        def cap(tag, extra):
            p = os.path.join(rd, "cycles",
                             f"cyc__{tag}__c4.jsonl")
            if os.path.exists(p):
                return
            jobs.append((f"cap_{tag}", [
                PY, "scripts/capture_eagle_proposal_cycles.py",
                "--run-dir", rd, "--target", "int4", "--draft-cfg",
                "d4p3_deploy", "--tag", tag, "--datasets", "c4",
                "--n-prompts", "20", "--pool", "calib",
                "--max-new-tokens", "96", "--ref-device", "cuda:1"]
                + extra, 2))
        for tag, _ in arms:
            ck = os.path.join(rd, "rotations",
                              tag[3:] + "_last.pt")
            cap("V" + tag, ["--alpha", MF, "--alpha-rec", MR]
                + rot_args(ck, "_pw" in tag))
        cap("VLRREF_EP3P", ["--alpha", MF, "--alpha-rec", MR])
        cap("VLRREF_FIXEDR", [
            "--alpha", MF, "--alpha-rec", MR, "--proj-rot-first",
            json.dumps(dict(family="cross", block=8192, seed=12,
                            interleave_chunk=1)),
            "--proj-rot-rec",
            json.dumps(dict(family="cross", block=8192, seed=12,
                            interleave_chunk=1))])
        gpus = [int(x) for x in os.environ.get(
            "LRGF_GPUS", "0,1,2,3,4,5,6,7").split(",")]
        sched(jobs, gpus, os.path.join(rd, "logs"))
        subprocess.run([PY, "scripts/compute_eagle_rcal_metrics.py",
                        "--run-dir", rd], cwd=ROOT)
        return 0


if __name__ == "__main__":
    sys.exit(main())
