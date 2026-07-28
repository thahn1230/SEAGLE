#!/usr/bin/env python
"""Offline best-validation checkpoint selection (study spec §18).

For one QAT run tag: evaluate every step-stamped checkpoint on the
HELD-OUT calib pool (c4 offset-500 — never test prompts), pick the
highest calib AL (tie-break: earlier step), then evaluate BOTH the best
and the final checkpoint on MT-Bench. Writes
tables/ckpt_selection_<tag>.json.
"""
import argparse, csv, glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tau_of(path):
    taus = [t for r in csv.DictReader(open(path))
            for t in json.loads(r["acceptance_list"])]
    return sum(taus) / max(len(taus), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", required=True, choices=["fp16", "int4"])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    cks = sorted(glob.glob(os.path.join(rd, "ckpts",
                                        f"{args.tag}_step*.pt")),
                 key=lambda p: int(re.search(r"_step(\d+)", p).group(1)))
    ev = [sys.executable,
          os.path.join(ROOT, "scripts", "eval_eagle_acceptance_length.py"),
          "--run-dir", rd, "--target", args.target,
          "--draft-cfg", "d4p3_deploy", "--alpha", str(args.alpha)]
    results = []
    for ck in cks:
        step = int(re.search(r"_step(\d+)", ck).group(1))
        tag = f"CKSEL_{args.tag}_st{step}"
        subprocess.run(ev + ["--draft-sd", ck, "--tag", tag,
                             "--datasets", "c4:20", "--pool", "calib"])
        sh = os.path.join(rd, "shards",
                          f"al__{tag}__{args.target}__c4__calib.csv")
        if os.path.exists(sh):
            results.append(dict(step=step, ckpt=ck,
                                calib_tau=round(tau_of(sh), 4)))
    last = os.path.join(rd, "ckpts", f"{args.tag}_last.pt")
    best = max(results, key=lambda r: (r["calib_tau"], -r["step"])) \
        if results else None
    out = dict(tag=args.tag, target=args.target, alpha=args.alpha,
               grid=results, best=best)
    # test-set evals for best + final
    if best:
        subprocess.run(ev + ["--draft-sd", best["ckpt"],
                             "--tag", f"{args.tag}_bestval",
                             "--datasets", "mtbench"])
    if os.path.exists(last):
        subprocess.run(ev + ["--draft-sd", last,
                             "--tag", f"{args.tag}_final",
                             "--datasets", "mtbench"])
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(
        rd, "tables", f"ckpt_selection_{args.tag}.json"), "w"), indent=1)
    print(f"[cksel] {args.tag}: best={best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
