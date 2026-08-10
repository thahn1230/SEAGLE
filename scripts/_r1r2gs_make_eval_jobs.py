#!/usr/bin/env python
"""Generate the confirmatory eval job list for the GS R1/R2 study.

Scans <run>/rotations/RD_GS_A{1,2,3}_s*.pt, reports each arm's per-seed
best-val loss and the MEDIAN-val seed (the pre-registered primary), and
writes <run>/configs/jobs_eval.txt: A0 (d4p3 baseline) + every seed
checkpoint (rot arm) x the four confirmatory datasets, one job per line
for _r1r2gs_gpuq.py. Also writes tables/median_seeds.json.
"""
import argparse, glob, json, os, sys

import torch

ALPHA = "32.89964245299412"          # 4096**0.42, canonical T4 GS
DSETS = ["mtbench:80", "gsm8k:200", "sharegpt:80", "humaneval:164"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--extra-ckpts", default="",
                    help="csv tag=path to append (composed/controls)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = args.run_dir
    jobs = []
    for ds in DSETS:
        jobs.append(
            f"python scripts/eval_eagle_acceptance_length.py "
            f"--target int4 --draft-cfg d4p3 --alpha {ALPHA} "
            f"--tag A0_BASE --datasets {ds} --max-new-tokens 128 "
            f"--run-dir {rd}")

    med = {}
    for arm in ("A1", "A2", "A3"):
        cks = sorted(glob.glob(os.path.join(
            rd, "rotations", f"RD_GS_{arm}_s*.pt")))
        cks = [c for c in cks if not c.endswith(".final.pt")]
        seeds = []
        for c in cks:
            d = torch.load(c, map_location="cpu", weights_only=False)
            bv = d["meta"].get("best_val") or {}
            seeds.append(dict(ckpt=c,
                              seed=int(c.rsplit("_s", 1)[1][:-3]),
                              val_loss=bv.get("val_loss"),
                              val_exptau=bv.get("val_expected_tau"),
                              best_step=bv.get("step")))
        seeds.sort(key=lambda s: (s["val_loss"] is None,
                                  s["val_loss"]))
        if seeds:
            med_seed = seeds[len(seeds) // 2]
            med[arm] = dict(median=med_seed, all=seeds)
            print(f"[evaljobs] {arm}: median-val seed "
                  f"s{med_seed['seed']} (val_loss "
                  f"{med_seed['val_loss']}) of "
                  f"{[ (s['seed'], s['val_loss']) for s in seeds ]}")
        for s in seeds:
            tag = f"{arm}_s{s['seed']}"
            for ds in DSETS:
                jobs.append(
                    f"python scripts/eval_eagle_acceptance_length.py "
                    f"--target int4 --draft-cfg rot "
                    f"--ckpt {s['ckpt']} --alpha {ALPHA} --tag {tag} "
                    f"--datasets {ds} --max-new-tokens 128 "
                    f"--run-dir {rd}")
    for spec in [x for x in args.extra_ckpts.split(",") if x]:
        tag, _, path = spec.partition("=")
        for ds in DSETS:
            jobs.append(
                f"python scripts/eval_eagle_acceptance_length.py "
                f"--target int4 --draft-cfg rot --ckpt {path} "
                f"--alpha {ALPHA} --tag {tag} --datasets {ds} "
                f"--max-new-tokens 128 --run-dir {rd}")

    out = args.out or os.path.join(rd, "configs", "jobs_eval.txt")
    with open(out, "w") as f:
        f.write("\n".join(jobs) + "\n")
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    json.dump(med, open(os.path.join(rd, "tables",
                                     "median_seeds.json"), "w"),
              indent=1, default=str)
    print(f"[evaljobs] {len(jobs)} jobs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
