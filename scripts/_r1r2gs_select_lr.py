#!/usr/bin/env python
"""Select the pilot LR per arm (A2/A3) by held-out best-val loss and
emit the full 9-job training list. Never touches eval datasets."""
import argparse, json, os, sys

import torch

ALPHA = "32.89964245299412"          # 4096**0.42, canonical T4 GS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    sel = {}
    for arm in ("A2", "A3"):
        best = None
        rows = []
        for lr_tag, lr in (("1e4", "1e-4"), ("3e4", "3e-4"),
                           ("1e3", "1e-3")):
            p = os.path.join(rd, "rotations",
                             f"PILOT_{arm}_lr{lr_tag}.pt")
            if not os.path.exists(p):
                print(f"[lrsel] MISSING {p}")
                return 1
            d = torch.load(p, map_location="cpu", weights_only=False)
            bv = d["meta"]["best_val"]
            rows.append(dict(lr=lr, val_loss=bv["val_loss"],
                             val_exptau=bv["val_expected_tau"],
                             best_step=bv["step"]))
            if best is None or bv["val_loss"] < best[1]:
                best = (lr, bv["val_loss"])
        sel[arm] = dict(chosen_lr=best[0], rows=rows)
        print(f"[lrsel] {arm}: chosen lr={best[0]} "
              f"(val_loss {best[1]:.5f}) from {rows}")
    json.dump(sel, open(os.path.join(rd, "tables",
                                     "lr_selection.json"), "w"),
              indent=1)

    corpus = os.path.join(rd, "manifests", "lkcorpus__rd_gs_all.json")
    base = (f"python scripts/train_eagle_lk_rotation.py --run-dir {rd} "
            f"--corpus {corpus} --batch 32 --accum 1 --steps 3000 "
            f"--eval-every 100 --teacher t4 --kv-bits 16 "
            f"--alpha-init {ALPHA} --objective hybrid --K 4 "
            f"--save-best-val --val-metric loss")
    jobs = []

    def add(cmd, out_path):
        # skip jobs already completed or currently running via an
        # early-start marker (<out>.running)
        if os.path.exists(out_path) or os.path.exists(
                out_path + ".running"):
            print(f"[lrsel] skip (exists/running): {out_path}")
            return
        jobs.append(cmd)

    for seed in (1001, 1002, 1003):
        o1 = f"{rd}/rotations/RD_GS_A1_s{seed}.pt"
        add(f"{base} --rot residual --r2-mode frozen --lr 3e-4 "
            f"--seed {seed} --out {o1}", o1)
        o2 = f"{rd}/rotations/RD_GS_A2_s{seed}.pt"
        add(f"{base} --rot shared --r2-mode residual "
            f"--lr {sel['A2']['chosen_lr']} --seed {seed} --out {o2}",
            o2)
        o3 = f"{rd}/rotations/RD_GS_A3_s{seed}.pt"
        add(f"{base} --rot residual --r2-mode residual "
            f"--lr {sel['A3']['chosen_lr']} --seed {seed} --out {o3}",
            o3)
    out = os.path.join(rd, "configs", "jobs_train.txt")
    with open(out, "w") as f:
        f.write("\n".join(jobs) + "\n")
    print(f"[lrsel] {len(jobs)} train jobs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
