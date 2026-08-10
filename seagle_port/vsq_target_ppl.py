"""T-PPL gate (§6/§27 of the VSQ update): target rotation quality ladder.

Arms: T-PPL0 fp16 | T-PPL1 W4A4 RTN no rotation | T-PPL2 W4A4 random-
Hadamard rotation | T-PPL3 W4A4 learned R1 only | T-PPL4 W4A4 learned
R1+R2 (Vanilla SpinQuant primary). Frozen protocol: wikitext-2 test,
16x2048 windows (extended from gate_b's 8 for stability) + FP rotation-
equivalence check (argmax agree + logit maxabs) for the learned R.bin.

Usage: python -m seagle_port.vsq_target_ppl --rbin <learned R.bin> \
    --seed-tag s0 --run-dir RD [--windows 16]
Writes/updates run-dir/tables/target_ppl.csv (one row per arm per seed).
R1-only variant R.bin (identity R2) is minted next to the learned one.
"""
import argparse
import csv
import json
import os

import torch

from . import spinquant_target as sq
from .gate_b import ppl_wikitext2, capture, _prompts

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HAD = "/home/thahn1230/dflash_workspace/outputs/rotations/llama31_hadamard/R.bin"


def mint_r1_only(rbin_path):
    out = rbin_path.replace("R.bin", "R1only.bin")
    if not os.path.exists(out):
        rb = sq.load_rbin(rbin_path)
        rb2 = {"R1": rb["R1"]}
        for k in rb:
            if k.endswith("self_attn.R2"):
                rb2[k] = torch.eye(rb[k].shape[0])
        torch.save(rb2, out)
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--seed-tag", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arm", default=None,
                    help="run a single arm in this process (memory-safe); "
                         "omit to orchestrate all arms via subprocesses")
    args = ap.parse_args()
    if args.arm is None:
        import subprocess, sys as _sys
        for arm in ("T-PPL0_fp16", "T-PPL1_rtn_norot", "T-PPL2_hadamard",
                    "T-PPL3_R1only", "T-PPL4_R1R2"):
            subprocess.run([_sys.executable, "-m",
                            "seagle_port.vsq_target_ppl", "--rbin",
                            args.rbin, "--seed-tag", args.seed_tag,
                            "--run-dir", args.run_dir, "--windows",
                            str(args.windows), "--device", args.device,
                            "--arm", arm], check=True)
        _verdict(args)
        return
    dev = args.device
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    all_arms = [
        ("T-PPL0_fp16", "fp16", None),
        ("T-PPL1_rtn_norot", "w4a4_norot", None),
        ("T-PPL2_hadamard", "w4a4", HAD),
        ("T-PPL3_R1only", "w4a4", mint_r1_only(args.rbin)),
        ("T-PPL4_R1R2", "w4a4", args.rbin),
    ]
    arms = [a for a in all_arms if a[0] == args.arm]
    for name, mode, rbin in arms:
        m = sq.build_target(MODEL, mode, rbin_path=rbin, device=dev)
        ppl = ppl_wikitext2(m, tok, dev, n_windows=args.windows)
        row = {"arm": name, "seed": args.seed_tag, "mode": mode,
               "rbin": rbin or "", "wikitext2_ppl": round(ppl, 4)}
        if name == "T-PPL0_fp16":
            ref = capture(m, tok, _prompts(4), dev)
            torch.save(ref, os.path.join(args.run_dir, "tables",
                                         "tppl_fp16_capture.pt"))
            row["fp_equiv"] = "ref"
        elif name == "T-PPL4_R1R2":
            del m
            torch.cuda.empty_cache()
            mr = sq.build_target(MODEL, "rot_fp16", rbin_path=rbin,
                                 device=dev)
            cap = capture(mr, tok, _prompts(4), dev)
            fp_ref = torch.load(os.path.join(args.run_dir, "tables",
                                             "tppl_fp16_capture.pt"),
                                weights_only=False)
            agree = sum((a["logits"].argmax(-1) == b["logits"].argmax(-1))
                        .float().mean().item()
                        for a, b in zip(fp_ref, cap)) / len(cap)
            dmax = max((a["logits"] - b["logits"]).abs().max().item()
                       for a, b in zip(fp_ref, cap))
            row["fp_equiv"] = f"argmax_agree={agree:.4f};logit_maxabs={dmax:.3f}"
            m = mr
        rows.append(row)
        print(row, flush=True)
        del m
        torch.cuda.empty_cache()
    out = os.path.join(args.run_dir, "tables", "target_ppl.csv")
    exists = os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "seed", "mode", "rbin",
                                          "wikitext2_ppl", "fp_equiv"])
        if not exists:
            w.writeheader()
        for r in rows:
            r.setdefault("fp_equiv", "")
            w.writerow(r)


def _verdict(args):
    import csv as _csv
    rows = [r for r in _csv.DictReader(
        open(os.path.join(args.run_dir, "tables", "target_ppl.csv")))
        if r["seed"] == args.seed_tag]
    g = lambda suf: float(next(r["wikitext2_ppl"] for r in rows
                               if r["arm"].endswith(suf)))
    fp16, rtn, had, r1r2 = g("fp16"), g("norot"), g("hadamard"), g("R1R2")
    verdict = "PASS" if (r1r2 < rtn and r1r2 < had) else "FAIL"
    print(f"GATE C ({args.seed_tag}): {verdict} "
          f"(fp16 {fp16} rtn {rtn} had {had} R1R2 {r1r2})")


if __name__ == "__main__":
    main()
