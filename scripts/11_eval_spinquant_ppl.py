#!/usr/bin/env python
"""SpinQuant PTQ wikitext-2 PPL evaluation (target 5 / task R-2).

Runs SpinQuant's own ptq.py (subprocess, torchrun) for a ladder of settings so we
have trustworthy, upstream-faithful PPL numbers to anchor the EAGLE experiments:
  fp16          : no rotation, w16a16     (baseline)
  w4a16_rot     : rotate + W4 weights only
  w4a4_rot      : rotate + W4A4           (KV16)
  w4a4kv4_rot   : rotate + W4A4KV4
  w4a4_learnedR : rotate + W4A4 with the learned R.bin (if present)

Weight quant defaults to RTN (--w-method gptq for the SpinQuant-default GPTQ).
Runtime mode = fake_quant_pytorch (QDQ, FP16 matmuls; NOT real low-bit speed).

Outputs results/ppl_metrics.jsonl + results/ppl_summary.csv.

Usage:
  python scripts/11_eval_spinquant_ppl.py --settings fp16 w4a4_rot w4a4kv4_rot
  python scripts/11_eval_spinquant_ppl.py --settings all --w-method gptq
"""

import argparse
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eagle_spinquant import experiment, logging_utils, metrics  # noqa: E402
from eagle_spinquant import spinquant_bridge as sb  # noqa: E402

LEARNED_R = os.path.join(PROJECT_ROOT, "outputs", "rotations", "learned_w16a4kv4", "R.bin")

SETTINGS = {
    #             w   a   kv  rotate  learned_R
    "fp16":        (16, 16, 16, False, False),
    "w4a16_rot":   (4, 16, 16, True,  False),
    "w4a4_rot":    (4,  4, 16, True,  False),
    "w4a4kv4_rot": (4,  4,  4, True,  False),
    "w4a4_learnedR": (4, 4, 16, True, True),
    "w4a4kv4_learnedR": (4, 4, 4, True, True),
}


def parse_ppl(log_path: str):
    txt = open(log_path).read() if os.path.isfile(log_path) else ""
    m = re.findall(r"wiki2 ppl is:\s*([0-9.]+)", txt)
    if m:
        return float(m[-1])
    # fallback: tensor(x)
    m = re.findall(r"ppl is:\s*tensor\(([0-9.]+)", txt)
    return float(m[-1]) if m else None


def peak_mem_from_log(log_path):
    return None  # ptq.py doesn't print mem; left null (see docs/00 fake-quant note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--settings", nargs="+", default=["fp16", "w4a4_rot", "w4a4kv4_rot"])
    ap.add_argument("--w-method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.settings == ["all"]:
        args.settings = list(SETTINGS.keys())

    cfg = experiment.load_config(args.config)
    input_model = cfg["model"]["target"]
    access_token = os.environ.get("HF_TOKEN")

    rows = []
    for name in args.settings:
        if name not in SETTINGS:
            print(f"[skip] unknown setting {name}"); continue
        w, a, kv, rotate, learned = SETTINGS[name]
        if learned and not os.path.isfile(LEARNED_R):
            print(f"[skip] {name}: learned R.bin not found at {LEARNED_R}")
            rows.append({"setting": name, "ppl": None, "skipped": "no learned R.bin",
                         "runtime_mode": metrics.RUNTIME_FAKE_QUANT})
            continue
        rot_path = LEARNED_R if learned else None
        log_path = os.path.join(PROJECT_ROOT, "runs", "spinquant_ppl", f"{name}_{args.w_method}.log")
        cmd = sb.build_ptq_eval_cmd(
            input_model=input_model, w_bits=w, a_bits=a, kv_bits=kv,
            optimized_rotation_path=rot_path, w_rtn=(args.w_method == "rtn"),
            rotate=rotate, eval_batch_size=args.eval_batch_size,
            access_token=access_token, master_port=29520)
        print(f"\n=== {name} (w{w}a{a}kv{kv} rotate={rotate} wq={args.w_method}) ===")
        print("CMD:", " ".join(cmd))
        t0 = time.time()
        res = sb.run_spinquant_cmd(cmd, log_path, timeout=args.timeout)
        elapsed = time.time() - t0
        ppl = parse_ppl(log_path)
        row = {
            "setting": name, "w_bits": w, "a_bits": a, "kv_bits": kv,
            "rotate": rotate, "w_method": args.w_method, "learned_R": learned,
            "ppl": ppl, "returncode": res["returncode"], "elapsed_s": round(elapsed, 1),
            "runtime_mode": (metrics.RUNTIME_FP16 if (w == 16 and a == 16)
                             else metrics.RUNTIME_FAKE_QUANT),
            "log": log_path,
        }
        rows.append(row)
        print(f"  ppl={ppl}  rc={res['returncode']}  {elapsed:.0f}s")

    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "ppl_metrics.jsonl"), rows)
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "ppl_summary.csv"), rows)
    print("\n=== PPL summary ===")
    for r in rows:
        print(f"  {r['setting']:20s} ppl={r['ppl']} ({r['runtime_mode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
