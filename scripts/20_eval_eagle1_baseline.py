#!/usr/bin/env python
"""EAGLE-1 FP16 baseline (target 6 / task M-2).

Runs stock EAGLE-1 speculative decoding vs vanilla autoregressive decoding on the
same prompts, reporting latency, throughput, average acceptance length, per-depth
acceptance, and peak memory. This is the reference the rotated/quantized variants
are compared against.

runtime_mode = fp16_bf16_baseline (real FP16 execution; the ONLY setting here that
is a genuine hardware speed number, since no quantization is involved).

Usage:
  python scripts/20_eval_eagle1_baseline.py --smoke          # 2 prompts, short
  python scripts/20_eval_eagle1_baseline.py --num-prompts 20 --max-new-tokens 256
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
from eagle_spinquant import eagle_bridge, experiment, logging_utils, metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-name", default="eagle_fp_baseline")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    if args.smoke:
        args.num_prompts = 2
        args.max_new_tokens = 48

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_name)
    logger = logging_utils.RunLogger(run_dir, args.run_name,
                                     config={"cfg": cfg, "args": vars(args), "paths": paths})

    print(f"loading EAGLE model\n  target={paths['target_path']}\n  draft={paths['draft_path']}")
    model = eagle_bridge.load_eagle_model(
        base_model_path=paths["target_path"], ea_model_path=paths["draft_path"],
        dtype=dtype, device_map="cuda")
    tok = eagle_bridge.get_tokenizer(model)

    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    if not prompts:
        prompts = [{"question_id": 0, "category": "smoke",
                    "text": "Compose an engaging travel blog post about Hawaii."}]
    print(f"{len(prompts)} prompts; max_new_tokens={args.max_new_tokens}")

    # warmup (compile/caches) — excluded from measurements
    warm_ids = eagle_bridge.build_llama2_chat_prompt(tok, prompts[0]["text"])
    for mode in ("eagle", "baseline"):
        eagle_bridge.generate_and_measure(model, warm_ids, mode=mode,
                                          max_new_tokens=8, warmup=True)

    rows = []
    for p in prompts:
        input_ids = eagle_bridge.build_llama2_chat_prompt(tok, p["text"])
        ctx_len = input_ids.shape[1]
        for mode in ("baseline", "eagle"):
            r = eagle_bridge.generate_and_measure(
                model, input_ids, mode=mode, max_new_tokens=args.max_new_tokens,
                temperature=cfg["evaluation"]["temperature"])
            r.update({
                "question_id": p["question_id"], "category": p["category"],
                "context_len": ctx_len, "batch_size": 1, "dtype": args.dtype,
                "runtime_mode": metrics.RUNTIME_FP16, "target_quant": "none",
            })
            logger.log(**r)
            rows.append(r)
            print(f"  q{p['question_id']:>3} {mode:8s} "
                  f"tok/s={r['tokens_per_s']:.1f} "
                  f"accept_len={r['avg_accept_length']:.2f} "
                  f"ms/tok={r['ms_per_token']:.2f}")

    # summary: mean per mode + speedup
    summary = summarize(rows)
    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "eagle_fp_baseline.jsonl"), rows)
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "eagle_fp_baseline_summary.csv"),
                            summary)
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['mode']:8s} mean tok/s={s['mean_tokens_per_s']:.1f} "
              f"mean accept_len={s['mean_accept_length']:.2f}")
    if len(summary) == 2:
        spd = _speedup(summary)
        print(f"  EAGLE token-level speedup (tok/s ratio) = {spd:.2f}x")
    return 0


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for mode in ("baseline", "eagle"):
        m = [r for r in rows if r["mode"] == mode]
        if not m:
            continue
        def mean(key):
            vals = [r[key] for r in m if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        out.append({
            "mode": mode, "n": len(m),
            "mean_tokens_per_s": mean("tokens_per_s"),
            "mean_ms_per_token": mean("ms_per_token"),
            "mean_accept_length": mean("avg_accept_length"),
            "mean_peak_mem_gib": mean("peak_mem_gib"),
            "runtime_mode": metrics.RUNTIME_FP16,
        })
    return out


def _speedup(summary: list[dict]) -> float:
    b = next(s for s in summary if s["mode"] == "baseline")["mean_tokens_per_s"]
    e = next(s for s in summary if s["mode"] == "eagle")["mean_tokens_per_s"]
    return e / b if b else float("nan")


if __name__ == "__main__":
    sys.exit(main())
