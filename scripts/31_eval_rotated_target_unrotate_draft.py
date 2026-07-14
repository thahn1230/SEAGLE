#!/usr/bin/env python
"""Variant A: interface-preserving unrotation (target 7 / task I-5).

Builds ONE rotated target and evaluates two adapters on it (adapter swap, no
rebuild):
  - naive     (spinquant_target_original_eagle): stock draft fed h_hat -> the break
  - unrotate  (spinquant_target_unrotate_interface): h = (h_hat @ R1^T)*gamma_f

Reports acceptance length, per-depth acceptance, latency, throughput, memory, and
the isolated unrotation overhead. Directly answers research Q1-Q3.

  --stage rotate_only : FP16 rotated target (isolates basis mismatch, no quant)
  --stage full        : adds W4A4(KV4) fake quant (basis + quant effects)

Usage:
  python scripts/31_eval_rotated_target_unrotate_draft.py --smoke
  python scripts/31_eval_rotated_target_unrotate_draft.py --stage full --num-prompts 10
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             variant_eval, spinquant_bridge as sb)
from eagle_spinquant.rotation_interface import DraftInterfaceAdapter  # noqa: E402


def ensure_r_bin(paths, learned=False) -> str:
    if learned:
        p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "learned_w16a4kv4", "R.bin")
        if os.path.isfile(p):
            return p
        print("[warn] learned R.bin not found; falling back to random-Hadamard R")
    p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "random_hadamard", "R.bin")
    if not os.path.isfile(p):
        from transformers import AutoConfig
        conf = AutoConfig.from_pretrained(paths["target_path"])
        sb.make_random_rotation_bin(conf, p, mode="hadamard", seed=0)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--stage", default="rotate_only", choices=["rotate_only", "full"])
    ap.add_argument("--w-method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--kv4", action="store_true", help="use W4A4KV4 (else W4A4)")
    ap.add_argument("--learned-r", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.smoke:
        args.num_prompts = 2
        args.max_new_tokens = 48
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    input_model_id = cfg["model"]["target"]
    r_bin = ensure_r_bin(paths, learned=args.learned_r)

    quant_config = None
    if args.stage == "full":
        quant_config = dict(cfg["quantization"])
        quant_config["w_method"] = args.w_method
        if not args.kv4:
            quant_config["k_bits"] = 16
            quant_config["v_bits"] = 16

    run_name = args.run_name or f"variantA_{args.stage}" + ("_kv4" if args.kv4 else "")
    run_dir = os.path.join(PROJECT_ROOT, "runs", run_name)
    logger = logging_utils.RunLogger(run_dir, run_name,
                                     config={"args": vars(args), "quant": quant_config,
                                             "r_bin": r_bin})

    print(f"building rotated target (stage={args.stage}, quant={quant_config})...")
    # Build once with the 'naive' adapter; swap to 'unrotate' after.
    model, adapter, stash = variant_eval.build_variant_model(
        paths, input_model_id, r_bin, variant="naive", stage=args.stage,
        quant_config=quant_config, dtype=dtype, device="cuda")
    tok = eagle_bridge.get_tokenizer(model)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts) or [
        {"question_id": 0, "category": "smoke", "text": "Explain how a rainbow forms."}]

    all_rows = []
    temperature = cfg["evaluation"]["temperature"]

    # (1) naive
    print("evaluating NAIVE (stock draft fed h_hat)...")
    rows = variant_eval.run_prompts(model, tok, prompts, args.max_new_tokens,
                                    temperature, method="spinquant_target_original_eagle",
                                    stage=args.stage, quant_config=quant_config,
                                    include_baseline=True)
    all_rows += rows

    # (2) swap adapter to unrotate (Variant A) — same rotated model
    adapter.uninstall()
    DraftInterfaceAdapter(model, R1=stash["R1"], gamma_f=stash["gamma_f"],
                          original_head_weight=stash["lm_head_weight"],
                          variant="unrotate").install()
    print("evaluating UNROTATE (Variant A)...")
    rows = variant_eval.run_prompts(model, tok, prompts, args.max_new_tokens,
                                    temperature, method="spinquant_target_unrotate_interface",
                                    stage=args.stage, quant_config=quant_config,
                                    include_baseline=False)
    all_rows += rows

    # unrotation overhead
    overhead = variant_eval.measure_unrotation_overhead(
        D=model.base_model.config.hidden_size,
        R1=stash["R1"].cuda().float(), gamma_f=stash["gamma_f"].cuda().float())
    all_rows.append({"method": "unrotation_overhead_probe", **overhead,
                     "runtime_mode": variant_eval.runtime_mode_for_stage(args.stage)})

    for r in all_rows:
        logger.log(**r)

    summary = summarize(all_rows)
    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "unrotate_interface_eval.jsonl"), all_rows)
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "unrotate_interface_summary.csv"), summary)
    print("\n=== summary (mean accept length / tok/s) ===")
    for s in summary:
        print(f"  {s['method']:42s} accept={fmt(s['mean_accept_length'])} "
              f"tok/s={fmt(s['mean_tokens_per_s'])}")
    print(f"\nunrotation overhead: {overhead['us_per_call']:.1f} us/call "
          f"(seq={overhead['seq']}, D={overhead['D']})")
    return 0


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def summarize(rows):
    methods = []
    for r in rows:
        if r.get("method") and r["method"] not in methods and "overhead" not in r["method"]:
            methods.append(r["method"])
    out = []
    for m in methods:
        mr = [r for r in rows if r.get("method") == m and "avg_accept_length" in r]
        if not mr:
            continue
        def mean(k):
            v = [r[k] for r in mr if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        out.append({"method": m, "n": len(mr),
                    "mean_accept_length": mean("avg_accept_length"),
                    "mean_tokens_per_s": mean("tokens_per_s"),
                    "mean_ms_per_token": mean("ms_per_token"),
                    "runtime_mode": mr[0]["runtime_mode"],
                    "target_quant": mr[0].get("target_quant")})
    return out


if __name__ == "__main__":
    sys.exit(main())
