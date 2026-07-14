#!/usr/bin/env python
"""EAGLE speculative decoding with a REAL packed-INT4 target (no fake quant).

Backends:
  fp16_ref       : unrotated fp16 target (reference arm: vanilla + stock)
  tinygemm_w4a16 : rotated (R1+R2 fused) target, per-layer linears swapped to
                   torch-native tinygemm packed-int4-weight kernels.
                   REAL W4A16 (activations 16-bit). NOT W4A4.
  quarot_w4a4    : rotated (R1+R2 fused, R4 folded into W_down) target,
                   per-layer linears swapped to QuaRot CUTLASS INT4xINT4
                   kernels with per-token int4 activation quantization.
                   REAL W4A4 for those linears; lm_head/embeddings/KV/attn
                   math remain fp16 (stated in every row).

Kernel dispatch is PROVEN per invocation with a torch-profiler capture on a
short warmup generation; kernel names go to profiler_kernel_names.txt and an
int4_dispatch_proven flag goes into every CSV row. If proof fails the rows
are still written but flagged false — never silently trusted.

Usage (one physical GPU per process):
  CUDA_VISIBLE_DEVICES=4 python scripts/run_realint4_study.py \
    --run-id rotstudy_realint4_<ts> --backend tinygemm_w4a16 \
    --variants vanilla,naive,A,B2 --num-prompts 80
"""

import argparse
import json
import os
import shlex
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             realint4, study)

CSV_COLUMNS = [
    "run_id", "timestamp", "gpu_name", "cuda_version", "torch_version",
    "model", "draft", "backend", "quant_mode", "fake_quant", "rotation",
    "rotation_type", "seed", "prompt_set", "prompt_id", "max_new_tokens",
    "variant", "acceptance_length", "generated_tokens", "target_forwards",
    "tokens_per_second", "total_ms", "target_time_ms_total",
    "draft_time_ms_total", "verify_time_ms_total",
    "unrotation_time_ms_total", "unrotation_time_us_per_call",
    "b2_swap_overhead_ms_total", "b2_swap_calls", "peak_mem_gib",
    "post_build_mem_alloc_gib", "int4_dispatch_proven", "notes",
]

QUANT_MODE = {
    "fp16_ref": "fp16",
    "tinygemm_w4a16": "real_w4a16_weight_only",
    "quarot_w4a4": "real_w4a4_linears_pertoken_a4",
}
VARIANT_ORDER = ["stock", "naive", "A", "B2", "vanilla"]


def profile_kernels(model, ids, tree, run_dir, tag, backend):
    """Short profiled generation; returns (proven, top_kernel_names)."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        gen = model.ea_generate(ids, temperature=0.0, max_steps=12,
                                tree_choices=tree)
        for _ in gen:
            pass
    evs = [(ev.key, ev.self_device_time_total) for ev in p.key_averages()
           if ev.self_device_time_total > 0]
    evs.sort(key=lambda kv: -kv[1])
    names = [k for k, _ in evs]
    if backend == "tinygemm_w4a16":
        proven = any("tinygemm" in n.lower() or "int4" in n.lower()
                     for n in names)
    elif backend == "quarot_w4a4":
        proven = any("int4" in n.lower() or "s4" in n.lower()
                     or ("cutlass" in n.lower() and "gemm" in n.lower())
                     or "sym_quant" in n.lower() for n in names)
    else:
        proven = any("tinygemm" in n.lower() or "int4" in n.lower()
                     for n in names)  # should be FALSE for fp16_ref
    with open(os.path.join(run_dir, "profiler_kernel_names.txt"), "a") as f:
        f.write(f"\n# ==== {tag} backend={backend} "
                f"int4_evidence={proven} ====\n")
        for k, us in evs[:40]:
            f.write(f"{k} | {us:.0f} us\n")
    return proven, names[:8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chat-template", default=None,
                    choices=["llama2", "vicuna"])
    ap.add_argument("--backend", required=True,
                    choices=["fp16_ref", "tinygemm_w4a16", "quarot_w4a4"])
    ap.add_argument("--variants", required=True,
                    help="comma list from stock|naive|A|B2|vanilla")
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--vanilla-num-prompts", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-variant-order", action="store_true",
                    help="run variants in the order given (order-confound "
                         "checks) instead of canonical VARIANT_ORDER")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, \
        "real-int4 study jobs use exactly ONE visible physical GPU"
    device = "cuda:0"
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    chat_template = (args.chat_template
                     or cfg.get("model", {}).get("chat_template", "llama2"))
    build_prompt = eagle_bridge.PROMPT_BUILDERS[chat_template]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_ORDER]
    assert not unknown, f"unknown variants {unknown}"
    if not args.keep_variant_order:
        variants = [v for v in VARIANT_ORDER if v in variants]
    if args.backend == "fp16_ref":
        assert all(v in ("stock", "vanilla") for v in variants), \
            "fp16_ref arm is unrotated; only stock/vanilla are meaningful"
    else:
        assert "stock" not in variants, \
            "no 'stock' on a rotated target (that is 'naive')"

    env = logging_utils.env_summary()
    rotation = "none" if args.backend == "fp16_ref" else "r1r2"
    torch.manual_seed(args.seed)

    print(f"[realint4] loading EaModel ({args.backend})...", flush=True)
    t0 = time.time()
    model = eagle_bridge.load_eagle_model(
        base_model_path=paths["target_path"], ea_model_path=paths["draft_path"],
        dtype=torch.float16, device_map=device)
    stash = None
    if rotation != "none":
        r_bin = study.r_bin_path("random_hadamard", args.seed,
                                 paths["target_path"], rotations_root)
        stash = study.apply_rotation_quant(model.base_model, rotation, r_bin,
                                           "none", cfg["model"]["target"],
                                           device)
        model.base_model.to(device)
        model.ea_layer.to(device)
    swap_info = {"post_swap_mem_alloc_gib": ""}
    if args.backend != "fp16_ref":
        print("[realint4] swapping target linears -> real INT4...", flush=True)
        swap_info = realint4.swap_target_linears(model.base_model,
                                                 args.backend, device)
        print(f"[realint4] {swap_info}", flush=True)
    build_s = time.time() - t0
    tok = eagle_bridge.get_tokenizer(model)
    print(f"[realint4] target ready in {build_s:.0f}s", flush=True)

    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)

    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    assert prompts, "no prompts"

    # ---- kernel-dispatch proof (once per invocation, warmup prompt) ----
    ids0 = build_prompt(tok, prompts[0]["text"]).to(device)
    proven, top_names = profile_kernels(
        model, ids0, tree, run_dir, args.tag or args.backend, args.backend)
    if args.backend != "fp16_ref" and not proven:
        print("[realint4] WARNING: NO INT4 KERNEL EVIDENCE IN PROFILE — "
              "rows will be flagged int4_dispatch_proven=false", flush=True)
    print(f"[realint4] int4_dispatch_proven={proven} top={top_names[:4]}",
          flush=True)

    base_row = {
        "run_id": args.run_id, "gpu_name": gpu["names"][0],
        "cuda_version": env.get("torch_cuda"), "torch_version": env.get("torch"),
        "model": cfg["model"]["target"], "draft": cfg["model"]["eagle_draft"],
        "backend": args.backend, "quant_mode": QUANT_MODE[args.backend],
        "fake_quant": "false", "rotation": rotation,
        "rotation_type": "random_hadamard" if rotation != "none" else "none",
        "seed": args.seed, "prompt_set": "mt_bench",
        "max_new_tokens": args.max_new_tokens,
        "post_build_mem_alloc_gib": swap_info.get("post_swap_mem_alloc_gib",
                                                  ""),
        "int4_dispatch_proven": proven,
    }

    rows = []
    for variant in variants:
        adapter = None
        if variant == "B2":
            adapter = realint4.TimedTwoPathAdapter(
                model, stash, device, torch.float16).install()
        elif variant in ("naive", "A"):
            adapter = study.ADAPTERS[variant](
                model, stash, device, torch.float16).install()
        timers = study.PhaseTimers(model).install()
        mode = "baseline" if variant == "vanilla" else "eagle"
        n_prompts = (min(args.vanilla_num_prompts, len(prompts))
                     if variant == "vanilla" else len(prompts))

        wi = build_prompt(tok, prompts[0]["text"])
        study.run_one_prompt(model, timers, wi, mode, 8, tree, max_depth=5)
        if adapter:
            adapter.unrot.reset()
            if hasattr(adapter, "reset_swap"):
                adapter.reset_swap()

        for p in prompts[:n_prompts]:
            ids = build_prompt(tok, p["text"])
            r = study.run_one_prompt(model, timers, ids, mode,
                                     args.max_new_tokens, tree, max_depth=5)
            un = adapter.unrotation_stats() if adapter else \
                {"unrotation_time_ms_total": 0.0, "unrotation_calls": 0}
            sw = adapter.swap_stats() if hasattr(adapter, "swap_stats") else \
                {"b2_swap_overhead_ms_total": "", "b2_swap_calls": ""}
            row = dict(base_row)
            row.update({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt_id": p["question_id"], "variant": variant,
                "acceptance_length": r["acceptance_length"],
                "generated_tokens": r["generated_tokens"],
                "target_forwards": r["target_forwards"],
                "tokens_per_second": r["tokens_per_second"],
                "total_ms": r["total_ms"],
                "target_time_ms_total": r["target_time_ms_total"],
                "draft_time_ms_total": r["draft_time_ms_total"],
                "verify_time_ms_total": r["verify_time_ms_total"],
                "unrotation_time_ms_total": un["unrotation_time_ms_total"],
                "unrotation_time_us_per_call":
                    (un["unrotation_time_ms_total"] * 1e3
                     / un["unrotation_calls"]) if un["unrotation_calls"] else "",
                "b2_swap_overhead_ms_total": sw["b2_swap_overhead_ms_total"],
                "b2_swap_calls": sw["b2_swap_calls"],
                "peak_mem_gib": r["peak_mem_gib"],
                "notes": json.dumps({
                    "tag": args.tag, "build_s": round(build_s, 1),
                    "runtime_mode": QUANT_MODE[args.backend],
                    "kv_cache": "fp16", "lm_head": "16bit",
                    "kernel_top": top_names[:4],
                }),
            })
            if adapter:
                adapter.unrot.reset()
                if hasattr(adapter, "reset_swap"):
                    adapter.reset_swap()
            rows.append(row)
        acc = [x["acceptance_length"] for x in rows
               if x["variant"] == variant
               and isinstance(x["acceptance_length"], (int, float))]
        tps = [x["tokens_per_second"] for x in rows if x["variant"] == variant]
        print(f"[realint4] {variant}: mean_accept="
              f"{(sum(acc)/len(acc)) if acc else float('nan'):.3f} "
              f"mean_tok_s={sum(tps)/len(tps):.2f} ({n_prompts} prompts)",
              flush=True)
        timers.uninstall()
        if adapter:
            adapter.uninstall()

    tag = args.tag or f"{args.backend}_{'-'.join(variants)}_s{args.seed}"
    shard = os.path.join(run_dir, "shards", f"realint4__{tag}.csv")
    logging_utils.write_csv(shard, rows, CSV_COLUMNS)
    print(f"[realint4] wrote {len(rows)} rows -> {shard}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
