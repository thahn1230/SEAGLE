#!/usr/bin/env python
"""Rotation-study runner: ONE target build per invocation, multiple variants
run against it over a prompt set; per-prompt rows appended to a shard CSV
(runs/<run_id>/shards/). scripts/summarize_results.py merges shards into the
canonical CSVs and computes statistics.

GPU policy: refuses to run unless CUDA_VISIBLE_DEVICES is a non-empty subset
of physical GPUs {4,5,6,7}. Uses logical cuda:0 of the visible set.

Example (single condition batch):
  CUDA_VISIBLE_DEVICES=4 python scripts/run_eagle_rotation_study.py \
    --run-id rotstudy_x --csv acceptance_main \
    --variants naive,A,B2,B,vanilla --quant none --rotation full \
    --rotation-type random_hadamard --seed 0 --num-prompts 80
"""

import argparse
import copy
import json
import os
import shlex
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study  # noqa: E402

CSV_COLUMNS = [
    "run_id", "timestamp", "git_commit_eagle", "git_commit_local", "gpu_name",
    "cuda_version", "torch_version", "transformers_version", "model", "draft",
    "prompt_set", "prompt_id", "seed", "max_new_tokens", "tree_depth", "top_k",
    "variant", "quant", "rotation", "rotation_type", "fake_quant",
    "accepted_tokens_total", "target_forwards", "acceptance_length",
    "generated_tokens", "vanilla_tokens_per_second",
    "speculative_tokens_per_second", "relative_speedup_same_runtime",
    "unrotation_time_ms_total", "unrotation_time_us_per_call",
    "draft_time_ms_total", "target_time_ms_total", "verify_time_ms_total",
    "notes",
]

# execution order inside one build: non-destructive first, C restores draft,
# B restores fc on uninstall, vanilla last (no adapter interference)
VARIANT_ORDER = ["stock", "naive", "A", "A_nogamma", "B2", "C", "B", "vanilla"]


def load_wikitext_prompts(n):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    paras = [t.strip() for t in ds["text"] if len(t.strip()) > 400][:n]
    return [{"question_id": 10_000 + i, "category": "wikitext",
             "text": "Continue the following text as naturally as possible:\n\n"
                     + p[:400]} for i, p in enumerate(paras)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv", required=True,
                    choices=["acceptance_main", "depth_sweep", "gamma_ablation",
                             "rotation_component_ablation"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--config", default=None,
                    help="experiment yaml (default: configs/default_experiment.yaml)")
    ap.add_argument("--chat-template", default=None,
                    choices=["llama2", "vicuna"],
                    help="prompt template; default = config model.chat_template "
                         "or llama2")
    ap.add_argument("--prompt-set", default="mt_bench", choices=["mt_bench", "wikitext"])
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--vanilla-num-prompts", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tree-depths", default="5",
                    help="comma list; 5 = full mc_sim_7b_63 tree")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--variants", required=True,
                    help="comma list from stock|naive|A|A_nogamma|B|B2|C|vanilla")
    ap.add_argument("--quant", default="none",
                    choices=["none", "w4a16", "w4a4", "w4a4kv4"])
    ap.add_argument("--rotation", default="full",
                    choices=["none", "r1", "r1r2", "r1r2r3r4", "full"])
    ap.add_argument("--rotation-type", default="random_hadamard",
                    choices=["random_hadamard", "learned"])
    ap.add_argument("--fake-quant", default="true")
    ap.add_argument("--draft-ckpt", default=os.path.join(
        PROJECT_ROOT, "outputs", "draft_ckpts", "variantC_draft_full",
        "draft_rotated.pt"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    gpu_info = study.assert_gpu_policy()
    device = "cuda:0"
    assert args.top_k == 10, "EAGLE v1 top_k is a module constant (10)"

    run_dir = args.output_dir or os.path.join(PROJECT_ROOT, "runs", args.run_id)
    shard_dir = os.path.join(run_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    # record the exact command
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    target_path = args.model or paths["target_path"]
    draft_path = args.draft or paths["draft_path"]
    input_model_id = cfg["model"]["target"]
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    chat_template = (args.chat_template
                     or cfg.get("model", {}).get("chat_template", "llama2"))
    build_prompt = eagle_bridge.PROMPT_BUILDERS[chat_template]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_ORDER]
    assert not unknown, f"unknown variants {unknown}"
    variants = [v for v in VARIANT_ORDER if v in variants]
    depths = [int(d) for d in args.tree_depths.split(",")]

    rotated = args.rotation != "none"
    for v in variants:
        if v in ("naive", "A", "A_nogamma", "B", "B2", "C"):
            assert rotated, f"variant {v} requires a rotated target"

    env = logging_utils.env_summary()
    base_row = {
        "run_id": args.run_id, "git_commit_eagle": env.get("eagle_commit"),
        "git_commit_local": logging_utils.git_commit(PROJECT_ROOT) or "n/a",
        "gpu_name": gpu_info["names"][0], "cuda_version": env.get("torch_cuda"),
        "torch_version": env.get("torch"),
        "transformers_version": env.get("transformers"),
        "model": input_model_id, "draft": cfg["model"]["eagle_draft"],
        "prompt_set": args.prompt_set, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "top_k": args.top_k,
        "quant": args.quant, "rotation": args.rotation,
        "rotation_type": args.rotation_type if rotated else "none",
        "fake_quant": "true" if args.quant != "none" else "false",
    }

    if args.prompt_set == "mt_bench":
        prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    else:
        prompts = load_wikitext_prompts(args.num_prompts)
    assert prompts, "no prompts loaded"
    print(f"[runner] {len(prompts)} prompts | variants={variants} | "
          f"quant={args.quant} rotation={args.rotation}/{args.rotation_type} "
          f"seed={args.seed} depths={depths}", flush=True)

    t0 = time.time()
    model, stash, r_bin = study.build_study_target(
        target_path, draft_path, input_model_id, args.rotation,
        args.rotation_type, args.quant, args.seed, device=device,
        rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    print(f"[runner] target built in {time.time()-t0:.0f}s (r_bin={r_bin})", flush=True)

    orig_draft_sd = None
    if "C" in variants:
        orig_draft_sd = {k: v.detach().cpu().clone()
                         for k, v in model.ea_layer.state_dict().items()}

    rows = []
    dtype = torch.float16

    for depth in depths:
        tree = study.truncate_tree(depth) if depth < 5 else None
        if tree is None:
            from eagle.model.choices import mc_sim_7b_63 as tree_full
            tree = [list(p) for p in tree_full]
        study.set_draft_tree(model, tree, device)
        # force target-side tree buffer rebuild for this tree
        if hasattr(model, "tree_choices"):
            del model.tree_choices

        for variant in variants:
            adapter = None
            if variant == "C":
                sd = torch.load(args.draft_ckpt, map_location=device,
                                weights_only=True)
                model.ea_layer.load_state_dict(sd, strict=False)
                model.ea_layer.to(dtype)
                study.set_draft_tree(model, tree, device)
                adapter = study.ADAPTERS["A"](model, stash, device, dtype).install()
            elif variant in study.ADAPTERS:
                adapter = study.ADAPTERS[variant](model, stash, device, dtype).install()
            # stock / vanilla: no adapter

            timers = study.PhaseTimers(model).install()
            mode = "baseline" if variant == "vanilla" else "eagle"
            n_prompts = (min(args.vanilla_num_prompts, len(prompts))
                         if variant == "vanilla" else len(prompts))

            # warmup (excluded from rows; clear its adapter accounting too)
            wi = build_prompt(tok, prompts[0]["text"])
            study.run_one_prompt(model, timers, wi, mode, 8, tree,
                                 max_depth=depth)
            if adapter:
                adapter.unrot.reset()

            for p in prompts[:n_prompts]:
                ids = build_prompt(tok, p["text"])
                r = study.run_one_prompt(model, timers, ids, mode,
                                         args.max_new_tokens, tree,
                                         max_depth=depth)
                un = adapter.unrotation_stats() if adapter else \
                    {"unrotation_time_ms_total": 0.0, "unrotation_calls": 0}
                row = dict(base_row)
                row.update({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "prompt_id": p["question_id"], "tree_depth": depth,
                    "variant": variant,
                    "accepted_tokens_total": r["accepted_tokens_total"],
                    "target_forwards": r["target_forwards"],
                    "acceptance_length": r["acceptance_length"],
                    "generated_tokens": r["generated_tokens"],
                    "vanilla_tokens_per_second":
                        r["tokens_per_second"] if variant == "vanilla" else "",
                    "speculative_tokens_per_second":
                        r["tokens_per_second"] if variant != "vanilla" else "",
                    "relative_speedup_same_runtime": "",
                    "unrotation_time_ms_total": un["unrotation_time_ms_total"],
                    "unrotation_time_us_per_call":
                        (un["unrotation_time_ms_total"] * 1e3 / un["unrotation_calls"])
                        if un["unrotation_calls"] else "",
                    "draft_time_ms_total": r["draft_time_ms_total"],
                    "target_time_ms_total": r["target_time_ms_total"],
                    "verify_time_ms_total": r["verify_time_ms_total"],
                    "notes": json.dumps({
                        "alpha_by_depth": r["alpha_by_depth"],
                        "peak_mem_gib": r["peak_mem_gib"],
                        "total_ms": r["total_ms"], "r_bin": r_bin,
                        "tag": args.tag,
                        "runtime_mode": ("fake_quant_pytorch"
                                         if args.quant != "none"
                                         else "fp16_bf16"),
                    }),
                })
                # reset per-prompt adapter unrotation accounting
                if adapter:
                    adapter.unrot.reset()
                rows.append(row)
            print(f"[runner] depth={depth} variant={variant}: "
                  f"mean_accept="
                  f"{_mean([x['acceptance_length'] for x in rows if x['variant']==variant and x['tree_depth']==depth]):.3f} "
                  f"({n_prompts} prompts)", flush=True)

            timers.uninstall()
            if adapter:
                adapter.uninstall()
            if variant == "C" and orig_draft_sd is not None:
                model.ea_layer.load_state_dict(orig_draft_sd, strict=True)
                model.ea_layer.to(dtype)
                study.set_draft_tree(model, tree, device)

    tag = args.tag or f"{'-'.join(variants)}_{args.quant}_{args.rotation}_" \
                      f"{args.rotation_type}_s{args.seed}_d{args.tree_depths.replace(',', '')}" \
                      f"_{args.prompt_set}"
    shard = os.path.join(shard_dir, f"{args.csv}__{tag}.csv")
    logging_utils.write_csv(shard, rows, CSV_COLUMNS)
    print(f"[runner] wrote {len(rows)} rows -> {shard}", flush=True)
    return 0


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    sys.exit(main())
