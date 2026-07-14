#!/usr/bin/env python
"""Rotation-aware EAGLE variant runner (Phases 2-4 of the rotation-aware study).

Variants:
  old  : stock naive A B B2 vanilla        (unchanged baselines)
  new  : D1 D2                             (rotated-loop oracles)
         E_r1 E_r1_gamma E_r1_cofold       (embedding-branch probes, D1 loop)
         F_R_gamma F_R_only                (algebraic fully rotated draft)

One target build per invocation; per-prompt rows to
runs/<run-id>/shards/<csv>__<tag>.csv with the spec §6 schema.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_rotation_aware_study.py \
    --run-id rotation_aware_eagle_<ts> --csv rotated_loop_oracle \
    --variants naive,A,B,B2,D1,D2 --num-prompts 20
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
                             rotation_aware as ra, study)

CSV_COLUMNS = [
    "model_pair", "variant", "prompt_id", "seed", "tree_depth", "quant",
    "rotation", "acceptance_length", "generated_tokens", "target_forwards",
    "tok_per_s", "draft_time_ms", "target_time_ms", "verify_time_ms",
    "basis_conversion_time_ms", "number_of_basis_conversions",
    "original_lm_head_used", "rotated_lm_head_used", "embedding_basis",
    "recycled_feature_basis", "notes",
]

ORDER = ["stock", "naive", "A", "B", "B2", "D1", "D2", "E_r1", "E_r1_gamma",
         "E_r1_cofold", "F_R_gamma", "F_R_only", "G_native", "vanilla"]

DEFAULT_META = {
    "stock": dict(original_lm_head_used=True, rotated_lm_head_used=False,
                  embedding_basis="original", recycled_feature_basis="original"),
    "naive": dict(original_lm_head_used=True, rotated_lm_head_used=False,
                  embedding_basis="original",
                  recycled_feature_basis="original (external is h_hat: mismatch)"),
    "A": dict(original_lm_head_used=True, rotated_lm_head_used=False,
              embedding_basis="original", recycled_feature_basis="original"),
    "B": dict(original_lm_head_used=True, rotated_lm_head_used=False,
              embedding_basis="original",
              recycled_feature_basis="original ENTERING FOLDED fc (the bug)"),
    "B2": dict(original_lm_head_used=True, rotated_lm_head_used=False,
               embedding_basis="original",
               recycled_feature_basis="original (two-path fc)"),
    "vanilla": dict(original_lm_head_used=True, rotated_lm_head_used=False,
                    embedding_basis="n/a", recycled_feature_basis="n/a"),
}


def make_adapter(variant, model, stash, device, dtype):
    if variant in ("stock", "vanilla"):
        return None
    if variant in study.ADAPTERS:
        return study.ADAPTERS[variant](model, stash, device, dtype).install()
    if variant == "D1":
        return ra.RotatedLoopAdapter(model, stash, device, dtype,
                                     score_basis="original").install()
    if variant == "D2":
        return ra.RotatedLoopAdapter(model, stash, device, dtype,
                                     score_basis="rotated").install()
    if variant.startswith("E_"):
        return ra.RotatedLoopAdapter(model, stash, device, dtype,
                                     score_basis="original",
                                     embed_mode=variant[2:]).install()
    if variant == "F_R_gamma":
        return ra.AlgebraicDraftAdapter(model, stash, device, dtype,
                                        mode="gamma").install()
    if variant == "F_R_only":
        return ra.AlgebraicDraftAdapter(model, stash, device, dtype,
                                        mode="r1").install()
    if variant == "G_native":
        return ra.GNativeAdapter(model, stash, device, dtype).install()
    raise ValueError(variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv", required=True,
                    choices=["rotated_loop_oracle",
                             "embedding_branch_ablation",
                             "algebraic_rotated_draft",
                             "rotation_aware_training"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--chat-template", default=None, choices=["llama2", "vicuna"])
    ap.add_argument("--variants", required=True)
    ap.add_argument("--rotation", default="full", choices=["none", "full"])
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--vanilla-num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-ckpt", default=None,
                    help="load a (trained) draft state dict before running "
                         "variants; used by Variant G evaluation")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one visible physical GPU per job"
    device = "cuda:0"
    ra.verify_fold_algebra()

    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")
    envp = os.path.join(run_dir, "environment.txt")
    if not os.path.isfile(envp):
        with open(envp, "w") as f:
            f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    tmpl = args.chat_template or cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    model_pair = f"{cfg['model']['target']}+{cfg['model']['eagle_draft']}"

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ORDER]
    assert not unknown, f"unknown {unknown}"
    variants = [v for v in ORDER if v in variants]
    rotated = args.rotation != "none"
    for v in variants:
        if v not in ("stock", "vanilla"):
            assert rotated, f"{v} requires the rotated target"
    # review findings: a trained ckpt must not leak into other variants, and
    # 'stock' on a rotated target would silently be 'naive' mislabeled
    if args.draft_ckpt:
        assert variants == ["G_native"], \
            "--draft-ckpt overwrites the shared draft; run G_native alone"
        assert args.seed == 0, \
            "G ckpts are trained against the seed-0 R.bin; evaluating under " \
            "another rotation seed would silently mismatch R1"
    if "stock" in variants:
        assert not rotated, "'stock' requires --rotation none"

    torch.manual_seed(args.seed)
    print(f"[rotaware] building target (rotation={args.rotation})...", flush=True)
    t0 = time.time()
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        args.rotation, "random_hadamard", "none", args.seed, device=device,
        rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    print(f"[rotaware] target built in {time.time()-t0:.0f}s", flush=True)

    if args.draft_ckpt:
        sd = torch.load(args.draft_ckpt, map_location=device, weights_only=True)
        missing = model.ea_layer.load_state_dict(sd, strict=False)
        model.ea_layer.to(torch.float16)
        print(f"[rotaware] loaded draft ckpt {args.draft_ckpt} "
              f"(missing={missing.missing_keys})", flush=True)

    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    assert prompts

    rows = []
    for variant in variants:
        adapter = make_adapter(variant, model, stash, device, torch.float16)
        timers = study.PhaseTimers(model).install()
        mode = "baseline" if variant == "vanilla" else "eagle"
        n_p = (min(args.vanilla_num_prompts, len(prompts))
               if variant == "vanilla" else len(prompts))

        wi = build_prompt(tok, prompts[0]["text"])
        study.run_one_prompt(model, timers, wi, mode, 8, tree, max_depth=5)
        if adapter:
            adapter.unrot.reset()
            if hasattr(adapter, "n_conversions"):
                adapter.n_conversions = 0

        for p in prompts[:n_p]:
            ids = build_prompt(tok, p["text"])
            r = study.run_one_prompt(model, timers, ids, mode,
                                     args.max_new_tokens, tree, max_depth=5)
            conv_ms = adapter.unrot.total_ms() if adapter and adapter.unrot.pairs else 0.0
            n_conv = (getattr(adapter, "n_conversions", None)
                      if adapter is not None else 0)
            if n_conv is None:  # A counts its unrotations as conversions
                n_conv = adapter.unrot.calls
            meta = (adapter.meta() if adapter is not None and hasattr(adapter, "meta")
                    else DEFAULT_META.get(variant, DEFAULT_META["stock"]))
            rows.append({
                "model_pair": model_pair, "variant": variant,
                "prompt_id": p["question_id"], "seed": args.seed,
                "tree_depth": 5, "quant": "none", "rotation": args.rotation,
                "acceptance_length": r["acceptance_length"],
                "generated_tokens": r["generated_tokens"],
                "target_forwards": r["target_forwards"],
                "tok_per_s": r["tokens_per_second"],
                "draft_time_ms": r["draft_time_ms_total"],
                "target_time_ms": r["target_time_ms_total"],
                "verify_time_ms": r["verify_time_ms_total"],
                "basis_conversion_time_ms": conv_ms,
                "number_of_basis_conversions": n_conv,
                "original_lm_head_used": meta["original_lm_head_used"],
                "rotated_lm_head_used": meta["rotated_lm_head_used"],
                "embedding_basis": meta["embedding_basis"],
                "recycled_feature_basis": meta["recycled_feature_basis"],
                "notes": json.dumps({"tag": args.tag, "r_bin": r_bin,
                                     "alpha_by_depth": r["alpha_by_depth"],
                                     "draft_ckpt": args.draft_ckpt or "",
                                     "runtime_mode": "fp16 (quant OFF)"}),
            })
            if adapter:
                adapter.unrot.reset()
                if hasattr(adapter, "n_conversions"):
                    adapter.n_conversions = 0
        acc = [x["acceptance_length"] for x in rows if x["variant"] == variant
               and isinstance(x["acceptance_length"], (int, float))]
        print(f"[rotaware] {variant}: mean_accept="
              f"{(sum(acc)/len(acc)) if acc else float('nan'):.3f} (n={n_p})",
              flush=True)
        timers.uninstall()
        if adapter:
            adapter.uninstall()

    tag = args.tag or f"{'-'.join(variants)}_s{args.seed}"
    shard = os.path.join(run_dir, "shards", f"{args.csv}__{tag}.csv")
    logging_utils.write_csv(shard, rows, CSV_COLUMNS)
    print(f"[rotaware] wrote {len(rows)} rows -> {shard}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
