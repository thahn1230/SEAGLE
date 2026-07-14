#!/usr/bin/env python
"""Task 4: acceptance/throughput comparison of the current SpinQuant tail (T1,
emits h_hat) vs the EAGLE-friendly tail (T5, emits h_R), and whether the B2
path split becomes unnecessary.

Conditions (one rotated target build):
  T1:naive       h_hat tail + frozen original draft (naive)   -> collapse
  T1:A           h_hat tail + Variant A (runtime unrotation)   -> recover (2 concepts)
  T1:B2          h_hat tail + Variant B2 (two-path fold)       -> recover, fc_paths=2
  T5:naive_frozen h_R tail + frozen original draft (naive)     -> collapse (control:
                                                                  h_R != original h)
  T5:pure_r1     h_R tail + pure-R1 draft (single fc path)     -> recover, fc_paths=1, NO B2

Per-prompt rows -> runs/<run>/shards/pure_r1__<tag>.csv.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_pure_r1_eagle.py \
      --run-dir runs/pure_r1_<ts> --num-prompts 20
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
                             pure_r1_eagle as pr, study, tail_unfused as tu)

CSV_COLUMNS = [
    "model_pair", "condition", "tail_tag", "tail_variant", "eagle_variant",
    "prompt_id", "seed", "acceptance_length", "generated_tokens",
    "target_forwards", "tokens_per_second", "total_ms", "target_time_ms",
    "draft_time_ms", "verify_time_ms", "tail_r1t_ms", "hidden_basis_exposed",
    "fc_paths", "uses_A_unrotation", "uses_B2_path_split", "notes",
]

# condition -> (tail_mode, eagle_kind)
CONDS = {
    "T1:naive":        ("spinquant_fused",   "naive"),
    "T1:A":            ("spinquant_fused",   "A"),
    "T1:B2":           ("spinquant_fused",   "B2"),
    "T5:naive_frozen": ("eagle_friendly_hR", "naive"),
    "T5:pure_r1":      ("eagle_friendly_hR", "pure_r1"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chat-template", default=None, choices=["llama2", "vicuna"])
    ap.add_argument("--conditions",
                    default="T1:naive,T1:A,T1:B2,T5:naive_frozen,T5:pure_r1")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    device = "cuda:0"
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
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
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    torch.manual_seed(args.seed)
    print("[pure-r1] building rotated target...", flush=True)
    t0 = time.time()
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "none", args.seed, device=device,
        rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    print(f"[pure-r1] built in {time.time()-t0:.0f}s", flush=True)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)

    rows = []
    for cond in conds:
        tail_mode, kind = CONDS[cond]
        tail = tu.TailAdapter(model, stash, tail_mode).install()
        draft_adapter = None
        if kind in ("naive", "A", "B2"):
            draft_adapter = study.ADAPTERS[kind](model, stash, device, torch.float16).install()
        elif kind == "pure_r1":
            draft_adapter = pr.PureR1Adapter(model, stash, device, torch.float16).install()
        timers = study.PhaseTimers(model).install()
        fc_paths = 2 if kind == "B2" else 1

        wi = build_prompt(tok, prompts[0]["text"])
        study.run_one_prompt(model, timers, wi, "eagle", 8, tree, max_depth=5)
        tail.reset_timing()
        if draft_adapter and hasattr(draft_adapter, "unrot"):
            draft_adapter.unrot.reset()

        for p in prompts:
            ids = build_prompt(tok, p["text"])
            tail.reset_timing()
            r = study.run_one_prompt(model, timers, ids, "eagle",
                                     args.max_new_tokens, tree, max_depth=5)
            tt = tail.timing()
            rows.append({
                "model_pair": model_pair, "condition": cond, "tail_tag":
                    tu.MODE_TAG[tail_mode], "tail_variant": tail_mode,
                "eagle_variant": kind, "prompt_id": p["question_id"],
                "seed": args.seed, "acceptance_length": r["acceptance_length"],
                "generated_tokens": r["generated_tokens"],
                "target_forwards": r["target_forwards"],
                "tokens_per_second": r["tokens_per_second"],
                "total_ms": r["total_ms"],
                "target_time_ms": r["target_time_ms_total"],
                "draft_time_ms": r["draft_time_ms_total"],
                "verify_time_ms": r["verify_time_ms_total"],
                "tail_r1t_ms": tt["r1t_ms_total"],
                "hidden_basis_exposed": tail.hidden_basis_exposed,
                "fc_paths": fc_paths, "uses_A_unrotation": kind == "A",
                "uses_B2_path_split": kind == "B2",
                "notes": json.dumps({"tag": args.tag, "r_bin": r_bin or "",
                                     "alpha_by_depth": r["alpha_by_depth"]}),
            })
            tail.reset_timing()
            if draft_adapter and hasattr(draft_adapter, "unrot"):
                draft_adapter.unrot.reset()
        acc = [x["acceptance_length"] for x in rows if x["condition"] == cond
               and isinstance(x["acceptance_length"], (int, float))]
        tps = [x["tokens_per_second"] for x in rows if x["condition"] == cond]
        print(f"[pure-r1] {cond:16s} accept="
              f"{(sum(acc)/len(acc)) if acc else float('nan'):.3f}  "
              f"tok/s={sum(tps)/len(tps):.2f}  fc_paths={fc_paths}  "
              f"basis={tail.hidden_basis_exposed}", flush=True)
        timers.uninstall()
        if draft_adapter:
            draft_adapter.uninstall()
        tail.uninstall()

    tag = args.tag or f"s{args.seed}"
    shard = os.path.join(run_dir, "shards", f"pure_r1__{tag}.csv")
    logging_utils.write_csv(shard, rows, CSV_COLUMNS)
    print(f"[pure-r1] wrote {len(rows)} rows -> {shard}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
