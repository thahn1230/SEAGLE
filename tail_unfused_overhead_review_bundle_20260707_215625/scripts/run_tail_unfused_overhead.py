#!/usr/bin/env python
"""End-to-end decode + EAGLE-interface study for the tail variants.

One target build per invocation:
  --target-rotation none : unrotated target (T0 / stock / vanilla-unrotated)
  --target-rotation full : SpinQuant-fused target (T1/T2/T3/T4 via TailAdapter)

Conditions are 'TAIL:EAGLE' pairs, e.g. T1:naive T1:A T1:B2 T2:naive T3:naive
T4:naive T1:vanilla. TAIL in {T0,T1,T2,T3,T4}; EAGLE in {naive,A,B2,stock,vanilla}.
Per-prompt rows carry throughput, acceptance, phase times, tail timing
(r1t/rmsnorm/lm_head), tail-call counts, memory, and the interface flags.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_tail_unfused_overhead.py \
    --run-dir runs/tail_unfused_<ts> --target-rotation full \
    --conditions T1:naive,T1:A,T1:B2,T2:naive,T3:naive,T4:naive,T1:vanilla \
    --num-prompts 20
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
                             metrics, study, tail_unfused as tu)

TAG2MODE = {"T0": "original_fp16", "T1": "spinquant_fused",
            "T2": "explicit_unfused_correct",
            "T3": "explicit_unfused_postnorm_equiv",
            "T4": "wrong_order_gamma_after_rotation"}

CSV_COLUMNS = [
    "model_pair", "tail_tag", "tail_variant", "eagle_variant", "prompt_id",
    "seed", "quant", "acceptance_length", "generated_tokens", "target_forwards",
    "tokens_per_second", "total_ms", "target_time_ms", "draft_time_ms",
    "verify_time_ms", "tail_time_ms", "r1t_time_ms", "rmsnorm_time_ms",
    "lm_head_time_ms", "num_tail_calls", "num_r1t_calls", "mem_allocated_gib",
    "max_mem_reserved_gib", "hidden_basis_exposed", "uses_A_unrotation",
    "uses_B2_path_split", "notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chat-template", default=None, choices=["llama2", "vicuna"])
    ap.add_argument("--target-rotation", required=True, choices=["none", "full"])
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--vanilla-num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"
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

    conds = []
    for c in args.conditions.split(","):
        c = c.strip()
        if not c:
            continue
        tail_tag, eagle = c.split(":")
        conds.append((tail_tag, eagle))
    rotated = args.target_rotation == "full"
    for tail_tag, eagle in conds:
        if rotated:
            assert tail_tag in ("T1", "T2", "T3", "T4"), \
                f"{tail_tag} needs --target-rotation none"
        else:
            assert tail_tag == "T0", "unrotated build supports only T0"

    torch.manual_seed(args.seed)
    print(f"[tail-e2e] building target rotation={args.target_rotation}...", flush=True)
    t0 = time.time()
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        args.target_rotation, "random_hadamard", "none", args.seed,
        device=device, rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    print(f"[tail-e2e] built in {time.time()-t0:.0f}s", flush=True)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)

    rows = []
    for tail_tag, eagle in conds:
        mode = TAG2MODE[tail_tag]
        # install a TailAdapter for every rotated condition so tail timing is
        # captured uniformly (T1 fused included; it just times the fused tail).
        tail = tu.TailAdapter(model, stash, mode).install() if rotated else None
        # Guard (review finding): a rotated 'eagle' condition WITHOUT a draft
        # adapter would route the draft's per-tree-level head calls through the
        # tail's lm_head timer, folding draft-head cost into tail_time. The
        # supported rotated eagle variants all install a head-substituting
        # adapter; anything else must be vanilla.
        assert eagle in ("naive", "A", "B2", "vanilla") or not rotated, \
            f"rotated eagle='{eagle}' has no head-substituting adapter; tail " \
            "timing would fold in draft-head GEMMs (use naive/A/B2/vanilla)"
        draft_adapter = None
        if eagle in ("naive", "A", "B2"):
            draft_adapter = study.ADAPTERS[eagle](model, stash, device, torch.float16).install()
        timers = study.PhaseTimers(model).install()
        gen_mode = "baseline" if eagle == "vanilla" else "eagle"
        n_p = (min(args.vanilla_num_prompts, len(prompts))
               if eagle == "vanilla" else len(prompts))

        # warmup
        wi = build_prompt(tok, prompts[0]["text"])
        study.run_one_prompt(model, timers, wi, gen_mode, 8, tree, max_depth=5)
        if tail:
            tail.reset_timing()
        if draft_adapter:
            draft_adapter.unrot.reset()

        for p in prompts[:n_p]:
            ids = build_prompt(tok, p["text"])
            if tail:
                tail.reset_timing()
            r = study.run_one_prompt(model, timers, ids, gen_mode,
                                     args.max_new_tokens, tree, max_depth=5)
            tt = tail.timing() if tail else {
                "r1t_ms_total": 0.0, "r1t_calls": 0, "rmsnorm_ms_total": 0.0,
                "rmsnorm_calls": 0, "lm_head_ms_total": 0.0, "lm_head_calls": 0}
            rows.append({
                "model_pair": model_pair, "tail_tag": tail_tag,
                "tail_variant": mode, "eagle_variant": eagle,
                "prompt_id": p["question_id"], "seed": args.seed, "quant": "none",
                "acceptance_length": r["acceptance_length"],
                "generated_tokens": r["generated_tokens"],
                "target_forwards": r["target_forwards"],
                "tokens_per_second": r["tokens_per_second"],
                "total_ms": r["total_ms"],
                "target_time_ms": r["target_time_ms_total"],
                "draft_time_ms": r["draft_time_ms_total"],
                "verify_time_ms": r["verify_time_ms_total"],
                "tail_time_ms": tt["r1t_ms_total"] + tt["rmsnorm_ms_total"]
                + tt["lm_head_ms_total"],
                "r1t_time_ms": tt["r1t_ms_total"],
                "rmsnorm_time_ms": tt["rmsnorm_ms_total"],
                "lm_head_time_ms": tt["lm_head_ms_total"],
                "num_tail_calls": tt["rmsnorm_calls"] or tt["r1t_calls"],
                "num_r1t_calls": tt["r1t_calls"],
                "mem_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "max_mem_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "hidden_basis_exposed": (tail.hidden_basis_exposed if tail
                                         else "original_h_unrotated"),
                "uses_A_unrotation": eagle == "A",
                "uses_B2_path_split": eagle == "B2",
                "notes": json.dumps({"tag": args.tag, "r_bin": r_bin or "",
                                     "alpha_by_depth": r["alpha_by_depth"]}),
            })
        acc = [x["acceptance_length"] for x in rows
               if x["tail_tag"] == tail_tag and x["eagle_variant"] == eagle
               and isinstance(x["acceptance_length"], (int, float))]
        tps = [x["tokens_per_second"] for x in rows
               if x["tail_tag"] == tail_tag and x["eagle_variant"] == eagle]
        print(f"[tail-e2e] {tail_tag}:{eagle}  accept="
              f"{(sum(acc)/len(acc)) if acc else float('nan'):.3f}  "
              f"tok/s={sum(tps)/len(tps):.2f}  "
              f"r1t/prompt={sum(x['r1t_time_ms'] for x in rows if x['tail_tag']==tail_tag and x['eagle_variant']==eagle)/max(n_p,1):.2f}ms "
              f"({n_p} prompts)", flush=True)
        timers.uninstall()
        if draft_adapter:
            draft_adapter.uninstall()
        if tail:
            tail.uninstall()

    tag = args.tag or f"{args.target_rotation}_s{args.seed}"
    shard = os.path.join(run_dir, "shards", f"tail__{tag}.csv")
    logging_utils.write_csv(shard, rows, CSV_COLUMNS)
    print(f"[tail-e2e] wrote {len(rows)} rows -> {shard}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
