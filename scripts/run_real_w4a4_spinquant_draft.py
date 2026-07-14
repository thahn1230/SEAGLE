#!/usr/bin/env python
"""Phase 9: REAL W4A4 (QuaRot CUTLASS INT4) target + pure-R1 SpinQuant draft.

Builds a rotated fp16 target, swaps its per-layer linears to REAL packed-INT4
QuaRot kernels, installs the unfused tail (exposes h) + the pure-R1 draft
(runtime h@R1 external-only, head W_lm@R1), proves INT4 kernel dispatch via the
torch profiler, and runs B3 (pure-R1) vs B2 (prev-failed control) n=2 greedy:
AR vs EAGLE tokens + acceptance.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_real_w4a4_spinquant_draft.py \
      --run-dir runs/spinquant_draft_pure_r1_<ts> --num-prompts 2 --max-new-tokens 16
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             realint4, spinquant_draft as spd, study,
                             w4a4_impl_fix as wf)

DEV = "cuda:0"


def profile_kernels(model, ids, tree, run_dir):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in model.ea_generate(ids, temperature=0.0, max_steps=8,
                                   tree_choices=tree):
            pass
    evs = sorted([(ev.key, ev.self_device_time_total) for ev in p.key_averages()
                  if ev.self_device_time_total > 0], key=lambda kv: -kv[1])
    names = [k for k, _ in evs]
    proven = any(("int4" in n.lower() or "s4" in n.lower() or "sym_quant" in n.lower()
                  or ("cutlass" in n.lower() and "gemm" in n.lower())) for n in names)
    with open(os.path.join(run_dir, "kernel_dispatch_trace.txt"), "w") as f:
        f.write(f"# real W4A4 (QuaRot) EAGLE kernel dispatch — int4_proven={proven}\n")
        for k, us in evs[:40]:
            f.write(f"{k} | {us:.0f} us\n")
    with open(os.path.join(run_dir, "profiler_kernel_names.txt"), "w") as f:
        f.write("\n".join(names[:40]) + "\n")
    return proven, names[:6]


@torch.no_grad()
def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        final = out
        if cur - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist(), deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]

    print("[real] loading + rotating (r1r2, fp16) target...", flush=True)
    # r1r2 (not 'full'): leaves plain nn.Linear (no ActQuantWrapper), so the
    # QuaRot swap can replace them; QuaRot's swap folds R4 into down_proj itself.
    # Stash still carries R1 + gamma_f for the unfused tail.
    model = eagle_bridge.load_eagle_model(paths["target_path"], paths["draft_path"],
                                          dtype=torch.float16, device_map=DEV)
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"], rr)
    stash = study.apply_rotation_quant(model.base_model, "r1r2", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    model.base_model.to(DEV); model.ea_layer.to(DEV)
    print("[real] swapping target linears -> QuaRot INT4...", flush=True)
    info = realint4.swap_target_linears(model.base_model, "quarot_w4a4", DEV)
    print(f"[real] {info}", flush=True)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, DEV)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)

    ids0 = build_prompt(tok, prompts[0]["text"]).to(DEV)
    proven, top = profile_kernels(model, ids0, tree, run_dir)
    print(f"[real] int4_dispatch_proven={proven} top={top[:3]}", flush=True)

    rows = []
    for name in ["B2", "B3"]:
        tail = wf.UnfusedTailAdapter(model, stash).install()
        if name == "B2":
            draft = wf.DraftProjTransformAdapter(model, stash, DEV, torch.float16,
                     first_embed="R1T", first_hidden="I", rec_embed="R1T",
                     rec_hidden="R1T", head_mode="fused").install()
        else:
            draft = spd.SpinquantDraftPureR1Adapter(model, stash, DEV, torch.float16).install()
        timers = study.PhaseTimers(model).install()
        for p in prompts:
            ids = build_prompt(tok, p["text"]).to(DEV)
            if hasattr(draft, "set_context"):
                draft.set_context(p["question_id"])
            ilen = ids.shape[1]
            ar, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                     max_steps=args.max_new_tokens + 4, tree_choices=tree), ilen, args.max_new_tokens)
            eg, deltas = run_gen(model.ea_generate(ids, temperature=0.0,
                     max_steps=args.max_new_tokens + 8, tree_choices=tree), ilen, args.max_new_tokens)
            n = min(len(ar), len(eg))
            rows.append(dict(model_pair=cfg["model"]["target"], quant_mode="real_w4a4_quarot",
                baseline_name=name, prompt_id=p["question_id"],
                acceptance_length_mean=(sum(deltas)/len(deltas)) if deltas else 0.0,
                acceptance_length_list=str(deltas),
                exact_token_match=bool(ar[:n] == eg[:n] and len(ar) == len(eg)),
                verifier_logit_rel_l2_mean=0.0, tokens_per_second="",
                int4_dispatch_proven=proven, notes=name))
        acc = [r["acceptance_length_mean"] for r in rows if r["baseline_name"] == name]
        mm = [r["exact_token_match"] for r in rows if r["baseline_name"] == name]
        print(f"[real] {name} accept={sum(acc)/len(acc):.3f} match={sum(mm)/len(mm):.2f}", flush=True)
        timers.uninstall(); draft.uninstall(); tail.uninstall()
    logging_utils.write_csv(os.path.join(run_dir, "shards", "real_w4a4_B2-B3.csv"), rows)
    print(f"[real] int4_proven={proven}; wrote {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
