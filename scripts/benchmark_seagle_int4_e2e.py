#!/usr/bin/env python
"""SEAGLE real-INT4 E2E latency: why the DRAFT must be quantized too.

Four configs on the same EAGLE tree-decoding loop (greedy, mtbench):
  T16D16  fp16 target + fp16 draft            (baseline)
  T4D16   INT4 target + fp16 draft            (target-only quant)
  T4D4    INT4 target + INT4 draft            (full)
  T16D4   fp16 target + INT4 draft            (control)

INT4 = real s4s4s32 Tensor-Core linears (RealInt4Linear, QuaRot/CUTLASS
backend, IMMA-verified) on every decoder linear (q/k/v/o/gate/up/down)
and, for the draft, also the 8192->4096 fusion projection. Embedding,
LM head, norms, RoPE, softmax, residual stay fp16 (study contract).

Measured per config: wall ms/output-token, cycles/s, per-phase split
(draft-side vs target-verify wall time via CUDA-synchronized wraps),
measured tau. Also reports PROJECTED tokens/s = cycles/s x AL_validated
using the fake-quant study taus (quality-calibrated EP3-P numbers),
separating kernel speed from this run's RTN-only acceptance.
Writes tables/seagle_e2e.json.
"""
import argparse, json, os, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.eval_datasets import load_eval_prompts
from eagle_spinquant.real_int4_linear import (RealInt4Linear,
                                              swap_llama_linears_int4)

KIND = "learned_chat_w4a4kv16"


class PhaseTimer:
    def __init__(self):
        self.t = {"draft": 0.0, "verify": 0.0}

    def wrap(self, obj, attr, phase):
        fn = getattr(obj, attr)
        tm = self

        def wrapped(*a, **k):
            torch.cuda.synchronize()
            t0 = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            tm.t[phase] += time.time() - t0
            return out
        setattr(obj, attr, wrapped)
        return fn


@torch.no_grad()
def run_config(name, quant_target, quant_draft, prompts, mx=128):
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "none", KIND, "none", 0,
        device="cuda:0", rotations_root=rr)
    n_t = n_d = 0
    if quant_target:
        n_t = swap_llama_linears_int4(model.base_model)
    if quant_draft:
        n_d = swap_llama_linears_int4(model.ea_layer)
        if isinstance(model.ea_layer.fc, torch.nn.Linear):
            model.ea_layer.fc = RealInt4Linear(model.ea_layer.fc,
                                               name="draft.fc")
            n_d += 1
    torch.cuda.empty_cache()
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, "cuda:0")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    pt = PhaseTimer()
    pt.wrap(model.ea_layer, "topK_genrate", "draft")
    pt.wrap(model.base_model.model, "forward", "verify")
    stats = dict(tokens=0, cycles=0, wall=0.0)
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :512].to("cuda:0")
        L0 = ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.time()
        prev = L0
        for out in model.ea_generate(ids, temperature=0.0,
                                     max_steps=mx + 8,
                                     tree_choices=tree):
            cur = out.shape[1]
            if cur > prev:
                stats["cycles"] += 1
                prev = cur
            if cur - L0 >= mx:
                break
        torch.cuda.synchronize()
        stats["wall"] += time.time() - t0
        stats["tokens"] += prev - L0
    res = dict(config=name, n_int4_target=n_t, n_int4_draft=n_d,
               tokens=stats["tokens"], cycles=stats["cycles"],
               wall_s=round(stats["wall"], 2),
               ms_per_token=round(1000 * stats["wall"]
                                  / max(stats["tokens"], 1), 2),
               cycles_per_s=round(stats["cycles"]
                                  / max(stats["wall"], 1e-9), 2),
               tau_measured=round(stats["tokens"]
                                  / max(stats["cycles"], 1), 3),
               draft_s=round(pt.t["draft"], 2),
               verify_s=round(pt.t["verify"], 2),
               draft_ms_per_cycle=round(1000 * pt.t["draft"]
                                        / max(stats["cycles"], 1), 2),
               verify_ms_per_cycle=round(1000 * pt.t["verify"]
                                         / max(stats["cycles"], 1),
                                         2),
               other_ms_per_cycle=round(
                   1000 * (stats["wall"] - pt.t["draft"]
                           - pt.t["verify"])
                   / max(stats["cycles"], 1), 2))
    del model
    torch.cuda.empty_cache()
    print(f"[e2e] {name}: {res['ms_per_token']} ms/tok, tau "
          f"{res['tau_measured']}, draft {res['draft_ms_per_cycle']}"
          f" verify {res['verify_ms_per_cycle']} ms/cycle",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--configs", default="T16D16,T4D16,T4D4,T16D4")
    args = ap.parse_args()
    rd = args.run_dir
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")
    grid = dict(T16D16=(False, False), T4D16=(True, False),
                T4D4=(True, True), T16D4=(False, True))
    out = []
    for name in args.configs.split(","):
        qt, qd = grid[name]
        out.append(run_config(name, qt, qd, prompts,
                              args.max_new_tokens))
        path = os.path.join(rd, "tables", "seagle_e2e.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # projected tokens/s with QUALITY-validated taus (fake-quant
        # studies: T4 target EP3-P draft tau 3.05, fp16 draft on T4
        # 3.27, fp16/fp16 3.58) — separates kernel speed from this
        # run's RTN-only acceptance
        VALIDATED_TAU = dict(T16D16=3.5762, T4D16=3.2744,
                             T4D4=3.0517, T16D4=2.9146)
        for r in out:
            r["tau_validated_ref"] = VALIDATED_TAU[r["config"]]
            r["projected_tok_per_s"] = round(
                r["cycles_per_s"] * r["tau_validated_ref"], 2)
        old = (json.load(open(path))
               if os.path.exists(path) else [])
        merged = {r["config"]: r for r in old}
        for r in out:
            merged[r["config"]] = r
        json.dump(list(merged.values()), open(path, "w"),
                  indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
