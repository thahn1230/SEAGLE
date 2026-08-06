#!/usr/bin/env python
"""PMG Phase 8: runtime overhead + explicit-op audit for one arm.

Measures on real mtbench decode (greedy, official tree):
  - per-cycle draft (topK_genrate) and verify (base forward) wall ms
    via CUDA-event sync wrappers (warmup then measured prompts)
  - PostProjectionR1 ms/call (CUDA events around its forward)
  - recurrent e-slice rescale count (EP3-P arms)
  - torch.profiler pass over a few cycles -> top CUDA ops filtered to
    rotation/scale/hadamard/quant signatures -> profiler summary
Writes <run>/tables/runtime__<tag>.json (+ profiler txt).

Usage mirrors eval_eagle_acceptance_length deploy flags.
"""
import argparse, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import eagle_bridge, experiment, study

KIND = "learned_chat_w4a4kv16"


class EvAccum:
    def __init__(self):
        self.ms, self.n = 0.0, 0

    def wrap(self, fn):
        def inner(*a, **k):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            out = fn(*a, **k)
            e.record(); e.synchronize()
            self.ms += s.elapsed_time(e); self.n += 1
            return out
        return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["fp16", "int4", "w8a8"])
    ap.add_argument("--draft-cfg", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alpha-rec", type=float, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--quant-first", default=None)
    ap.add_argument("--quant-recurrent", default=None)
    ap.add_argument("--quant-ar", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    # reuse the official evaluator's model/adapter assembly by import
    sys.argv = (["eval"] + [a for pair in [
        ("--target", args.target), ("--draft-cfg", args.draft_cfg),
        ("--tag", "RTB_" + args.tag), ("--run-dir", args.run_dir),
        ("--datasets", "mtbench"), ("--pool", "eval"),
        ("--n-prompts", "1")] for a in pair])
    # manual assembly instead (evaluator has no library entry point)
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16" else
                  ("full", "w8a8") if args.target == "w8a8" else
                  ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], rot, KIND, quant, 0, device=dev,
        rotations_root=rr)
    if rot == "none":
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                        rr), map_location="cpu",
                       weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight \
            .detach().float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight \
            .detach().float().cpu().clone()
    ea = model.ea_layer
    if args.draft_sd:
        sd_new = torch.load(args.draft_sd, map_location="cpu",
                            weights_only=False)
        sd_new = sd_new.get("draft_state_dict",
                            sd_new.get("model", sd_new))
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                            for k, v in sd_new.items()}, strict=True)
        ea.to(dev)
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveDraftAdapter)
    fhm = "identity" if args.target == "fp16" else "gamma_R1"
    D4 = dict(quant_first=args.quant_first or "fake_w4a4",
              quant_recurrent=args.quant_recurrent or "fake_w4a4",
              quant_ar=args.quant_ar or "fake_w4a4", ar_r2r4=True)
    ad = None
    if args.draft_cfg == "stock":
        if args.target != "fp16":
            from eagle_spinquant.causal_interface import (
                RestoredInterfaceCSAdapter)
            ad = RestoredInterfaceCSAdapter(
                model, stash, dev, torch.float16, variant="folded",
                first_hidden_mode="identity", trace=False)
    elif args.draft_cfg == "rot_ep3p":
        ck = torch.load(args.ckpt, map_location="cpu",
                        weights_only=False)
        st = dict(stash)
        R_T = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
        kw = dict(embed_scale_alpha=args.alpha, **D4)
        if args.alpha_rec is not None:
            kw["embed_scale_alpha_rec"] = args.alpha_rec
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, first_fold_R=R_T,
            **kw)
    else:
        kw = dict(embed_scale_alpha=args.alpha, **D4)
        if args.alpha_rec is not None:
            kw["embed_scale_alpha_rec"] = args.alpha_rec
        ad = ConcatSelectiveDraftAdapter(
            model, st if False else stash, dev, torch.float16,
            variant="folded", first_hidden_mode=fhm, trace=False,
            **kw)
    if ad is not None:
        ad.install()

    # instrument
    post = EvAccum(); draft = EvAccum(); verify = EvAccum()
    if ad is not None and hasattr(ad, "split") and \
            hasattr(ad.split, "post_projection_R1"):
        pp = ad.split.post_projection_R1
        pp.forward = post.wrap(pp.forward)
    ea.topK_genrate = draft.wrap(ea.topK_genrate)
    model.base_model.model.forward = verify.wrap(
        model.base_model.model.forward)

    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    tok = eagle_bridge.get_tokenizer(model)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    from eagle_spinquant.eval_datasets import load_eval_prompts
    prompts, _ = load_eval_prompts("mtbench",
                                   args.n_warmup + args.n_prompts,
                                   "eval")

    stats = dict(tokens=0, cycles=0, wall=0.0)
    for i, p in enumerate(prompts):
        if i == args.n_warmup:                     # reset after warmup
            post.ms = post.n = draft.n = verify.n = 0
            draft.ms = verify.ms = 0.0
            stats = dict(tokens=0, cycles=0, wall=0.0)
        ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
        L0 = ids.shape[1]; prev = L0
        torch.cuda.synchronize(); t0 = time.time()
        for out in model.ea_generate(ids, temperature=0.0,
                                     max_steps=args.max_new_tokens + 8,
                                     tree_choices=tree):
            cur = out.shape[1]
            if cur > prev:
                stats["cycles"] += 1; prev = cur
            if cur - L0 >= args.max_new_tokens:
                break
        torch.cuda.synchronize()
        stats["wall"] += time.time() - t0
        stats["tokens"] += prev - L0

    # short profiler pass (1 prompt) for explicit-op audit
    from torch.profiler import profile, ProfilerActivity
    ids = build_prompt(tok, prompts[0]["text"])[:, :1024].to(dev)
    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        n = 0
        for out in model.ea_generate(ids, temperature=0.0,
                                     max_steps=12, tree_choices=tree):
            n += 1
            if n >= 6:
                break
    rows = prof.key_averages()
    sig = ("gemm", "hadamard", "mul", "quant", "copy_", "cat")
    def _ct(r):
        return getattr(r, "cuda_time_total",
                       getattr(r, "device_time_total", 0))
    top = sorted([r for r in rows
                  if any(s in r.key.lower() for s in sig)],
                 key=lambda r: -_ct(r))[:25]
    with open(os.path.join(args.run_dir, "tables",
                           f"profiler__{args.tag}.txt"), "w") as f:
        for r in top:
            f.write(f"{r.key:60s} cuda_ms={_ct(r)/1e3:.2f} "
                    f"calls={r.count}\n")

    cyc = max(stats["cycles"], 1)
    res = dict(tag=args.tag, target=args.target,
               draft_cfg=args.draft_cfg,
               n_prompts=args.n_prompts,
               ms_per_token=round(1000 * stats["wall"]
                                  / max(stats["tokens"], 1), 3),
               tokens_per_s=round(stats["tokens"]
                                  / max(stats["wall"], 1e-9), 2),
               draft_ms_per_cycle=round(draft.ms / cyc, 3),
               verify_ms_per_cycle=round(verify.ms / max(verify.n, 1)
                                         * (verify.n / cyc), 3),
               postproj_ms_per_call=round(post.ms / max(post.n, 1), 4),
               postproj_calls_per_cycle=round(post.n / cyc, 2),
               postproj_ms_per_cycle=round(post.ms / cyc, 4),
               cycles=stats["cycles"], tokens=stats["tokens"])
    path = os.path.join(args.run_dir, "tables",
                        f"runtime__{args.tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[rtb] {args.tag}: {res['ms_per_token']} ms/tok, draft "
          f"{res['draft_ms_per_cycle']} verify "
          f"{res['verify_ms_per_cycle']} postR1 "
          f"{res['postproj_ms_per_cycle']} ms/cyc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
