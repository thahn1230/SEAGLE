#!/usr/bin/env python
"""Universal EAGLE Average Acceptance Length (tau) evaluator for the
PTQ-vs-QAT study. Greedy, official tree (mc_sim_7b_63), official micro
aggregation: tau = sum(accepted+1)/cycles pooled over cycles.

Targets
  fp16 : stock unquantized target
  int4 : learned-R_T SpinQuant W4A4, KV16 (primary INT4 contract)

Draft configs (--draft-cfg)
  stock          : original draft, no adapter (fp16 target) or RESTORED
                   interface (int4 target: inverse rotation + final gamma
                   exactly once, fp16 draft)
  d4p3           : strict training-free D4P3 PTQ (proj+AR W4A4, P3 alpha,
                   embed/head fp16, KV16). Interface auto: identity under
                   fp16 target, gamma_R1 (shared R_T) under int4 target.
  naive_w4a4     : D4 without P3 (no alpha migration)
  p2             : P2 separate activation scales (branchwise), no P3
  rot            : D4P3 + local residual rotation ckpt (--ckpt, R_D+alpha)
  fp16_deploy    : FP16 draft deployment of a retrained state dict
                   (--draft-sd), stock/restored interface per target
  d4p3_deploy    : D4P3 deployment of a retrained state dict (--draft-sd):
                   load sd into ea_layer, then fold+quantize (QAT deploy,
                   or FP16-retrain-then-PTQ)

Records per prompt: acceptance_list, first_rejection_depths(=tau per cycle),
n_cycles. Writes shards/al__<tag>__<target>__<ds>.csv.
No training. Weights only ever LOADED, never updated here.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"
ID_ALPHA, ROT_ALPHA = 45.254834, 32.0


def run_gen(gen, ilen, mx):
    deltas, prev = [], ilen
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= mx:
            break
    return deltas


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["fp16", "int4"])
    ap.add_argument("--draft-cfg", required=True,
                    choices=["stock", "d4p3", "naive_w4a4", "p2", "rot",
                             "fp16_deploy", "d4p3_deploy"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=None, help="rotation ckpt (rot)")
    ap.add_argument("--draft-sd", default=None,
                    help="retrained draft state dict (.pt)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--datasets", default="mtbench")
    ap.add_argument("--pool", default="eval", choices=["eval", "calib"],
                    help="calib = disjoint offset-500 pool (alpha sweeps)")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    if args.target == "fp16":
        rot, quant = "none", "none"
    else:
        rot, quant = "full", "w4a4"
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    if rot == "none":
        # unrotated target: populate stash from R.bin so adapters can fold
        # (identity interface itself stays unrotated; validated pattern from
        # calibrate_embedding_alpha.py)
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight.detach() \
            .float().cpu().clone()
    ea = model.ea_layer
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]

    # optionally load a retrained draft state dict FIRST (adapters fold
    # from the live ea_layer state)
    if args.draft_sd:
        sd_new = torch.load(args.draft_sd, map_location="cpu",
                            weights_only=False)
        if "draft_state_dict" in sd_new:
            sd_new = sd_new["draft_state_dict"]
        ea.load_state_dict(
            {k: v.to(ea.fc.weight.dtype) for k, v in sd_new.items()},
            strict=True)
        ea.to(dev)

    fhm = "identity" if args.target == "fp16" else "gamma_R1"
    def_alpha = ID_ALPHA if args.target == "fp16" else ROT_ALPHA
    alpha = args.alpha if args.alpha is not None else def_alpha
    D4 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True)

    ad = None
    if args.draft_cfg in ("stock", "fp16_deploy"):
        if args.target == "int4":
            from eagle_spinquant.causal_interface import (
                RestoredInterfaceCSAdapter)
            ad = RestoredInterfaceCSAdapter(
                model, stash, dev, torch.float16, variant="folded",
                first_hidden_mode="identity", trace=False)
        # fp16 target: no adapter at all (pure stock path)
    elif args.draft_cfg == "d4p3":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False,
            embed_scale_alpha=alpha, **D4)
    elif args.draft_cfg == "naive_w4a4":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, **D4)
    elif args.draft_cfg == "p2":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, branch_act=(4, 4), **D4)
    elif args.draft_cfg == "rot":
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        st = dict(stash)
        R_T = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, first_fold_R=R_T,
            embed_scale_alpha=(args.alpha if args.alpha is not None
                               else float(ck.get("alpha", def_alpha))),
            **D4)
    elif args.draft_cfg == "d4p3_deploy":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False,
            embed_scale_alpha=alpha, **D4)
    if ad is not None:
        ad.install()

    for ds_spec in args.datasets.split(","):
        ds_name, _, ds_n = ds_spec.partition(":")
        n_prompts = int(ds_n) if ds_n else args.n_prompts
        sfx = "" if args.pool == "eval" else f"__{args.pool}"
        out_csv = os.path.join(
            args.run_dir, "shards",
            f"al__{args.tag}__{args.target}__{ds_name}{sfx}.csv")
        if os.path.exists(out_csv):
            print(f"[al] {args.tag}/{ds_name}: exists, skip", flush=True)
            continue
        prompts, _ = load_eval_prompts(ds_name, n_prompts, args.pool)
        rows = []
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            if ad is not None and hasattr(ad, "set_context"):
                ad.set_context(p["row_id"])
            deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            rows.append(dict(tag=args.tag, target=args.target,
                             dataset=ds_name, prompt_id=p["row_id"],
                             acceptance_list=json.dumps(deltas),
                             n_cycles=len(deltas)))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        logging_utils.write_csv(out_csv, rows)
        taus = [t for r in rows for t in json.loads(r["acceptance_list"])]
        print(f"[al] {args.tag}/{args.target}/{ds_name}: "
              f"tau={sum(taus)/max(len(taus),1):.4f} "
              f"({len(taus)} cycles)", flush=True)
    if ad is not None:
        ad.uninstall()
    print(f"[al] {args.tag} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
