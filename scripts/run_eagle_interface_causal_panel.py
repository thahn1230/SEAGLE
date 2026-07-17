#!/usr/bin/env python
"""Phases C+D: interface causal panel (C0-C4, D8 + D16 controls) on a
pinned dataset. micro-AL evaluated from per-cycle records; Gate E enforced
by construction (each config differs only in the intended factor).

Configs (target build, adapter, draft bits):
  C0 T16_IDENTITY_D8   stock fp16      CS(identity)          D8
  C1 TR16_RESTORED_D8  rot-fp16        RestoredCS(identity)  D8
  C2 TR16_ROTATED_D8   rot-fp16        CS(gamma_R1)          D8
  C3 T8_RESTORED_D8    rot-w8a8        RestoredCS(identity)  D8
  C4 T8_ROTATED_D8     rot-w8a8        CS(gamma_R1)          D8
  + D16 controls: T16_IDENTITY_D16 (no adapter), TR16_RESTORED_D16
    (UnrotateAdapter), TR16_ROTATED_D16 (CS gamma_R1 fp16),
    T8_RESTORED_D16, T8_ROTATED_D16.
"""
import argparse, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.causal_interface import RestoredInterfaceCSAdapter
from eagle_spinquant.study import UnrotateAdapter
from eagle_spinquant.eval_datasets import (load_eval_prompts,
                                           write_or_verify_manifest)

KIND = "learned_chat_w4a4kv16"
D8 = dict(quant_first="fake_w8a8", quant_recurrent="fake_w8a8",
          quant_ar="fake_w8a8", ar_r2r4=True)

# name -> (build_key, adapter_kind, draft_kwargs)
PANEL = {
    "T16_IDENTITY_D8": ("stock", "cs_identity", D8),
    "TR16_RESTORED_D8": ("rot_fp16", "restored", D8),
    "TR16_ROTATED_D8": ("rot_fp16", "cs_gamma", D8),
    "T8_RESTORED_D8": ("rot_w8a8", "restored", D8),
    "T8_ROTATED_D8": ("rot_w8a8", "cs_gamma", D8),
    "T16_IDENTITY_D16": ("stock", "none", None),
    "TR16_RESTORED_D16": ("rot_fp16", "unrotate", None),
    "TR16_ROTATED_D16": ("rot_fp16", "cs_gamma", {}),
    "T8_RESTORED_D16": ("rot_w8a8", "unrotate", None),
    "T8_ROTATED_D16": ("rot_w8a8", "cs_gamma", {}),
}
BUILDS = {"stock": ("none", "none"), "rot_fp16": ("full", "none"),
          "rot_w8a8": ("full", "w8a8")}


def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        final = out
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist(), deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="mtbench",
                    choices=["mtbench", "sharegpt", "c4", "gsm8k",
                             "humaneval"])
    ap.add_argument("--configs", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    names = list(PANEL)
    if args.configs:
        names = [n for n in names if n in set(args.configs.split(","))]
    rd = args.run_dir
    os.makedirs(os.path.join(rd, "shards"), exist_ok=True)
    with open(os.path.join(rd, "commands.sh"), "a") as f:
        f.write(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="
                f"{os.environ['CUDA_VISIBLE_DEVICES']} "
                + " ".join(sys.argv) + "\n")

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, manifest = load_eval_prompts(args.dataset, args.num_prompts,
                                          "eval")
    msha = write_or_verify_manifest(rd, args.dataset, "eval", manifest)
    print(f"[panel] dataset={args.dataset} manifest={msha[:12]}", flush=True)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    # group configs by build to reuse targets
    by_build = {}
    for n in names:
        by_build.setdefault(PANEL[n][0], []).append(n)
    for bkey, group in by_build.items():
        rot, quant = BUILDS[bkey]
        pending = [n for n in group if not os.path.exists(os.path.join(
            rd, "shards", f"panel__{args.dataset}__{n}.csv"))]
        if not pending:
            print(f"[panel] build {bkey}: all done, skip", flush=True)
            continue
        print(f"[panel] building {bkey} ...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, KIND, quant, 0, device=dev, rotations_root=rr)
        if rot == "none":
            R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                            rr),
                           map_location="cpu", weights_only=False)
            stash["R1"] = R["R1"].clone()
            stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
                .float().cpu().clone()
            stash["lm_head_weight"] = model.base_model.lm_head.weight \
                .detach().float().cpu().clone()
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        ids_list = [build_prompt(tok, p["text"])[:, :1024].to(dev)
                    for p in prompts]

        for n in pending:
            _, akind, dkw = PANEL[n]
            if akind == "none":
                ad = None
            elif akind == "unrotate":
                ad = UnrotateAdapter(model, stash, dev, torch.float16,
                                     with_gamma=True)
            elif akind == "cs_identity":
                ad = ConcatSelectiveDraftAdapter(
                    model, stash, dev, torch.float16, variant="folded",
                    first_hidden_mode="identity", trace=False, **(dkw or {}))
            elif akind == "cs_gamma":
                ad = ConcatSelectiveDraftAdapter(
                    model, stash, dev, torch.float16, variant="folded",
                    first_hidden_mode="gamma_R1", trace=False, **(dkw or {}))
            elif akind == "restored":
                ad = RestoredInterfaceCSAdapter(
                    model, stash, dev, torch.float16, variant="folded",
                    trace=False, **(dkw or {}))
            if ad is not None:
                ad.install()
            rows = []
            t0 = time.time()
            for pi, ids in enumerate(ids_list):
                if ad is not None and hasattr(ad, "set_context"):
                    ad.set_context(prompts[pi]["row_id"])
                eg, deltas = run_gen(model.ea_generate(
                    ids, temperature=0.0,
                    max_steps=args.max_new_tokens + 8,
                    tree_choices=tree), ids.shape[1], args.max_new_tokens)
                rows.append(dict(config=n, dataset=args.dataset,
                                 prompt_id=prompts[pi]["row_id"],
                                 acceptance_list=json.dumps(deltas),
                                 n_new_tokens=len(eg),
                                 n_cycles=len(deltas)))
            if ad is not None:
                ad.uninstall()
            taus = [t for r in rows for t in json.loads(r["acceptance_list"])]
            mal = sum(taus) / max(len(taus), 1)
            logging_utils.write_csv(os.path.join(
                rd, "shards", f"panel__{args.dataset}__{n}.csv"), rows)
            print(f"[panel] {args.dataset}/{n}: micro-AL={mal:.4f} "
                  f"cycles={len(taus)} ({time.time()-t0:.0f}s)", flush=True)
        del ad, model, stash, ids_list, tok
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[panel] dataset {args.dataset} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
