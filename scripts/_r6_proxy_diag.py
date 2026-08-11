#!/usr/bin/env python
"""R6 quantization-proxy diagnostics (secondary): v/o W4 NMSE +
pre-quant activation stats, R5+baseline-R2 vs R5+learned-R6, both under
GS and the identical quantizer. Never used for selection.

Usage: _r6_proxy_diag.py --run-dir <rd> --r5-ckpt <R5> --r6-ckpt <ck>
Writes <run>/geometry/r6_proxy_diag.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts
from eagle_spinquant import eagle_bridge
from eval_eagle_acceptance_length import run_gen  # noqa: E402
from _gsr5_proxy_diag import ActStat, wstats  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_G = D ** 0.42
D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--r5-ckpt", required=True)
    ap.add_argument("--r6-ckpt", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    ea = model.ea_layer
    R_T = stash["R1"].clone()
    R5 = torch.load(args.r5_ckpt, map_location="cpu",
                    weights_only=False)["R_D"].double()
    R6 = torch.load(args.r6_ckpt, map_location="cpu",
                    weights_only=False)["R6"].double()
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")

    st = dict(stash)
    st["R1"] = R5
    out = {}
    for tag, r6 in (("R5_baselineR2", None), ("R5_R6", R6)):
        orig = fq._weight_fake_quant
        fq._weight_fake_quant = lambda w, b: w.clone()
        ad_fp = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
            r2_override=r6, **D4GS)
        ad_fp.install()
        W_fp = {k: getattr(ea.layers[0].self_attn,
                           f"{k}_proj").w_fake.detach().float().cpu()
                .clone() for k in ("v", "o")}
        ad_fp.uninstall()
        fq._weight_fake_quant = orig
        torch.cuda.empty_cache()

        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
            r2_override=r6, **D4GS)
        ad.install()
        sites = {}
        stats, hooks = {}, []
        for k in ("v", "o"):
            qw = getattr(ea.layers[0].self_attn,
                         f"{k}_proj").w_fake.detach().float().cpu()
            w0 = W_fp[k]
            sites[k] = dict(
                w4_nmse=round(float(((qw - w0) ** 2).sum()
                                    / (w0 ** 2).sum()), 6),
                **{kk: round(vv, 4) for kk, vv in wstats(w0).items()})
            stats[k] = ActStat()

            def mk(nm):
                def h(mod, inp):
                    stats[nm].update(inp[0])
                return h
            hooks.append(getattr(ea.layers[0].self_attn, f"{k}_proj")
                         .register_forward_pre_hook(mk(k)))
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            run_gen(model.ea_generate(
                ids, temperature=0.0,
                max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
        for h in hooks:
            h.remove()
        for k in sites:
            sites[k]["act"] = stats[k].result()
        ad.uninstall()
        torch.cuda.empty_cache()
        out[tag] = sites
        print(f"[r6-proxy] {tag} done", flush=True)

    dst = os.path.join(args.run_dir, "geometry", "r6_proxy_diag.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_G, r6_ckpt=args.r6_ckpt,
                   legs=out), open(dst, "w"), indent=1)
    print(f"[r6-proxy] -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
