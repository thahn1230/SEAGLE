#!/usr/bin/env python
"""§12 draft-quality probe: for the key draft variants, measure the draft's
FIRST-proposal top-1/top-k agreement against the FP16 draft on identical
contexts (prefixes sampled from the stock target's greedy continuations).

Runs under the stock FP16 target (isolates DRAFT changes; the draft root
distribution depends only on (prefix, hidden) which the shared target
provides identically for every variant).

Writes draft_agreement.csv into --run-dir.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)

KIND = "learned_chat_w4a4kv16"

VARIANTS = {
    "draft_fp16": dict(),
    "proj_W4A4_shared": dict(quant_first="fake_w4a4",
                             quant_recurrent="fake_w4a4"),
    "proj_W4A4_separate": dict(quant_first="fake_w4a4",
                               quant_recurrent="fake_w4a4",
                               branch_act=(4, 4)),
    "proj_W4A4_alpha": dict(quant_first="fake_w4a4",
                            quant_recurrent="fake_w4a4",
                            embed_scale_alpha="auto"),
    "full_policy_W4A4": dict(quant_first="fake_w4a4",
                             quant_recurrent="fake_w4a4",
                             quant_ar="fake_w4a4", ar_r2r4=True),
    "full_W4A4_alpha": dict(quant_first="fake_w4a4",
                            quant_recurrent="fake_w4a4",
                            quant_ar="fake_w4a4", ar_r2r4=True,
                            embed_scale_alpha="auto"),
}


@torch.no_grad()
def draft_root_logits(model, ad, ids, n_ctx=12, step=8):
    """For prefixes ids[:L+k*step], run target prefill then the draft FIRST
    projection + AR + head once; return root logits rows."""
    bm = model.base_model
    ea = model.ea_layer
    outs = []
    L = ids.shape[1]
    for k in range(n_ctx):
        end = L + k * step
        if end > ids.shape[1]:
            break
        seq = ids[:, :end]
        bm.model.tree_mask = None
        out = bm.model(seq, use_cache=False)
        hidden = out[0]
        tok_embed = ea.embed_tokens(seq)
        z = torch.cat([tok_embed[:, -1:], hidden[:, -1:]], dim=-1) \
            if False else None
        # use the adapter's own draft-first path via topK_genrate-equivalent:
        # emulate one draft step: concat(e_{t}, h_{t-1}) per EAGLE cnets
        e = ea.embed_tokens(seq[:, -1:])
        h = hidden[:, -1:]
        zin = torch.cat([e, h], dim=-1)
        ad.split.select = "first"
        y = ad.split(zin.to(torch.float16))
        ad.split.select = None
        lg = ad.head(y.to(ad.head.weight.dtype)) if hasattr(ad, "head") \
            else bm.lm_head(y)
        outs.append(lg[0, -1].float().cpu())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=8)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "none", KIND, "none", 0, device=dev, rotations_root=rr)
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    stash["R1"] = R["R1"].clone()
    stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
        .float().cpu().clone()
    stash["lm_head_weight"] = model.base_model.lm_head.weight.detach() \
        .float().cpu().clone()
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    # contexts: prompt + first 96 tokens of the saved fp16 naive continuation
    naive = torch.load(os.path.join(args.run_dir, "shards",
                                    "naive__fp16.pt"), weights_only=True)
    ids_list = []
    for p, nv in zip(prompts, naive):
        base = build_prompt(tok, p["text"])
        ids_list.append(torch.cat(
            [base, nv[:96][None]], dim=1).to(dev))

    alpha = json.load(open(os.path.join(
        args.run_dir, "alpha_selected.json")))["fp16"]["alpha"]
    ref = {}
    rows = []
    for vname, kw in VARIANTS.items():
        kw = dict(kw)
        if kw.get("embed_scale_alpha") == "auto":
            kw["embed_scale_alpha"] = alpha
        kw.setdefault("first_hidden_mode", "identity")
        ad = ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                         variant="folded", trace=False, **kw)
        ad.install()
        logits = []
        for ids in ids_list:
            logits += draft_root_logits(model, ad, ids)
        ad.uninstall()
        if vname == "draft_fp16":
            ref = logits
            rows.append(dict(variant=vname, n_contexts=len(logits),
                             top1_agree=1.0, top5_cover=1.0, kl=0.0))
            continue
        t1 = float(np.mean([int(a.argmax() == b.argmax())
                            for a, b in zip(logits, ref)]))
        t5 = float(np.mean([int(a.argmax() in b.topk(5).indices)
                            for a, b in zip(logits, ref)]))
        kl = float(np.mean([torch.nn.functional.kl_div(
            torch.log_softmax(a, -1), torch.softmax(b, -1),
            reduction="sum").item() for a, b in zip(logits, ref)]))
        rows.append(dict(variant=vname, n_contexts=len(logits),
                         top1_agree=round(t1, 4), top5_cover=round(t5, 4),
                         kl=round(kl, 4)))
        print(f"[probe] {vname}: top1={t1:.3f} top5={t5:.3f} KL={kl:.3f}",
              flush=True)
    logging_utils.write_csv(os.path.join(args.run_dir,
                                         "draft_agreement.csv"), rows)
    print("[probe] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
