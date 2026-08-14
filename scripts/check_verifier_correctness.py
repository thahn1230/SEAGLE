#!/usr/bin/env python
"""S26 verifier correctness: EAGLE greedy output must equal target-only
greedy output. Exact for KV16 configs; for target-KV4 the quantizer sees
different append groupings (tree burst + compaction vs one-token AR), so
the caches differ numerically and divergence is possible — we MEASURE it
instead of assuming.

For each config: N prompts, 128 tokens; EAGLE ea_generate tokens vs
base-model AR greedy tokens (same quantized target, same KV4 wrapper).
Reports exact-prefix match length fraction and first-divergence index.
Writes tables/verifier_correctness.csv.
"""
import argparse, csv, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.kv4_cache import install_kv4_on_past, DraftKV4Patch
from eagle_spinquant.eval_datasets import load_eval_prompts
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True, embed_scale_alpha=32.0)
CONFIGS = {  # name: (target_quant, target_kv_bits, draft_on, draft_kv)
    "T16_D16": ("none", 16, False, 16),
    "T4KV16_D4P3": ("w4a4", 16, True, 16),
    "T4KV4_D4P3KV4": ("w4a4", 4, True, 4),
}


def ar_greedy(model, ids, n, ar_kv):
    """Target-only greedy through the SAME persistent KVCache path.
    ar_kv = (past, cur) allocated once per config and reused."""
    from eagle.model.utils import reset_tree_mode
    reset_tree_mode(model)          # ea_generate leaves tree mask armed
    past, cur = ar_kv
    toks = []
    cur.zero_()
    inp = ids
    for _ in range(n):
        out = model.base_model(inp, past_key_values=past,
                               use_cache=True)
        nxt = out.logits[0, -1].argmax().item()
        toks.append(nxt)
        inp = torch.tensor([[nxt]], device=ids.device)
    return toks


def run_gen(gen, ilen, mx):
    final = None
    for out in gen:
        final = out
        if out.shape[1] - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    # AA-QAT study (2026-08-14): parity-INVARIANCE for retrained drafts —
    # same T4KV16_D4P3 lane, optionally with a QAT state dict, a fixed
    # R_D basis, and P3 scales; output suffixed with --tag
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--rd-ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alpha-rec", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default=None,
                    help="run just this CONFIGS key (e.g. T4KV16_D4P3)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    tdir = os.path.join(args.run_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    for cname, (tq, t_kv, d_on, d_kv) in CONFIGS.items():
        if args.only and cname != args.only:
            continue
        shard = os.path.join(tdir, f"vc__{cname}{args.tag}.csv")
        if os.path.exists(shard):
            print(f"[vc] {cname}{args.tag}: exists, skip", flush=True)
            continue
        rows = []
        kind = "none" if tq == "none" else KIND
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], "full" if tq != "none" else "none",
            kind, tq if tq != "none" else "none", 0, device=dev,
            rotations_root=rr)
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        build_prompt = eagle_bridge.PROMPT_BUILDERS[
            cfg.get("model", {}).get("chat_template", "llama2")]
        if t_kv < 16:
            past, pkv_data, cur_len = initialize_past_key_values(
                model.base_model)
            model.past_key_values = past
            model.past_key_values_data = pkv_data
            model.current_length_data = cur_len
            install_kv4_on_past(past, bits=t_kv)
        ad = None
        if d_on:
            if args.draft_sd:
                sd_new = torch.load(args.draft_sd, map_location="cpu",
                                    weights_only=False)
                sd_new = sd_new.get("draft_state_dict",
                                    sd_new.get("model", sd_new))
                ea = model.ea_layer
                ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                                    for k, v in sd_new.items()},
                                   strict=True)
                ea.to(dev)
            st_use = stash
            kw = dict(D4P3)
            if args.rd_ckpt:
                rck = torch.load(args.rd_ckpt, map_location="cpu",
                                 weights_only=False)
                st_use = dict(stash)
                kw["first_fold_R"] = stash["R1"].clone()
                st_use["R1"] = rck["R_D"].double()
            if args.alpha is not None:
                kw["embed_scale_alpha"] = args.alpha
            if args.alpha_rec is not None:
                kw["embed_scale_alpha_rec"] = args.alpha_rec
            ad = ConcatSelectiveDraftAdapter(
                model, st_use, dev, torch.float16, variant="folded",
                first_hidden_mode="gamma_R1", trace=False, **kw)
            ad.install()
        dpatch = DraftKV4Patch(model.ea_layer, bits=d_kv).install() \
            if d_kv < 16 else None
        ar_past, ar_pkv, ar_cur = initialize_past_key_values(
            model.base_model)
        if t_kv < 16:
            install_kv4_on_past(ar_past, bits=t_kv)
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            ea = run_gen(model.ea_generate(
                ids, temperature=0.0,
                max_steps=args.max_new_tokens + 8, tree_choices=tree),
                ids.shape[1], args.max_new_tokens)
            ar = ar_greedy(model, ids, args.max_new_tokens,
                           (ar_past, ar_cur))
            div = next((i for i, (a, b) in enumerate(zip(ea, ar))
                        if a != b), len(ea))
            rows.append(dict(config=cname, prompt_id=p["row_id"],
                             match_prefix=div, n=len(ea),
                             frac=round(div / len(ea), 4)))
            print(f"[vc] {cname} {p['row_id']}: prefix {div}/{len(ea)}",
                  flush=True)
        logging_utils.write_csv(shard, rows)
        fr = [r["frac"] for r in rows]
        print(f"[vc] {cname}: mean prefix-match frac = "
              f"{sum(fr)/len(fr):.4f}", flush=True)
        if dpatch:
            dpatch.uninstall()
        if ad:
            ad.uninstall()
        if hasattr(model, "past_key_values"):
            del model.past_key_values, model.past_key_values_data, \
                model.current_length_data
        del model, stash, ad, dpatch, tok, build_prompt, \
            ar_past, ar_pkv, ar_cur
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    merged = []
    for cname in CONFIGS:
        p = os.path.join(tdir, f"vc__{cname}.csv")
        if os.path.exists(p):
            with open(p) as f:
                merged.extend(list(csv.DictReader(f)))
    logging_utils.write_csv(
        os.path.join(tdir, "verifier_correctness.csv"), merged)
    print("[vc] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
