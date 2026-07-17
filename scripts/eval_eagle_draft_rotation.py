#!/usr/bin/env python
"""Evaluate a saved draft rotation R_D in the REAL EAGLE runtime.

Constructs the concat-selective adapter with the draft gauge = R_D
(stash R1 <- R_D) and the first-path hidden fold kept at the TARGET
rotation R_T (first_fold_R; the folded T->D bridge). R_D == R_T reproduces
the validated shared-rotation path bit-for-bit.

--rotation FILE.pt   trainer checkpoint (R_D + alpha) or 'shared'
--target {t8,t4,t4kv4}
--datasets mtbench,c4,... --n-prompts 20 (screening) or 80 (full)
Writes shards/drot__<tag>__<dataset>.csv into --run-dir.
"""
import argparse, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle_spinquant.eval_datasets import (load_eval_prompts,
                                           write_or_verify_manifest)
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
TARGETS = {"t8": ("w8a8", 16), "t4": ("w4a4", 16), "t4kv4": ("w4a4", 4)}
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True)
D8 = dict(quant_first="fake_w8a8", quant_recurrent="fake_w8a8",
          quant_ar="fake_w8a8", ar_r2r4=True)


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
    ap.add_argument("--rotation", required=True,
                    help="checkpoint .pt or 'shared'")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", default="t4", choices=list(TARGETS))
    ap.add_argument("--draft", default="d4p3", choices=["d4p3", "d8"])
    ap.add_argument("--datasets", default="mtbench,sharegpt,c4,gsm8k,"
                                          "humaneval")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    rd = args.run_dir
    quant, t_kv = TARGETS[args.target]
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, quant, 0, device=dev, rotations_root=rr)
    if t_kv < 16:
        past, pkv_data, cur_len = initialize_past_key_values(
            model.base_model)
        model.past_key_values = past
        model.past_key_values_data = pkv_data
        model.current_length_data = cur_len
        install_kv4_on_past(past, bits=t_kv)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]

    R_T = stash["R1"].clone()
    kw = dict(D4P3 if args.draft == "d4p3" else D8)
    if args.rotation == "shared":
        alpha = 32.0
        if args.draft == "d4p3":
            kw["embed_scale_alpha"] = alpha
    else:
        ck = torch.load(args.rotation, map_location="cpu",
                        weights_only=False)
        stash = dict(stash)
        stash["R1"] = ck["R_D"].double()
        kw["first_fold_R"] = R_T                     # folded T->D bridge
        if args.draft == "d4p3":
            kw["embed_scale_alpha"] = float(ck.get("alpha", 32.0))
    for ds_name in args.datasets.split(","):
        out_csv = os.path.join(
            rd, "shards", f"drot__{args.tag}__{args.target}__{ds_name}.csv")
        if os.path.exists(out_csv):
            print(f"[drot] {args.tag}/{ds_name}: exists, skip", flush=True)
            continue
        prompts, man = load_eval_prompts(ds_name, args.n_prompts, "eval")
        write_or_verify_manifest(rd, ds_name, "eval", man) \
            if args.n_prompts == 80 else None
        ids_list = [build_prompt(tok, p["text"])[:, :1024].to(dev)
                    for p in prompts]
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, **kw)
        ad.install()
        rows = []
        for pi, ids in enumerate(ids_list):
            if hasattr(ad, "set_context"):
                ad.set_context(prompts[pi]["row_id"])
            _, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            rows.append(dict(tag=args.tag, target=args.target,
                             dataset=ds_name,
                             prompt_id=prompts[pi]["row_id"],
                             acceptance_list=json.dumps(deltas),
                             n_cycles=len(deltas)))
        ad.uninstall()
        taus = [t for r in rows for t in json.loads(r["acceptance_list"])]
        mal = sum(taus) / max(len(taus), 1)
        logging_utils.write_csv(out_csv, rows)
        print(f"[drot] {args.tag}/{args.target}/{ds_name}: "
              f"micro-AL={mal:.4f} ({len(taus)} cycles)", flush=True)
    print(f"[drot] {args.tag} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
