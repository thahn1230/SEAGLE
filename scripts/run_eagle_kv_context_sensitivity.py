#!/usr/bin/env python
"""Phase F: KV4 context-length sensitivity (spec §14).

Natural-text prefixes (wikitext-2 test, held-out from calibration) padded
to ~{256, 512, 1024, 2048, 4096} tokens, then 64 generated tokens with:

  A) T4_KV16 + D16          B) T4_KV4 + D16      (target KV4 effect)
  C) T4_KV16 + D4P3_KV16    D) T4_KV16 + D4P3_KV4 (draft KV4 effect)
  E) T4_KV4  + D4P3_KV4     (combined)

Records micro-AL, first-rejection depths, K/V NMSE, peak memory.
12 prompts per length bucket.
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
from eagle_spinquant.study import UnrotateAdapter
from eagle_spinquant.kv4_cache import install_kv4_on_past, DraftKV4Patch
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
ALPHA_G = 32.0
LENGTHS = [256, 512, 1024, 2048, 4096]
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G, first_hidden_mode="gamma_R1")
CONFIGS = {
    "T4KV16_D16": (16, None, 16),
    "T4KV4_D16": (4, None, 16),
    "T4KV16_D4P3KV16": (16, D4P3, 16),
    "T4KV16_D4P3KV4": (16, D4P3, 4),
    "T4KV4_D4P3KV4": (4, D4P3, 4),
}


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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-length", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    rd = args.run_dir
    os.makedirs(os.path.join(rd, "shards"), exist_ok=True)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    # one target build serves KV16 and KV4 (separate persistent pasts)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    all_ids = tok(text, return_tensors="pt").input_ids[0]

    rows = []
    for cname, (t_kv, dkw, d_kv) in CONFIGS.items():
        out_csv = os.path.join(rd, "shards", f"ctx__{cname}.csv")
        if os.path.exists(out_csv):
            print(f"[ctx] {cname}: exists, skip", flush=True)
            continue
        past, pkv_data, cur_len = initialize_past_key_values(
            model.base_model)
        model.past_key_values = past
        model.past_key_values_data = pkv_data
        model.current_length_data = cur_len
        t_stats = install_kv4_on_past(past, bits=t_kv) if t_kv < 16 else None
        ad = None
        if dkw is None:
            ad = UnrotateAdapter(model, stash, dev, torch.float16,
                                 with_gamma=True)
        else:
            ad = ConcatSelectiveDraftAdapter(
                model, stash, dev, torch.float16, variant="folded",
                trace=False, **dkw)
        ad.install()
        dpatch = DraftKV4Patch(model.ea_layer, bits=d_kv).install() \
            if d_kv < 16 else None
        crow = []
        for L in LENGTHS:
            taus_all = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(args.n_per_length):
                s = 1000 + i * 4200
                ids = all_ids[s:s + L - 8][None].to(dev)
                if ids.shape[1] < L - 16:
                    continue
                _, deltas = run_gen(model.ea_generate(
                    ids, temperature=0.0,
                    max_steps=args.max_new_tokens + 8,
                    tree_choices=tree), ids.shape[1], args.max_new_tokens)
                taus_all += deltas
            rec = dict(config=cname, ctx_len=L,
                       micro_al=round(sum(taus_all) / max(len(taus_all), 1),
                                      4),
                       n_cycles=len(taus_all),
                       peak_mem_gb=round(
                           torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            if t_stats:
                rec.update(t_k_nmse=round(t_stats[0].nmse, 6),
                           t_v_nmse=round(t_stats[1].nmse, 6))
            if dpatch:
                rec.update(d_k_nmse=round(dpatch.k_stats.nmse, 6),
                           d_v_nmse=round(dpatch.v_stats.nmse, 6))
            crow.append(rec)
            print(f"[ctx] {cname} L={L}: micro-AL={rec['micro_al']} "
                  f"cycles={rec['n_cycles']}", flush=True)
        if dpatch:
            dpatch.uninstall()
        ad.uninstall()
        del model.past_key_values, model.past_key_values_data, \
            model.current_length_data, past
        torch.cuda.empty_cache()
        logging_utils.write_csv(out_csv, crow)
    print("[ctx] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
