#!/usr/bin/env python
"""Phase E: 4x4 target x draft acceptance matrix with KV4 (spec §13).

Targets: T16_KV16, T8_KV16, T4_KV16, T4_KV4
Drafts:  D16_KV16, D8_KV16, D4P3_KV16, D4P3_KV4

Target KV4 = fake-quant-at-append on the EAGLE KVCache (install_kv4_on_past
on the model's persistent past). Draft KV4 = DraftKV4Patch on ea_layer
(append-time quantization of stable/tree KV). Counters recorded per cell;
a KV4 cell aborts if its quantizer never ran (no silent fallback).

MT-Bench 80 prompts, greedy, 128 new tokens, mc_sim_7b_63, seed 0.
Alpha (P3): per interface mode from the validated LRAS calibration
(identity 45.2548, gamma_R1 32.0).
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
ALPHA = {"identity": 45.254833995939045, "gamma_R1": 32.0}
TARGETS = {
    "T16_KV16": ("none", "none", 16),
    "T8_KV16": ("full", "w8a8", 16),
    "T4_KV16": ("full", "w4a4", 16),
    "T4_KV4": ("full", "w4a4", 4),
}
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True)
D8 = dict(quant_first="fake_w8a8", quant_recurrent="fake_w8a8",
          quant_ar="fake_w8a8", ar_r2r4=True)
DRAFTS = {
    "D16_KV16": (None, 16),
    "D8_KV16": (D8, 16),
    "D4P3_KV16": (D4P3, 16),
    "D4P3_KV4": (D4P3, 4),
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
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
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
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    for tname in args.targets.split(","):
        rot, quant, t_kv = TARGETS[tname]
        pending = [d for d in DRAFTS if not os.path.exists(os.path.join(
            rd, "shards", f"kv4mat__{tname}__{d}.csv"))]
        if not pending:
            print(f"[kv4] {tname}: done, skip", flush=True)
            continue
        print(f"[kv4] building {tname} ...", flush=True)
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
        # persistent past (reused by ea_generate; lengths reset per prompt)
        past, pkv_data, cur_len = initialize_past_key_values(model.base_model)
        model.past_key_values = past
        model.past_key_values_data = pkv_data
        model.current_length_data = cur_len
        t_kv_stats = None
        if t_kv < 16:
            t_kv_stats = install_kv4_on_past(past, bits=t_kv)
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
        fhm = "identity" if rot == "none" else "gamma_R1"

        for dname in pending:
            dkw, d_kv = DRAFTS[dname]
            ad, dpatch = None, None
            if dkw is None:
                if rot != "none":
                    ad = UnrotateAdapter(model, stash, dev, torch.float16,
                                         with_gamma=True)
            else:
                kw = dict(dkw)
                if kw.get("quant_first") == "fake_w4a4":
                    kw["embed_scale_alpha"] = ALPHA[fhm]     # P3
                ad = ConcatSelectiveDraftAdapter(
                    model, stash, dev, torch.float16, variant="folded",
                    first_hidden_mode=fhm, trace=False, **kw)
            if ad is not None:
                ad.install()
            if d_kv < 16:
                dpatch = DraftKV4Patch(model.ea_layer, bits=d_kv).install()
            t_tok0 = (t_kv_stats[0].n_tokens_quantized
                      if t_kv_stats else 0)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            t0 = time.time()
            for pi, ids in enumerate(ids_list):
                if ad is not None and hasattr(ad, "set_context"):
                    ad.set_context(prompts[pi]["question_id"])
                eg, deltas = run_gen(model.ea_generate(
                    ids, temperature=0.0,
                    max_steps=args.max_new_tokens + 8,
                    tree_choices=tree), ids.shape[1], args.max_new_tokens)
                depths = [d - 1 for d in deltas]
                rows.append(dict(
                    target=tname, draft=dname,
                    prompt_id=prompts[pi]["question_id"],
                    acceptance_list=json.dumps(deltas),
                    first_rejection_depths=json.dumps(depths),
                    n_new_tokens=len(eg), n_cycles=len(deltas)))
            meta = dict(
                cell=f"{tname}__{dname}", quant_mode="fake",
                target_kv_bits=t_kv, draft_kv_bits=d_kv,
                peak_mem_gb=round(
                    torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                wall_s=round(time.time() - t0, 1))
            if t_kv_stats:
                ks, vs = t_kv_stats
                assert ks.n_tokens_quantized > t_tok0, \
                    f"{tname}: target KV4 requested but not invoked"
                meta.update(target_k_nmse=round(ks.nmse, 6),
                            target_v_nmse=round(vs.nmse, 6),
                            target_kv_tokens=ks.n_tokens_quantized)
            if dpatch is not None:
                assert dpatch.k_stats.n_tokens_quantized > 0, \
                    "draft KV4 requested but not invoked"
                meta.update(draft_k_nmse=round(dpatch.k_stats.nmse, 6),
                            draft_v_nmse=round(dpatch.v_stats.nmse, 6),
                            draft_kv_tokens=dpatch.k_stats.n_tokens_quantized)
                dpatch.uninstall()
            if ad is not None:
                ad.uninstall()
            taus = [t for r in rows
                    for t in json.loads(r["acceptance_list"])]
            mal = sum(taus) / max(len(taus), 1)
            meta["micro_al"] = round(mal, 4)
            logging_utils.write_csv(os.path.join(
                rd, "shards", f"kv4mat__{tname}__{dname}.csv"), rows)
            with open(os.path.join(
                    rd, "shards", f"kv4mat__{tname}__{dname}.json"),
                    "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[kv4] {tname}/{dname}: micro-AL={mal:.4f} "
                  f"{ {k: v for k, v in meta.items() if 'nmse' in k} } "
                  f"({meta['wall_s']}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("[kv4] matrix DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
