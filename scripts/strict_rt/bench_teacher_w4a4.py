#!/usr/bin/env python
"""Strict SEAGLE-RT pre-flight: benchmark the DEPLOYED W4A4 SpinQuant
teacher forward (native tap) vs the plain FP16 teacher.

Purpose (speed-amendment items 1/5/18): measure sec/conversation for
 (a) deployed W4A4 target  (build_study_target rot='full' quant='w4a4',
     canonical learned_chat_w4a4kv16 R.bin, tap = base_model.model(...)
     .last_hidden_state — the EXACT eval-time native tensor a_t)
 (b) plain FP16 target (the old online-teacher cost reference)

Also records tokens/s, peak VRAM, and the per-token cache footprint so
the storage analytics can be cross-checked. Timing-only inputs come from
the surviving 12.4k token cache (content irrelevant for speed).

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/strict_rt/bench_teacher_w4a4.py \
          --out runs/<run>/tables/bench_teacher.json [--n 24]
"""
import argparse, json, os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

OLD_TOK_CACHE = "/data/thahn1230/datasets/sharegpt/eagle_tok_cache_v1.pt"
KIND = "learned_chat_w4a4kv16"


def load_convs(n):
    obj = torch.load(OLD_TOK_CACHE, map_location="cpu", weights_only=False)
    rows = obj["rows"] if isinstance(obj, dict) and "rows" in obj else obj
    out = []
    for r in rows:
        ids = r["input_ids"] if isinstance(r, dict) else r[0]
        ids = ids.to(torch.long).view(1, -1)
        if ids.shape[1] >= 256:
            out.append(ids)
        if len(out) == n:
            break
    return out


def bench(fwd, convs, dev, warmup=3):
    for ids in convs[:warmup]:
        fwd(ids.to(dev))
    torch.cuda.synchronize()
    t0, ntok = time.time(), 0
    for ids in convs:
        fwd(ids.to(dev))
        ntok += ids.shape[1]
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dict(sec_per_conv=round(dt / len(convs), 4),
                tok_per_s=round(ntok / dt, 1),
                n_convs=len(convs), n_tokens=ntok,
                total_sec=round(dt, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda:0"
    torch.set_grad_enabled(False)

    from eagle_spinquant import experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    convs = load_convs(args.n)
    res = {"convs_mean_len": round(sum(c.shape[1] for c in convs)
                                   / len(convs), 1)}

    # (a) deployed W4A4 target — exact eval-time build
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=None)
    bm = model.base_model.model
    torch.cuda.reset_peak_memory_stats()
    res["w4a4_native"] = bench(lambda ids: bm(input_ids=ids), convs, dev)
    res["w4a4_native"]["peak_mem_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    # record one sample tensor contract
    h = bm(input_ids=convs[0].to(dev)).last_hidden_state
    res["tap"] = dict(shape=list(h.shape), dtype=str(h.dtype),
                      rms=round(float(h.float().pow(2).mean().sqrt()), 4),
                      bytes_per_token=int(h.element_size() * h.shape[-1]))
    del model, bm, h
    torch.cuda.empty_cache()

    # (b) plain FP16 target (reference)
    from transformers import AutoModelForCausalLM
    tgt = AutoModelForCausalLM.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(dev).eval()
    torch.cuda.reset_peak_memory_stats()
    res["fp16_plain"] = bench(lambda ids: tgt.model(input_ids=ids), convs, dev)
    res["fp16_plain"]["peak_mem_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    res["w4a4_over_fp16"] = round(
        res["w4a4_native"]["sec_per_conv"]
        / res["fp16_plain"]["sec_per_conv"], 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
