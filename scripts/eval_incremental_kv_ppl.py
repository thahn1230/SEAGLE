#!/usr/bin/env python
"""Phase B2: incremental KV-sensitive PPL (spec §10).

Protocol: identical wikitext-2 test token stream (official tokenizer
contract) for every configuration. Prime a 256-token prefix (chunked
prefill through the SAME KVCache path), then evaluate tokens one at a time
with use_cache, recording per-token NLL and position, up to position 4096.

KV4 = fake-quant-at-append on the EAGLE KVCache (kv4_cache.install_kv4_on
_past) — the exact cache path the acceptance experiments use. Counters
assert the quantizer actually ran (no silent FP16 fallback).

FP16 incremental logits are validated against full-prefix logits first.

Writes <run>/tables/incremental_kv_ppl.csv + per-token parquet.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from eagle_spinquant import experiment, study
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle_spinquant.official_ppl import get_official_test_tokens

KIND = "learned_chat_w4a4kv16"
BUCKETS = [(0, 511), (512, 1023), (1024, 2047), (2048, 4095)]


@torch.no_grad()
def incremental_nll(model, ids, dev, kv_bits=16, n_positions=4096,
                    prime=256, validate_fp16=False):
    from eagle.model.kv_cache import initialize_past_key_values
    bm = model.base_model
    bm.model.tree_mask = None
    past, _pd, _cl = initialize_past_key_values(bm)
    stats = None
    if kv_bits < 16:
        k_stats, v_stats = install_kv4_on_past(past, bits=kv_bits)
        stats = (k_stats, v_stats)
    seq = ids[:, :n_positions + 1].to(dev)
    # prime prefix in chunks of 64 (multi-token appends exercise block cat)
    logits_prime = None
    pos = 0
    while pos < prime:
        chunk = seq[:, pos:pos + 64]
        out = bm(chunk, past_key_values=past, use_cache=True)
        logits_prime = out.logits
        pos += chunk.shape[1]
    rows = []
    if validate_fp16:
        full = bm(seq[:, :prime]).logits[0, -1].float()
        inc = logits_prime[0, -1].float()
        d = float((full - inc).abs().max())
        assert d < 0.02, f"incremental vs full-prefix logits diff {d}"
        print(f"[b2] fp16 incremental==full-prefix validated (max|d|={d:.4f})",
              flush=True)
    # token-by-token
    logits_last = logits_prime[0, -1]
    for t in range(prime, n_positions):
        tgt = seq[0, t]
        nll = float(F.cross_entropy(logits_last[None].float(),
                                    tgt[None]))
        rows.append(dict(pos=t, nll=nll))
        out = bm(seq[:, t:t + 1], past_key_values=past, use_cache=True)
        logits_last = out.logits[0, -1]
    if stats is not None:
        ks, vs = stats
        assert ks.n_tokens_quantized > 0 and vs.n_tokens_quantized > 0, \
            "KV4 requested but quantizer never invoked"
        kv_meta = dict(k_tokens=ks.n_tokens_quantized, k_nmse=ks.nmse,
                       v_tokens=vs.n_tokens_quantized, v_nmse=vs.nmse)
    else:
        kv_meta = dict(k_tokens=0, k_nmse=0.0, v_tokens=0, v_nmse=0.0)
    del past
    torch.cuda.empty_cache()
    return rows, kv_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--configs", default="fp16,w8a8,w4a4,rh0_w4a4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-positions", type=int, default=4096)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    ids = get_official_test_tokens(paths["target_path"])
    print(f"[b2] token stream: {ids.shape[1]} tokens (using first "
          f"{args.n_positions + 1})", flush=True)

    BUILD = {
        "fp16": ("none", KIND, "none"),
        "w8a8": ("full", KIND, "w8a8"),
        "w4a4": ("full", KIND, "w4a4"),
        "rh0_w4a4": ("full", "random_hadamard", "w4a4"),
    }
    summary, tok_rows = [], []
    for name in args.configs.split(","):
        rot, kind, quant = BUILD[name]
        print(f"[b2] building {name} ...", flush=True)
        model, stash, rbin = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, kind, quant, 0, device=dev, rotations_root=rr)
        for kv_bits in (16, 4):
            tag = f"{name}_kv{kv_bits}"
            rows, kv_meta = incremental_nll(
                model, ids, dev, kv_bits=kv_bits,
                n_positions=args.n_positions,
                validate_fp16=(name == "fp16" and kv_bits == 16))
            nlls = np.array([r["nll"] for r in rows])
            rec = dict(config=tag, rotation=kind if rot != "none" else "none",
                       wa=quant, kv_bits=kv_bits,
                       incremental_ppl=round(float(np.exp(nlls.mean())), 4),
                       incremental_ce=round(float(nlls.mean()), 5),
                       n_tokens=len(nlls), **{f"kv_{k}": round(v, 6)
                                              if isinstance(v, float) else v
                                              for k, v in kv_meta.items()})
            for lo, hi in BUCKETS:
                sel = nlls[[i for i, r in enumerate(rows)
                            if lo <= r["pos"] <= hi]]
                rec[f"ppl_{lo}_{hi}"] = round(float(np.exp(sel.mean())), 4) \
                    if len(sel) else None
            summary.append(rec)
            for r in rows:
                tok_rows.append(dict(config=tag, **r))
            print(f"[b2] {tag}: inc-PPL={rec['incremental_ppl']} "
                  f"kNMSE={kv_meta['k_nmse']:.5f} "
                  f"vNMSE={kv_meta['v_nmse']:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    pd.DataFrame(summary).to_csv(
        os.path.join(args.run_dir, "tables", "incremental_kv_ppl.csv"),
        index=False)
    pd.DataFrame(tok_rows).to_parquet(
        os.path.join(args.run_dir, "tables", "incremental_nll_per_token.parquet"),
        index=False)
    print("[b2] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
