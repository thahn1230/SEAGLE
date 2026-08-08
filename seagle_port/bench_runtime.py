"""Phase 27: per-stage runtime breakdown (CUDA events, fake-quant timings).

Stages per decode cycle: draft_forward (incl. fc/hidden_norm/ctx K/V),
embed_restore, head_restore, ctx_transform, target_verify, extract_concat.
Plus prefill and end-to-end ms/token. mean/median/p95 over cycles after
2-prompt warmup. FAKE-QUANT NUMBERS — no real low-bit kernel claims.

Usage: python -m seagle_port.bench_runtime --arm <name> --run-dir RD ...
Arms are specified with the same flags as eval_al (subset).
"""
import argparse
import json
import os

import numpy as np
import torch

from . import spinquant_target as sq
from . import interfaces
from dflash.model import extract_context_feature, sample

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"


class Ev:
    def __init__(self):
        self.acc = {}

    def timed(self, name, fn, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*a, **k)
        e.record()
        e.synchronize()
        self.acc.setdefault(name, []).append(s.elapsed_time(e))
        return out


@torch.inference_mode()
def run(args, dev="cuda:0"):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from dflash.benchmark import _apply_chat_template
    from transformers import AutoTokenizer, DynamicCache

    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, args.target_mode, rbin_path=args.rbin,
                             device=dev)
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    R1 = sq.load_rbin(args.rbin)["R1"] if args.rbin else None
    mp3 = [float(x) for x in args.mp3_scales.split(",")] \
        if args.mp3_scales else None
    rotated = args.target_mode != "fp16"
    ctx_t = interfaces.make_ctx_transform(args.interface, R1=R1,
                                          mp3_scales=mp3)
    ef = hf = None
    if rotated:
        ef, hf = interfaces.make_embed_head_restore(target, R1)
    if args.interface in ("folded", "rotate"):
        draft = interfaces.fold_wc(draft, R1 if R1 is not None
                                   else torch.eye(4096), mp3_scales=mp3)
    if args.draft_mode != "fp16":
        from . import draft_quant
        b = {"w8a8": (8, 8), "w4a4": (4, 4)}[args.draft_mode]
        draft = draft_quant.quantize_draft(
            draft, w_bits=b[0], a_bits=b[1],
            fc_branch_dims=[4096] * 5 if args.fc_p2 else None)
    ef = ef or target.model.embed_tokens
    hf = hf or target.lm_head

    rows = [json.loads(l) for l in open(os.path.join(
        os.path.dirname(__file__), "..", "cache", "mt-bench.jsonl"))][
        :args.n_prompts]
    ev = Ev()
    e2e = []
    B = draft.block_size
    mask_id = draft.mask_token_id
    for pi, inst in enumerate(rows):
        warm = pi < 2
        msgs = [{"role": "user", "content": inst["turns"][0]}]
        ids = tok.encode(_apply_chat_template(tok, msgs, False),
                         return_tensors="pt").to(dev)
        n_in = ids.shape[1]
        maxlen = n_in + args.max_new_tokens
        out_ids = torch.full((1, maxlen + B), mask_id, dtype=torch.long,
                             device=dev)
        pos = torch.arange(out_ids.shape[1], device=dev).unsqueeze(0)
        kvt, kvd = DynamicCache(), DynamicCache()
        E = Ev() if warm else ev
        o = E.timed("prefill", target, ids, position_ids=pos[:, :n_in],
                    past_key_values=kvt, use_cache=True, logits_to_keep=1,
                    output_hidden_states=True)
        out_ids[:, :n_in] = ids
        out_ids[:, n_in:n_in + 1] = sample(o.logits, 0.0)
        th = E.timed("extract_concat", extract_context_feature,
                     o.hidden_states, draft.target_layer_ids)
        if ctx_t is not None:
            th = E.timed("ctx_transform", ctx_t, th)
        start = n_in
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        while start < maxlen:
            blk = out_ids[:, start:start + B].clone()
            ne = E.timed("embed_restore", ef, blk)
            dh = E.timed("draft_forward", draft, target_hidden=th,
                         noise_embedding=ne,
                         position_ids=pos[:, kvd.get_seq_length():start + B],
                         past_key_values=kvd, use_cache=True, is_causal=False)
            dl = E.timed("head_restore", hf, dh[:, 1 - B:, :])
            kvd.crop(start)
            blk[:, 1:] = sample(dl, 0.0)
            o = E.timed("target_verify", target, blk,
                        position_ids=pos[:, start:start + B],
                        past_key_values=kvt, use_cache=True,
                        output_hidden_states=True)
            post = sample(o.logits, 0.0)
            L = (blk[:, 1:] == post[:, :-1]).cumprod(1).sum(1)[0].item()
            out_ids[:, start:start + L + 1] = blk[:, :L + 1]
            out_ids[:, start + L + 1] = post[:, L]
            start += L + 1
            kvt.crop(start)
            th = E.timed("extract_concat", extract_context_feature,
                         o.hidden_states, draft.target_layer_ids)[
                :, :L + 1, :]
            if ctx_t is not None:
                th = E.timed("ctx_transform", ctx_t, th)
            if tok.eos_token_id in out_ids[0, n_in:start]:
                break
        torch.cuda.synchronize()
        if not warm:
            e2e.append((time.perf_counter() - t0) * 1000 / max(start - n_in, 1))
    res = {"arm": args.arm,
           "config": {k: v for k, v in vars(args).items()
                      if k not in ("run_dir",)},
           "ms_per_token_e2e": {"mean": float(np.mean(e2e)),
                                "median": float(np.median(e2e)),
                                "p95": float(np.percentile(e2e, 95))}}
    for k, v in ev.acc.items():
        res[k] = {"mean_ms": float(np.mean(v)),
                  "median_ms": float(np.median(v)),
                  "p95_ms": float(np.percentile(v, 95)), "n": len(v)}
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(
        args.run_dir, "tables", f"runtime__{args.arm}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("[bench] DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--target-mode", default="fp16")
    ap.add_argument("--interface", default="stock")
    ap.add_argument("--rbin", default=None)
    ap.add_argument("--draft-mode", default="fp16")
    ap.add_argument("--fc-p2", action="store_true")
    ap.add_argument("--mp3-scales", default=None)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
