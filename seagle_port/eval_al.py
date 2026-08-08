"""Acceptance-length evaluation for DFlash × SpinQuant arms.

Writes shards/al__<tag>__<target>__<dataset>.csv with per-prompt cycle taus,
optionally cycles/cyc__<tag>__<dataset>.jsonl for RCAL replay.

Gate H (--gate-h): on target fp16 + stock interface, asserts hooked generate
reproduces upstream dflash_generate output_ids exactly on the first N prompts.

Usage:
  python -m seagle_port.eval_al --tag A0_T16D16 --target-mode fp16 \
      --interface stock --dataset mtbench --max-samples 80 --run-dir <RD>
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from . import spinquant_target as sq
from .generate import dflash_generate_hooked
from . import interfaces

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"

DATASET_ALIAS = {"mtbench": "mt-bench", "gsm8k": "gsm8k",
                 "humaneval": "humaneval", "math500": "math500",
                 "mbpp": "mbpp", "sharegpt": "sharegpt",
                 "gsm8kcalib": "gsm8kcalib", "gsm8kvalid": "gsm8kvalid"}


def load_dataset_rows(name, max_samples):
    from dflash.benchmark import _prepare_dataset, DATASETS
    if name in ("gsm8kcalib", "gsm8kvalid"):
        # disjoint train-split pools (SEAGLE convention: calib@500, valid@1000)
        from datasets import load_dataset as _ld
        off = 500 if name == "gsm8kcalib" else 1000
        ds = _ld("openai/gsm8k", "main", split="train")
        return [{"turns": [ds[i]["question"] +
                           "\nPlease reason step by step, and put your final "
                           "answer within \\boxed{}."]}
                for i in range(off, off + max_samples)]
    ds = DATASET_ALIAS[name]
    if ds == "sharegpt":
        path = os.path.join(os.path.dirname(__file__), "..", "cache",
                            "sharegpt.jsonl")
        rows = [json.loads(l) for l in open(path)]
    else:
        path = _prepare_dataset(ds) if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "cache",
                         f"{ds}.jsonl")) else os.path.join(
            os.path.dirname(__file__), "..", "cache", f"{ds}.jsonl")
        rows = [json.loads(l) for l in open(path)]
    return rows[:max_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target-mode", required=True,
                    choices=["fp16", "rot_fp16", "w8a8", "w4a4"])
    ap.add_argument("--interface", default="stock",
                    choices=["stock", "naive", "explicit", "folded",
                             "rotate"])
    ap.add_argument("--rbin", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--record-cycles", action="store_true")
    ap.add_argument("--gate-h", action="store_true")
    ap.add_argument("--draft-mode", default="fp16",
                    choices=["fp16", "w8a8", "w4a4"])
    ap.add_argument("--draft-components", default=None,
                    help="comma list (Gate F one-component runs)")
    ap.add_argument("--fc-p2", action="store_true",
                    help="independent per-branch act scales on fc")
    ap.add_argument("--mp3-scales", default=None,
                    help="comma list of 5 branch scales m_i (weight side "
                         "folded, activation side via ctx_transform)")
    ap.add_argument("--rc-ckpt", default=None,
                    help="R_C checkpoint; wraps draft in RCDraft (ctx-view "
                         "quantized K/V + context rotation)")
    ap.add_argument("--rc-identity", action="store_true",
                    help="RCDraft with R_C=I (RC0 protocol baseline)")
    ap.add_argument("--draft-transform", default=None,
                    help="module:callable applied to draft after load")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    dev = "cuda:0"
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel, dflash_generate
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, args.target_mode, rbin_path=args.rbin,
                             device=dev)
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()

    R1 = sq.load_rbin(args.rbin)["R1"] if args.rbin else None
    rotated = args.target_mode != "fp16"

    mp3 = None
    if args.mp3_scales:
        mp3 = [float(x) for x in args.mp3_scales.split(",")]
        assert len(mp3) == 5
    ctx_transform = interfaces.make_ctx_transform(args.interface, R1=R1,
                                                  mp3_scales=mp3)
    embed_fn = head_fn = None
    if rotated:
        embed_fn, head_fn = interfaces.make_embed_head_restore(target, R1)
    if args.interface == "folded":
        assert rotated
        draft = interfaces.fold_wc(draft, R1, mp3_scales=mp3)
    elif args.interface == "rotate":
        assert not rotated, "interface-only rotation is for fp16 targets"
        draft = interfaces.fold_wc(draft, R1, mp3_scales=mp3)
    elif mp3 is not None:
        draft = interfaces.fold_wc(draft, torch.eye(4096), mp3_scales=mp3)
    draft_prequant = draft
    if args.draft_mode != "fp16":
        from . import draft_quant
        bits = {"w8a8": (8, 8), "w4a4": (4, 4)}[args.draft_mode]
        comps = (args.draft_components.split(",")
                 if args.draft_components else None)
        draft = draft_quant.quantize_draft(
            draft, w_bits=bits[0], a_bits=bits[1], components=comps,
            fc_branch_dims=[4096] * 5 if args.fc_p2 else None)
        aud = draft_quant.audit_quantized(draft)
        print(f"[quant] {len(aud)} QLinear, all_w_changed="
              f"{all(r['w_changed'] for r in aud if r['a_bits'] < 16)}")
    if args.rc_ckpt or args.rc_identity:
        from .rc import load_rc_draft
        wb = ab = 4 if args.draft_mode == "w4a4" else \
            (8 if args.draft_mode == "w8a8" else 16)
        draft = load_rc_draft(draft, draft_prequant,
                              rc_ckpt=args.rc_ckpt, w_bits=wb,
                              a_bits=ab).to(dev)
        print(f"[rc] RCDraft active ckpt={args.rc_ckpt or 'identity'}")
    if args.draft_transform:
        mod, fn = args.draft_transform.split(":")
        import importlib
        draft = getattr(importlib.import_module(mod), fn)(draft)

    rows = load_dataset_rows(args.dataset, args.max_samples)
    bs = args.block_size or draft.block_size
    os.makedirs(os.path.join(args.run_dir, "shards"), exist_ok=True)
    shard = os.path.join(
        args.run_dir, "shards",
        f"al__{args.tag}__{args.target_mode}__{args.dataset}.csv")
    cyc_f = None
    if args.record_cycles:
        os.makedirs(os.path.join(args.run_dir, "cycles"), exist_ok=True)
        cyc_f = open(os.path.join(
            args.run_dir, "cycles",
            f"cyc__{args.tag}__{args.dataset}.jsonl"), "w")

    from dflash.benchmark import _apply_chat_template
    all_taus = []
    with open(shard, "w") as f:
        f.write("prompt_id,turn,n_cycles,taus\n")
        for pid, inst in enumerate(rows):
            messages = []
            for ti, user_content in enumerate(inst["turns"]):
                messages.append({"role": "user", "content": user_content})
                text = _apply_chat_template(tok, messages, False)
                ids = tok.encode(text, return_tensors="pt").to(dev)
                out = dflash_generate_hooked(
                    draft, target, ids, args.max_new_tokens,
                    [tok.eos_token_id], 0.0, block_size=bs,
                    ctx_transform=ctx_transform, embed_fn=embed_fn,
                    head_fn=head_fn, record_cycles=args.record_cycles)
                if args.gate_h and pid < 3 and not rotated \
                        and args.interface == "stock":
                    torch.manual_seed(0)
                    ref = dflash_generate(draft, target=target, input_ids=ids,
                                          max_new_tokens=args.max_new_tokens,
                                          stop_token_ids=[tok.eos_token_id],
                                          temperature=0.0, block_size=bs)
                    assert torch.equal(out.output_ids, ref), \
                        f"GATE H FAIL prompt {pid}"
                    print(f"[gate_h] prompt {pid} identical")
                taus = out.acceptance_lengths
                all_taus.extend(taus)
                f.write(f"{pid},{ti},{len(taus)},\"{';'.join(map(str, taus))}\"\n")
                if cyc_f:
                    cyc_f.write(json.dumps({
                        "type": "turn_header", "prompt_id": pid, "turn": ti,
                        "input_ids": ids[0].tolist()}) + "\n")
                    for c in out.cycles:
                        c.update({"prompt_id": pid, "turn": ti})
                        cyc_f.write(json.dumps(c) + "\n")
                gen = out.output_ids[0, out.num_input_tokens:]
                messages.append({"role": "assistant", "content": tok.decode(
                    gen, skip_special_tokens=True)})
            if (pid + 1) % 10 == 0:
                print(f"[{args.tag}] {pid+1}/{len(rows)} "
                      f"cycle-tau={np.mean(all_taus):.4f}", flush=True)
    if cyc_f:
        cyc_f.close()
    print(f"[{args.tag}] DONE {args.dataset} cycle-pooled tau = "
          f"{np.mean(all_taus):.4f} over {len(all_taus)} cycles")


if __name__ == "__main__":
    main()
