"""Build the draft rotation/QAT training corpus (§5/§18 of the update).

Sources (test-disjoint):
  - previous RCCAP deployed trajectories (gsm8k-train offset-500 pool)
  - fresh greedy target continuations on sharegpt-calib prompts
    (filtered order offset 500, mirrors eval_datasets convention)
Format: jsonl rows {input_ids: [...], gen_start: int} — loss_mask is
1 for positions >= gen_start (generated part), 0 for the prompt.
Held-out validation = last --val-frac of rows (fixed order, no shuffle).
"""
import argparse
import json
import os

import torch

from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PREV = "runs/dflash_seagle_transfer_20260807_180238"


def from_rccap(path):
    rows = []
    turns = {}
    order = []
    for ln in open(path):
        r = json.loads(ln)
        key = (r["prompt_id"], r["turn"])
        if r.get("type") == "turn_header":
            turns[key] = {"ids": r["input_ids"], "blocks": []}
            order.append(key)
        else:
            turns[key]["blocks"].append(r["block"][:r["tau"]])
    for key in order:
        t = turns[key]
        gen = [x for b in t["blocks"] for x in b]
        rows.append({"input_ids": t["ids"] + gen,
                     "gen_start": len(t["ids"])})
    return rows


@torch.inference_mode()
def gen_sharegpt_calib(n, max_new, device):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, "fp16", device=device)
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    prompts, i = [], 0
    for idx in range(len(ds)):
        conv = ds[idx].get("conversations") or []
        human = next((c["value"] for c in conv if c.get("from") == "human"),
                     None)
        if not human or not (60 <= len(human) <= 1200):
            continue
        if i >= 500:                      # calib pool offset (disjoint)
            prompts.append(human.strip())
            if len(prompts) == n:
                break
        i += 1
    rows = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.encode(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
            return_tensors="pt").to(device)
        out = target.generate(ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        rows.append({"input_ids": out[0].tolist(),
                     "gen_start": ids.shape[1]})
        print(f"[corpus] sharegpt {len(rows)}/{n}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sharegpt", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows = from_rccap(f"{PREV}/cycles/cyc__RCCAP__gsm8kcalib.jsonl")
    print(f"[corpus] rccap rows: {len(rows)}")
    rows += gen_sharegpt_calib(args.n_sharegpt, args.max_new, args.device)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[corpus] DONE {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
