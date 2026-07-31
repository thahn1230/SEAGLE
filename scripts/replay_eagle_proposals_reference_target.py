#!/usr/bin/env python
"""Offline FP16-reference replay of captured proposal trees (study §15).

Reads cycles/cyc__<tag>__<ds>.jsonl (which stores, per cycle, the
flattened tree candidate tokens and the deployed accepted trajectory),
rebuilds the reference T0 KV by teacher-forcing the DEPLOYED trajectory,
re-verifies every stored tree, and checks that the recomputed accepted
sequence S_0 matches the one recorded by the inline lockstep capture.

This is the reference-cache-equivalence control: inline lockstep KV
compression and offline trajectory reconstruction must agree exactly.
Writes tables/replay_equiv_<tag>.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment
from eagle_spinquant.eval_datasets import load_eval_prompts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="mtbench")
    ap.add_argument("--pool", default="eval")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rd, dev = args.run_dir, args.device
    cyc_p = os.path.join(rd, "cycles",
                         f"cyc__{args.tag}__{args.dataset}.jsonl")
    recs = [json.loads(x) for x in open(cyc_p)]
    if not recs or "tree_tokens" not in recs[0]:
        print(f"[replay] {args.tag}: records lack tree_tokens "
              f"(older capture) -> skip")
        return 0
    by_prompt = {}
    for r in recs:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    ref = KVLlama.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(dev).eval()

    class TokOnly:
        base_model = ref
    tok = __import__("transformers").AutoTokenizer.from_pretrained(
        paths["target_path"])
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts(args.dataset, args.n_prompts,
                                   args.pool)
    pmap = {p["row_id"]: p for p in prompts}
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    from eagle.model.ea_model import generate_tree_buffers
    from eagle.model.kv_cache import initialize_past_key_values
    tb = generate_tree_buffers([list(p) for p in tree_full], device=dev)
    ri = tb["retrieve_indices"]

    n_cyc = n_match = 0
    for pid, rows in list(by_prompt.items())[:args.max_prompts]:
        if pid not in pmap:
            continue
        ids = build_prompt(tok, pmap[pid]["text"])[:, :1024].to(dev)
        ref.model.tree_mask = None
        past, past_data, cur_len = initialize_past_key_values(ref)
        ref(ids, past_key_values=past, use_cache=True)
        plen = ids.shape[1]
        for r in sorted(rows, key=lambda x: x["cycle"]):
            assert r["prefix_len"] == plen, (r["prefix_len"], plen)
            tree_cands = torch.tensor([r["tree_tokens"]], device=dev)
            ref.model.tree_mask = tb["tree_attn_mask"]
            pos = (tb["tree_position_ids"] + plen)[None]
            lg0 = ref(tree_cands, past_key_values=past,
                      position_ids=pos, use_cache=True).logits[0, ri]
            ext = torch.cat([tree_cands[0], torch.zeros(
                1, dtype=torch.long, device=dev)])
            cands = ext[ri]   # -1 padding wraps to appended 0
            pm0 = (cands[:, 1:] == torch.argmax(lg0[:, :-1],
                                                dim=-1)).int()
            cal0 = torch.cumprod(pm0, dim=1).sum(dim=1)
            len_0 = int(cal0.max())
            best_0 = (int(torch.argmax(cal0)) if len_0 > 0 else 0)
            S_0 = cands[best_0, 1:1 + len_0].tolist()
            n_cyc += 1
            n_match += int(S_0 == r["S_0"] and len_0 == r["R_0"])
            # advance along DEPLOYED trajectory (deployed best/len)
            len_q = r["R_q"]
            sel = ri[r["best_q"], :len_q + 1] + plen
            for rp in past_data:
                tgt = rp[..., sel.to(rp.device), :]
                rp[..., plen:plen + tgt.shape[-2], :].copy_(tgt)
            cur_len.fill_(plen + len_q + 1)
            plen = plen + len_q + 1
        del past, past_data, cur_len
        torch.cuda.empty_cache()
    out = dict(tag=args.tag, dataset=args.dataset, n_cycles=n_cyc,
               n_match=n_match,
               equivalent=bool(n_cyc > 0 and n_match == n_cyc))
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(
        rd, "tables", f"replay_equiv_{args.tag}.json"), "w"), indent=1)
    print(f"[replay] {args.tag}: {n_match}/{n_cyc} cycles match "
          f"inline lockstep -> equivalent={out['equivalent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
