"""Phase 15 (§15): context-vs-draft K/V input distributions + A4 sensitivity.

Architecture fact (model.py L226-233): every draft layer applies the SAME
shared k_proj/v_proj to (a) the global fused context H_t (NO per-layer norm on
the ctx branch) and (b) input_layernorm_i(H_d) per layer. So the ctx input
distribution is one tensor; the draft input differs per layer.

For each draft layer i: RMS/absmax of both inputs, A4 NMSE of the input,
K_ctx/K_draft/V_ctx/V_draft output NMSE under A4-input + W4-weight quant,
and K1 proxy (ctx-separate A4 scale is automatic — per-token quant already
separates tokens; the real K1/K2 question is migration scale s_ctx), so we
also report the optimal scalar s_ctx per layer minimizing K_ctx NMSE.

Writes tables/ctx_kv_stats.csv.
"""
import argparse
import csv
import json
import os

import torch

from . import SPINQUANT_ROOT  # noqa: F401
from .wc_stats import calib_prompts, a4_fake, w4_fake
from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-prompts", type=int, default=12)
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, "fp16", device=dev)
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()

    caps = {i: {"ctx": [], "noise": []} for i in range(5)}
    counters = {}

    def mk_hook(li):
        def hook(mod, inp):
            k = counters.get(li, 0)
            counters[li] = k + 1
            branch = "ctx" if k % 2 == 0 else "noise"
            caps[li][branch].append(inp[0][0].float().cpu())
        return hook

    handles = [draft.layers[i].self_attn.k_proj.register_forward_pre_hook(
        mk_hook(i)) for i in range(5)]

    mask_id = draft.mask_token_id
    B = draft.block_size
    for p in calib_prompts(args.n_prompts):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=640).input_ids.to(dev)
        o = target(ids, output_hidden_states=True, use_cache=False)
        th = torch.cat([o.hidden_states[l + 1] for l in SRC], dim=-1)
        first = o.logits[:, -1:].argmax(-1)
        blk = torch.cat([first, torch.full((1, B - 1), mask_id,
                                           device=dev)], dim=1)
        ne = target.model.embed_tokens(blk)
        pos = torch.arange(ids.shape[1] + B, device=dev).unsqueeze(0)
        draft(target_hidden=th, noise_embedding=ne,
              position_ids=pos, use_cache=False, is_causal=False)
    for h in handles:
        h.remove()
    del target
    torch.cuda.empty_cache()

    rows = []
    for li in range(5):
        ctx = torch.cat(caps[li]["ctx"]).to(dev)
        noi = torch.cat(caps[li]["noise"]).to(dev)
        att = draft.layers[li].self_attn
        rec = {"layer": li,
               "ctx_rms": ctx.pow(2).mean().sqrt().item(),
               "ctx_absmax": ctx.abs().max().item(),
               "noise_rms": noi.pow(2).mean().sqrt().item(),
               "noise_absmax": noi.abs().max().item()}
        for pname, proj in (("k", att.k_proj), ("v", att.v_proj)):
            W = proj.weight.data.float()
            Wq, _ = w4_fake(W)
            for bname, x in (("ctx", ctx), ("noise", noi)):
                y = x @ W.t()
                xq, zr, _ = a4_fake(x)
                yq = xq @ Wq.t()
                rec[f"{pname}_{bname}_nmse"] = \
                    ((yq - y).pow(2).mean() / y.pow(2).mean()).item()
                if pname == "k":
                    rec[f"{bname}_a4_zero"] = zr
        # optimal scalar s_ctx (K2 proxy): grid over 2^k
        best = (1.0, rec["k_ctx_nmse"])
        W = att.k_proj.weight.data.float()
        y = ctx @ W.t()
        for s in [0.25, 0.5, 0.7, 1.4, 2.0, 4.0]:
            xq, _, _ = a4_fake(ctx * s)
            Wq2, _ = w4_fake(W / s)
            v = (((xq @ Wq2.t()) - y).pow(2).mean() /
                 y.pow(2).mean()).item()
            if v < best[1]:
                best = (s, v)
        rec["k_ctx_s_opt"], rec["k_ctx_nmse_s_opt"] = best
        rows.append(rec)
        print(rec, flush=True)

    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables", "ctx_kv_stats.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[ctx_kv] DONE")


if __name__ == "__main__":
    main()
