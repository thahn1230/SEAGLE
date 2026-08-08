"""Train context-only rotation R_C (§18-19). Frozen: target, draft weights,
MP3 scales. Trainable: R_C only (orthogonal, Cayley parametrization).

Data: cycle-replay on CALIB cycles (cyc__*gsm8kcalib*.jsonl from a deployed
quantized arm). For each recorded cycle the target hidden trajectory is
teacher-forced (no grad), then the RCDraft forward (SAME code as deployment,
Gate H) predicts the block; loss is position-weighted CE/KL against the
recorded verifier posterior (L0, gamma=5) or the adaptive TV/KL hybrid (L1).

Draft ctx KV cache across cycles: past ctx K/V recomputed with the CURRENT
R_C but detached (truncated backprop); the current cycle carries gradient.
Simplification: each cycle is trained as an independent forward with the
full ctx feature set accumulated so far (mirrors deployment exactly).

Output: rc_ckpt {R_C, meta} + training log.
"""
import argparse
import json
import math
import os
import random

import torch
from torch import nn

from . import spinquant_target as sq
from . import interfaces
from .rc import RCDraft

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]


def build_drafts(mp3, rbin, draft_mode, device, fc_p2=False):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from . import draft_quant
    orig = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(device).eval()
    R1 = sq.load_rbin(rbin)["R1"] if rbin else torch.eye(4096)
    base = interfaces.fold_wc(orig, R1, mp3_scales=mp3)
    if draft_mode != "fp16":
        bits = {"w8a8": (8, 8), "w4a4": (4, 4)}[draft_mode]
        base = draft_quant.quantize_draft(
            base, w_bits=bits[0], a_bits=bits[1],
            fc_branch_dims=[4096] * 5 if fc_p2 else None)
    return base.to(device), orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", required=True)
    ap.add_argument("--target-mode", default="w4a4")
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--mp3-scales", default=None)
    ap.add_argument("--draft-mode", default="w4a4")
    ap.add_argument("--fc-p2", action="store_true")
    ap.add_argument("--objective", default="L0", choices=["L0", "L1"])
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    mp3 = [float(x) for x in args.mp3_scales.split(",")] \
        if args.mp3_scales else None
    target = sq.build_target(MODEL, args.target_mode, rbin_path=args.rbin,
                             device=dev)
    for p in target.parameters():
        p.requires_grad_(False)
    base, orig = build_drafts(mp3, args.rbin, args.draft_mode, dev,
                              fc_p2=args.fc_p2)
    for p in base.parameters():
        p.requires_grad_(False)
    rcd = RCDraft(base, orig, w_bits=4 if args.draft_mode == "w4a4" else 8,
                  a_bits=4 if args.draft_mode == "w4a4" else 8).to(dev)
    del orig

    lin = nn.Linear(4096, 4096, bias=False).to(dev)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4096))
    lin = nn.utils.parametrizations.orthogonal(lin, orthogonal_map="cayley")
    rcd._rc_param = lin
    opt = torch.optim.Adam(lin.parameters(), lr=args.lr)

    turns, order = {}, []
    for ln in open(args.cycles):
        r = json.loads(ln)
        key = (r["prompt_id"], r["turn"])
        if r.get("type") == "turn_header":
            turns[key] = {"ids": r["input_ids"], "cycles": []}
            order.append(key)
        else:
            turns[key]["cycles"].append(r)
    random.shuffle(order)

    R1 = sq.load_rbin(args.rbin)["R1"].float().to(dev) if args.rbin else None
    step, acc, losses = 0, 0, []
    log = []
    while step < args.steps:
        for key in order:
            if step >= args.steps:
                break
            t = turns[key]
            ids = torch.tensor([t["ids"]], device=dev)
            with torch.no_grad():
                # deployed-trajectory teacher forcing on the QUANTIZED target
                # deployed sequence = prompt + block[:tau] per cycle (each
                # cycle's block[0] IS the previous cycle's bonus token)
                traj = [ids[0].tolist()]
                for c in t["cycles"]:
                    traj.append(c["block"][:c["tau"]])
                flat = [x for seg in traj for x in seg]
                full = torch.tensor([flat], device=dev)
                o = target(full, output_hidden_states=True, use_cache=False)
                Hcat = torch.cat([o.hidden_states[l + 1][0] for l in SRC],
                                 dim=-1)                      # [T, 20480]
                del o
            if mp3 is not None:
                sc = torch.tensor(mp3, device=dev).repeat_interleave(4096)
                Hfeed = (Hcat * sc).to(torch.bfloat16)
            else:
                Hfeed = Hcat.to(torch.bfloat16)
            n_cyc = min(len(t["cycles"]), 6)
            for c in random.sample(t["cycles"], n_cyc):
                pl = c["prefix_len"]
                th = Hfeed[:pl].unsqueeze(0)
                block = torch.tensor([c["block"]], device=dev)
                ne = target.model.embed_tokens(block)
                pos = torch.arange(pl + block.shape[1],
                                   device=dev).unsqueeze(0)
                out = rcd(target_hidden=th, noise_embedding=ne,
                          position_ids=pos, use_cache=False,
                          is_causal=False)
                logits = target.lm_head(out[:, :-1, :]).float()
                labels = torch.tensor([c["posterior"][:-1]], device=dev)
                B = block.shape[1]
                w = torch.exp(-torch.arange(B - 1, device=dev).float()
                              / args.gamma)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), reduction="none").reshape(1, -1)
                if args.objective == "L0":
                    loss = (ce * w).sum() / w.sum()
                else:
                    p = torch.softmax(logits, -1)
                    tv = 1 - p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    loss = ((0.5 * ce + 0.5 * tv) * w).sum() / w.sum()
                (loss / args.accum).backward()
                losses.append(loss.item())
                acc += 1
                if acc % args.accum == 0:
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    if step % 10 == 0:
                        R = rcd.rc_matrix()
                        orth = (R @ R.t() - torch.eye(
                            4096, device=R.device)).abs().max().item()
                        m = sum(losses[-40:]) / len(losses[-40:])
                        rec = {"step": step, "loss": round(m, 4),
                               "orth_err": orth,
                               "dev_from_I": (R - torch.eye(
                                   4096, device=R.device)).norm().item()}
                        log.append(rec)
                        print(rec, flush=True)
                    if step >= args.steps:
                        break

    R = rcd.rc_matrix().detach().cpu()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"R_C": R, "meta": vars(args), "log": log}, args.out)
    orth = (R @ R.t() - torch.eye(4096)).abs().max().item()
    print(f"[train_rc] DONE orth_err={orth:.2e} -> {args.out}")
    assert orth < 1e-3, "Gate G violation"


if __name__ == "__main__":
    main()
