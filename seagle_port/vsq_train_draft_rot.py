"""Draft SpinQuant rotation optimization (Operation A, §9/§13 of update).

Frozen draft weights; trainable R1_D (+R2_D) of RotQuantDraft; SpecForge-
parity block forward + dflash loss (γ=5). Teacher hidden = cached DEPLOYED
target interface (H1). Shared embed/head loaded in their ORIGINAL basis:
with input boundary e@R1_D and output boundary norm@D_γf R1_D^T this is
function-identical to the deployed rotated-target composition (the
R1_T-basis embed output times R1_T^T R1_D equals original-basis e times
R1_D). fc inside RotQuantDraft is pre-folded with R1_T (mandatory
VSQ-CORRECT interface fold) by the caller flag --fold-fc-rbin.

Checkpoint selection: best held-out block CE (last --val-rows rows),
evaluated every --val-every steps. Never sees the 4 AL test sets.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .vsq_draft_rot import RotQuantDraft
from . import vsq_specforge_port as sf

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"


def load_shared_modules(device):
    """Original-basis shared embed/lm_head (weights only, bf16)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    embed = torch.nn.Embedding.from_pretrained(
        m.model.embed_tokens.weight.data.clone()).to(device)
    head = torch.nn.Linear(4096, m.lm_head.weight.shape[0],
                           bias=False).to(torch.bfloat16)
    head.weight.data.copy_(m.lm_head.weight.data)
    head = head.to(device)
    del m
    return embed, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--fold-fc-rbin", required=True,
                    help="target R.bin whose R1 folds into fc (VSQ-CORRECT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--use-r2", action="store_true", default=True)
    ap.add_argument("--no-r2", dest="use_r2", action="store_false")
    ap.add_argument("--val-rows", type=int, default=15)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    base = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1T = sq.load_rbin(args.fold_fc_rbin)["R1"]
    base = interfaces.fold_wc(base, R1T)
    rq = RotQuantDraft(base, w_bits=args.w_bits, a_bits=args.a_bits,
                       use_r2=args.use_r2, train_rotations=True,
                       device=dev)
    del base
    embed, head = load_shared_modules(dev)
    for p in list(embed.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    files = sorted(os.listdir(args.cache_dir))
    files = [f for f in files if f.endswith(".npz")]
    val_files = files[-args.val_rows:]
    train_files = files[:-args.val_rows]
    rng = random.Random(args.seed)

    params = [p for p in rq.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)

    def row_forward(f, n_anchors, grad=True):
        z = np.load(os.path.join(args.cache_dir, f))
        ids = torch.tensor(z["input_ids"], device=dev).unsqueeze(0)
        H = torch.tensor(z["hidden"], device=dev,
                         dtype=torch.float32).unsqueeze(0)
        gs = int(z["gen_start"])
        lm = torch.zeros(1, ids.shape[1], device=dev)
        lm[:, gs:] = 1.0
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            loss, acc, _ = sf.specforge_block_forward(
                rq, embed, head, ids, H.to(torch.bfloat16), lm,
                num_anchors=n_anchors, block_size=rq.block_size,
                mask_token_id=rq.mask_token_id, gamma=5.0, generator=None)
        return loss, acc

    def validate():
        tot, accs = 0.0, []
        for f in val_files:
            l, a = row_forward(f, 24, grad=False)
            tot += l.item()
            accs.append(a.item())
        return tot / len(val_files), sum(accs) / len(accs)

    log = []
    v0, a0 = validate()
    print(f"[init] val_ce {v0:.4f} acc {a0:.4f}", flush=True)
    best = (v0, 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def save(tag):
        torch.save({"R1_D": rq.R1().detach().cpu(),
                    "R2_D": [rq.R2(i).detach().cpu() if rq.R2(i) is not None
                             else None for i in range(rq.n_layers)],
                    "meta": vars(args), "log": log},
                   args.out + (".best" if tag == "best" else ""))

    save("best")
    for step in range(1, args.steps + 1):
        f = rng.choice(train_files)
        loss, acc = row_forward(f, args.anchors, grad=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0:
            R = rq.R1()
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "acc": round(acc.item(), 4),
                   "dev_from_I": round((R - torch.eye(
                       4096, device=R.device)).norm().item(), 3)}
            log.append(rec)
            print(rec, flush=True)
        if step % args.val_every == 0:
            v, a = validate()
            log.append({"step": step, "val_ce": round(v, 4),
                        "val_acc": round(a, 4)})
            print(f"[val] step {step} ce {v:.4f} acc {a:.4f}", flush=True)
            if v < best[0]:
                best = (v, step)
                save("best")
    save("final")
    R = rq.R1().detach()
    orth = (R @ R.t() - torch.eye(4096, device=R.device)).abs().max().item()
    json.dump({"best_val_ce": best[0], "best_step": best[1],
               "init_val_ce": v0, "final_orth_err": orth},
              open(args.out + ".summary.json", "w"), indent=1)
    print(f"[train_draft_rot] DONE best_val {best[0]:.4f} @ {best[1]} "
          f"init {v0:.4f} orth {orth:.2e}")
    assert orth < 1e-3


if __name__ == "__main__":
    main()
