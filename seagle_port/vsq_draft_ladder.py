"""D-P ladder (§8 of the VSQ plan): draft quantization quality on held-out
DFlash-native block CE (SpecForge-parity forward). NOT causal-LM PPL —
labeled block CE / "block pseudo-PPL" only.

Arms (same held-out rows as rotation training validation):
  D-P0 fp16 draft            (bits 16, R=I)
  D-P1 W4A4 no rotation      (bits 4,  R=I)
  D-P2 W4A4 random rotation  (bits 4,  R=random orth, R2=random)
  D-P3 W4A4 learned R1_D     (bits 4,  R1Donly ckpt)
  D-P4 W4A4 learned R1_D+R2_D(bits 4,  best ckpt)
Also reports target-vs-draft KL and top-1 agreement vs the FP16 draft arm.
Writes tables/draft_quality.csv.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .vsq_draft_rot import RotQuantDraft
from . import vsq_specforge_port as sf
from .vsq_train_draft_rot import load_shared_modules

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"


def orth(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g,
                                       dtype=torch.float64))
    return (q * torch.sign(torch.diag(r))).float()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--fold-fc-rbin", required=True)
    ap.add_argument("--best-ckpt", required=True)
    ap.add_argument("--r1only-ckpt", required=True)
    ap.add_argument("--val-rows", type=int, default=15)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    base = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1T = sq.load_rbin(args.fold_fc_rbin)["R1"]
    base = interfaces.fold_wc(base, R1T)
    embed, head = load_shared_modules(dev)
    files = sorted(f for f in os.listdir(args.cache_dir)
                   if f.endswith(".npz"))[-args.val_rows:]

    def evaluate(rq, ref_logits=None):
        tot, accs, kls, agrees, n = 0.0, [], [], [], 0
        outs = []
        for fi, f in enumerate(files):
            z = np.load(os.path.join(args.cache_dir, f))
            ids = torch.tensor(z["input_ids"], device=dev).unsqueeze(0)
            H = torch.tensor(z["hidden"], device=dev,
                             dtype=torch.bfloat16).unsqueeze(0)
            lm = torch.zeros(1, ids.shape[1], device=dev)
            lm[:, int(z["gen_start"]):] = 1.0
            torch.manual_seed(1000 + fi)      # same anchors across arms
            loss, acc, aux = sf.specforge_block_forward(
                rq, embed, head, ids, H, lm, num_anchors=24,
                block_size=rq.block_size, mask_token_id=rq.mask_token_id,
                gamma=5.0)
            tot += loss.item()
            accs.append(acc.item())
            lg = aux["logits"]
            outs.append((lg.float().cpu(), aux["valid"].cpu()))
        ce = tot / len(files)
        row = {"block_ce": round(ce, 4),
               "block_pseudo_ppl": round(float(np.exp(ce)), 3),
               "top1_acc": round(float(np.mean(accs)), 4)}
        if ref_logits is not None:
            for (lg, v), (rlg, rv) in zip(outs, ref_logits):
                p = torch.softmax(rlg, -1)
                q = torch.log_softmax(lg, -1)
                kl = (p * (torch.log_softmax(rlg, -1) - q)).sum(-1)
                kls.append(kl[v & rv].mean().item())
                agrees.append(((lg.argmax(-1) == rlg.argmax(-1))[v & rv])
                              .float().mean().item())
            row["kl_vs_fp16draft"] = round(float(np.mean(kls)), 4)
            row["top1_agree_vs_fp16draft"] = round(float(np.mean(agrees)),
                                                   4)
        return row, outs

    rows = []

    def arm(name, rq, ref=None):
        r, outs = evaluate(rq, ref)
        r["arm"] = name
        rows.append(r)
        print({"arm": name, **r}, flush=True)
        return outs

    rq = RotQuantDraft(base, w_bits=16, a_bits=16, use_r2=False,
                       train_rotations=False, device=dev)
    ref = arm("D-P0_fp16", rq)
    rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=False,
                       train_rotations=False, device=dev)
    arm("D-P1_w4a4_norot", rq, ref)
    R1r = orth(4096, 11).to(dev)
    R2r = [orth(128, 20 + i).to(dev) for i in range(5)]
    rq.R1 = lambda: R1r
    rq.cfg["use_r2"] = True
    rq.R2 = lambda i: R2r[i]
    arm("D-P2_w4a4_random", rq, ref)
    ck = torch.load(args.r1only_ckpt + ".best", map_location="cpu",
                    weights_only=False)
    rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=False,
                       train_rotations=False, device=dev)
    R1l = ck["R1_D"].to(dev)
    rq.R1 = lambda: R1l
    arm("D-P3_w4a4_R1D", rq, ref)
    ck = torch.load(args.best_ckpt + ".best", map_location="cpu",
                    weights_only=False)
    rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=True,
                       train_rotations=False, device=dev)
    R1b = ck["R1_D"].to(dev)
    R2b = [t.to(dev) for t in ck["R2_D"]]
    rq.R1 = lambda: R1b
    rq.R2 = lambda i: R2b[i]
    arm("D-P4_w4a4_R1D_R2D", rq, ref)

    out = os.path.join(args.run_dir, "tables", "draft_quality.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ce = {r["arm"]: r["block_ce"] for r in rows}
    ok = ce["D-P4_w4a4_R1D_R2D"] < ce["D-P1_w4a4_norot"] and \
        ce["D-P4_w4a4_R1D_R2D"] < ce["D-P2_w4a4_random"]
    print(f"GATE E: {'PASS' if ok else 'FAIL'} ({ce})")


if __name__ == "__main__":
    main()
