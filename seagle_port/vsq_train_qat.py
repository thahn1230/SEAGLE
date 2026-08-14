"""Q-family QAT (§9-§12 of the update): SpecForge-semantics DFlash W4A4 QAT
(upstream SpecForge has NO QAT — this is our labeled extension).

Arms (identical trainer/data/steps/selection; only init differs):
  Q2 generic : R1_D = I, R2_D = I         (no rotations)
  Q3 vsq-init: learned R1_D/R2_D frozen   (M3 basis)
  Q5 seagle  : Q3 + R_C = R1_T ctx path   (M5 basis)
Trainable: draft weights only (gamma-fused views incl. ctx K/V views + fc).
Rotations always frozen. Same corpus/cache as rotation training; val =
same held-out rows; best-val checkpoint. LR from a shared pilot grid.
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
from .vsq_train_draft_rot import load_shared_modules

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"


def build(arm, rbin, rot_ckpt, device, rc_path=None):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    base = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1T = sq.load_rbin(rbin)["R1"]
    base = interfaces.fold_wc(base, R1T)
    rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=(arm != "Q2"),
                       train_rotations=False, device=device,
                       train_weights=True)
    if arm in ("Q3", "Q5"):
        ck = torch.load(rot_ckpt + ".best", map_location="cpu",
                        weights_only=False)
        R1b = ck["R1_D"].to(device)
        R2b = [t.to(device) for t in ck["R2_D"]]
        rq.R1 = lambda: R1b
        rq.R2 = lambda i: R2b[i]
    if arm == "Q5":
        # R1DCE: --rc-path overrides the ctx rotation (default R1_T reuse)
        if rc_path:
            Rc = torch.load(rc_path, map_location="cpu",
                            weights_only=False)["R_C"]
            rq.rc_matrix_buf = Rc.float().to(device)
        else:
            rq.rc_matrix_buf = R1T.float().to(device)
    for p in rq.r1.parameters():
        p.requires_grad_(False)
    for m in rq.r2:
        for p in m.parameters():
            p.requires_grad_(False)
    return rq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["Q2", "Q3", "Q5"])
    ap.add_argument("--fc-p2", action="store_true",
                    help="Q6 supplementary: train WITH per-branch fc "
                         "activation scales (closes the M6 eval-composition "
                         "caveat)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--rot-ckpt", required=True)
    ap.add_argument("--rc-path", default=None,
                    help="R1DCE: ctx rotation ckpt {'R_C':...} for Q5 "
                         "instead of R1_T reuse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--val-rows", type=int, default=15)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rq = build(args.arm, args.rbin, args.rot_ckpt, dev,
               rc_path=args.rc_path)
    if args.fc_p2:
        rq.cfg["fc_p2"] = True
    embed, head = load_shared_modules(dev)
    for p in list(embed.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    files = sorted(f for f in os.listdir(args.cache_dir)
                   if f.endswith(".npz"))
    val_files, train_files = files[-args.val_rows:], files[:-args.val_rows]
    rng = random.Random(args.seed)
    params = [p for p in rq.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[qat {args.arm}] trainable {n_tr/1e6:.1f}M params lr {args.lr}")
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    def row_forward(f, n_anchors, grad=True):
        z = np.load(os.path.join(args.cache_dir, f))
        ids = torch.tensor(z["input_ids"], device=dev).unsqueeze(0)
        H = torch.tensor(z["hidden"], device=dev,
                         dtype=torch.bfloat16).unsqueeze(0)
        lm = torch.zeros(1, ids.shape[1], device=dev)
        lm[:, int(z["gen_start"]):] = 1.0
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return sf.specforge_block_forward(
                rq, embed, head, ids, H, lm, num_anchors=n_anchors,
                block_size=rq.block_size, mask_token_id=rq.mask_token_id,
                gamma=5.0)[:2]

    def validate():
        tot = 0.0
        for f in val_files:
            l, _ = row_forward(f, 16, grad=False)
            tot += l.item()
        return tot / len(val_files)

    log = []
    v0 = validate()
    best = (v0, 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def save():
        torch.save({"state": {k: v.detach().to(torch.bfloat16).cpu()
                              for k, v in rq.state_dict().items()
                              if not k.startswith(("r1.", "r2."))},
                    "meta": vars(args), "log": log}, args.out + ".best")

    save()
    print(f"[init] val_ce {v0:.4f}", flush=True)
    for step in range(1, args.steps + 1):
        f = rng.choice(train_files)
        loss, acc = row_forward(f, args.anchors, grad=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0:
            log.append({"step": step, "loss": round(loss.item(), 4),
                        "acc": round(acc.item(), 4)})
            print(log[-1], flush=True)
        if step % args.val_every == 0:
            v = validate()
            log.append({"step": step, "val_ce": round(v, 4)})
            print(f"[val] {step} ce {v:.4f}", flush=True)
            if v < best[0]:
                best = (v, step)
                save()
    json.dump({"arm": args.arm, "lr": args.lr, "init_val_ce": v0,
               "best_val_ce": best[0], "best_step": best[1]},
              open(args.out + ".summary.json", "w"), indent=1)
    print(f"[qat {args.arm}] DONE init {v0:.4f} best {best[0]:.4f} "
          f"@ {best[1]}")


if __name__ == "__main__":
    main()
