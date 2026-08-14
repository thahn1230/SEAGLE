"""R1DCE §16 — C9: ONE shared draft+context rotation R1_DC.

Construction: R1_DC = R1_D_frozen @ D with D a trainable identity-init
Cayley-orthogonal Linear (guaranteed warm start at exactly R1_D; avoids the
Cayley right-inverse assignment failure mode).  The SAME tensor drives BOTH
the draft residual basis (rq.R1) and the context rotation
(rq.rc_matrix_buf, assigned as a live graph tensor each forward), so one
matrix serves both interfaces.  R2_D frozen (§18).  Draft/target weights,
W_c, R1_T frozen.

Dual objective (§16):
    L = lambda_draft * L_DFlash_native(specforge_block_forward, W4A4,
        gamma=5, exact validated semantics)
      + lambda_ctx   * L_ctxKV_W4A4  (FP-path H_t reconstruction NMSE,
        same objective as the C8 trainer)

Checkpoint selection: best combined held-out val (last --val-rows rows).
Saves {"R1_D": R1_DC, "R2_D": frozen, "R_C": R1_DC (self), "D": ...} so
eval_al can consume it as BOTH --vsq-draft ckpt (.best) and --vsq-rc file.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn

from . import interfaces
from . import spinquant_target as sq
from .vsq_draft_rot import RotQuantDraft
from . import vsq_specforge_port as sf
from .vsq_train_draft_rot import load_shared_modules
from .r1dce_train_deltar import ctx_loss, rms_bare

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=f"{VSQ_RD}/hcache_w4a4_s1")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--lambda-draft", type=float, default=1.0)
    ap.add_argument("--lambda-ctx", type=float, default=1.0)
    ap.add_argument("--val-rows", type=int, default=15)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
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
    R1T = sq.load_rbin(S1RBIN)["R1"]
    base = interfaces.fold_wc(base, R1T)
    rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=True,
                       train_rotations=False, device=dev)
    del base
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1D = ck["R1_D"].float().to(dev)
    R2D = [t.float().to(dev) for t in ck["R2_D"]]
    rq.R2 = lambda i: R2D[i]

    lin = nn.Linear(4096, 4096, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4096))
    D = nn.utils.parametrizations.orthogonal(
        lin, orthogonal_map="cayley").to(dev)
    rq.R1 = lambda: R1D @ D.weight          # shared matrix, live graph

    embed, head = load_shared_modules(dev)
    for p in list(embed.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    # frozen FP ctx views for L_ctx (same convention as the C8 trainer)
    views = []
    for i in range(rq.n_layers):
        wk = getattr(rq, f"wk_ctx_{i}").float().detach()
        wv = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                          R2D[i], "out").detach()
        views.append((wk, wv))
    fc_w = rq.fc_w.detach()

    files = sorted(f for f in os.listdir(args.cache_dir)
                   if f.endswith(".npz"))
    val_files = files[-args.val_rows:]
    train_files = files[:-args.val_rows]
    rng = random.Random(args.seed)
    opt = torch.optim.Adam(D.parameters(), lr=args.lr)

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
            R = rq.R1()
            rq.rc_matrix_buf = R                # SHARED ctx rotation (live)
            l_draft, acc, _ = sf.specforge_block_forward(
                rq, embed, head, ids, H.to(torch.bfloat16), lm,
                num_anchors=n_anchors, block_size=rq.block_size,
                mask_token_id=rq.mask_token_id, gamma=5.0, generator=None)
            Ht = rms_bare(H[0] @ fc_w.t())
            l_ctx = ctx_loss(Ht, R, views)
            loss = args.lambda_draft * l_draft + args.lambda_ctx * l_ctx
        return loss, l_draft.detach(), l_ctx.detach(), acc

    @torch.no_grad()
    def validate():
        ce = cx = ac = 0.0
        for f in val_files:
            _, ld, lc, a = row_forward(f, 24, grad=False)
            ce += ld.item()
            cx += lc.item()
            ac += a.item()
        n = len(val_files)
        return ce / n, cx / n, ac / n

    ce0, cx0, ac0 = validate()
    comb0 = args.lambda_draft * ce0 + args.lambda_ctx * cx0
    print(f"[c9 l{args.lambda_ctx}] init val: draftCE={ce0:.4f} "
          f"ctxNMSE={cx0:.5f} acc={ac0:.4f} comb={comb0:.4f}", flush=True)
    best = (comb0, 0, ce0, cx0)
    log = [{"step": 0, "val_ce": ce0, "val_ctx": cx0, "val_acc": ac0}]

    def save():
        R = (R1D @ D.weight.detach()).cpu().float()
        torch.save({"R1_D": R, "R2_D": [t.cpu() for t in R2D],
                    "R_C": R, "D": D.weight.detach().cpu().float(),
                    "meta": vars(args), "log": log},
                   args.out + ".best")

    save()
    for step in range(1, args.steps + 1):
        f = rng.choice(train_files)
        loss, ld, lc, acc = row_forward(f, args.anchors, grad=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            dist = (torch.linalg.norm(
                D.weight.detach() - torch.eye(4096, device=dev),
                ord="fro") / 64).item()
            print(f"[c9] step {step:4d} draftCE={ld.item():.4f} "
                  f"ctxNMSE={lc.item():.5f} ||D-I||/sqrt(d)={dist:.4f}",
                  flush=True)
        if step % args.val_every == 0:
            ce, cx, ac = validate()
            comb = args.lambda_draft * ce + args.lambda_ctx * cx
            log.append({"step": step, "val_ce": ce, "val_ctx": cx,
                        "val_acc": ac})
            mark = ""
            if comb < best[0]:
                best = (comb, step, ce, cx)
                save()
                mark = " *best"
            print(f"[c9 val] step {step} ce={ce:.4f} ctx={cx:.5f} "
                  f"acc={ac:.4f} comb={comb:.4f}{mark}", flush=True)

    summ = {"init": {"ce": ce0, "ctx": cx0, "acc": ac0},
            "best_comb": best[0], "best_step": best[1],
            "best_ce": best[2], "best_ctx": best[3],
            "lambda_draft": args.lambda_draft,
            "lambda_ctx": args.lambda_ctx,
            "R1DC_dist_from_R1D":
                (torch.linalg.norm(
                    (R1D @ D.weight.detach()) - R1D, ord="fro") / 64).item()}
    json.dump(summ, open(args.out + ".summary.json", "w"), indent=1)
    print(f"[c9] DONE {summ}")


if __name__ == "__main__":
    main()
