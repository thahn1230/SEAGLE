#!/usr/bin/env python
"""SRT_W4A4_SQ / SRT_W8A8_SQ: learn the interface-boundary rotation
R_D for the STRICT natively-trained draft.

Geometry-preserving parameterization (validated family; the old
FullRotation+QR-retraction setup is forbidden):

    R_D = R_init @ Cayley(A),  A = W - W^T (skew by construction)
    R_init = normalized Hadamard-4096 (the SRT_W4A4_HAD solution)

R_D enters EXACTLY where the HAD arm's fixed rotation does (fc h-half:
runtime input rotation + W_h@R fold — the quantizers see rotated
act/weight; function preserved in exact arithmetic). AR sites are
frozen at their deployment RTN fake-quant (R-independent); down_proj
keeps the canonical R4 online Hadamard. EAGLE-specific deviation from
official SpinQuant (which rotates a full residual stream) is by
architecture: the strict draft's only rotatable quantization boundary
shared by first+recurrent paths is the fc h-half. Documented per spec
§23.

Objective (self-distillation, LK-style): SmoothL1(f_q, f_fp16) +
0.1 * softCE(head(f_fp16), head(f_q)) on cached native features
(train-split convs only; c4-calib AL used ONLY for checkpoint
selection downstream). Optimizer AdamW(0.9,0.95), cosine, 100-step
warmup, clip 0.5 (canonical LK recipe).

Usage: CUDA_VISIBLE_DEVICES=k python scripts/strict_rt/train_native_rd.py \
  --out <run>/rotations/RD_NATIVE_<mode>_s<seed>.pt --mode fake_w4a4 \
  --steps 1000 --seed 0
"""
import argparse, json, math, os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "strict_rt"))

import torch
import torch.nn.functional as F

import train_eagle1_strict_rt as srt
from eagle_spinquant.native_raw_draft import hadamard_4096, D
from eagle_spinquant.concat_selective_projection import QUANT_BITS

CKPT = "/data/thahn1230/strict_rt_ckpts/strict_rt_final.pt"


def cayley(A):
    n = A.shape[0]
    In = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(In - A / 2, In + A / 2)


def w_fake_sym_ste(W, bits):
    """per-out-channel symmetric RTN (no clip search in-graph; the
    deployed MSE clip is applied at EXPORT time — training uses plain
    sym RTN as the differentiable surrogate, deviation recorded)."""
    maxq = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / maxq
    q = (W / s).round().clamp(-maxq - 1, maxq) * s
    return W + (q - W).detach()


def a_fake_asym_ste(x, bits):
    maxq = 2 ** bits - 1
    xmin = x.amin(dim=-1, keepdim=True).clamp_max(0)
    xmax = x.amax(dim=-1, keepdim=True).clamp_min(0)
    s = ((xmax - xmin).clamp_min(1e-8)) / maxq
    z = (-xmin / s).round()
    q = ((x / s + z).round().clamp(0, maxq) - z) * s
    return x + (q - x).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fake_w4a4",
                    choices=["fake_w4a4", "fake_w8a8"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--cache-dir",
                    default="/data/thahn1230/strict_rt_cache")
    ap.add_argument("--save-every", type=int, default=250)
    args = ap.parse_args()
    dev = "cuda:0"
    torch.manual_seed(args.seed)
    w_bits, a_bits = QUANT_BITS[args.mode]

    blob = torch.load(srt.TOK_CACHE, weights_only=False)
    rows, split = blob["rows"], blob["split"]
    teacher = srt.CachedTeacher(args.cache_dir, dev)
    ids_pool = [i for i in range(split) if i in teacher]  # train-only

    # FP16 reference draft (frozen)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref, _ = srt.build_draft(dev, args.seed)
    ref.load_state_dict(ck["model"])
    ref = ref.half().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.gradient_checkpointing = False

    # frozen fp32 master of fc h-half + quantized-frozen everything else
    Wfc = ref.fc.weight.data.float()          # [4096, 8192]
    W_e, W_h = Wfc[:, :D].contiguous(), Wfc[:, D:].contiguous()
    bias = ref.fc.bias.data.float()

    H = hadamard_4096(dev, torch.float32)
    A_p = torch.zeros(D, D, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([A_p], lr=args.lr, betas=(0.9, 0.95))
    warm = 100

    def lr_at(t):
        if t < warm:
            return args.lr * t / warm
        p = (t - warm) / max(args.steps - warm, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    hobj = torch.load(os.path.join(args.cache_dir, "native_head.pt"),
                      map_location="cpu", weights_only=False)
    headW = hobj["weight"].to(dev).half()

    # quantized-forward pieces: reuse the REF draft for everything
    # after fc (AR sites frozen at RTN fake-quant = deployment-exact)
    from eagle_spinquant.native_raw_draft import apply_native_raw_quant
    qdraft, _ = srt.build_draft(dev, args.seed)
    qdraft.load_state_dict(ck["model"])
    qdraft = qdraft.half().eval()
    for p in qdraft.parameters():
        p.requires_grad_(False)
    qdraft.gradient_checkpointing = False
    apply_native_raw_quant(qdraft, args.mode,
                           site_mask=[s for s in
                                      ("q_proj", "k_proj", "v_proj",
                                       "o_proj", "gate_proj", "up_proj",
                                       "down_proj")])

    class DiffRotFc(torch.nn.Module):
        """Differentiable replica of RotatedFcInput+FakeW4A4Linear:
        h-half rotated by the CURRENT R, then ONE per-token asym act
        quant over the full 8192-dim input (matching the deployed
        single-scale contract), weight = [W_e | W_h@R] sym-RTN STE."""

        def __init__(self):
            super().__init__()
            self.R = None
            self.weight = torch.nn.Parameter(  # device shim for cnets
                torch.zeros(1, device=dev), requires_grad=False)

        def forward(self, z):
            e, hh = z[..., :D].float(), z[..., D:].float()
            zc = torch.cat([e, hh @ self.R], dim=-1)
            zq = a_fake_asym_ste(zc, a_bits) if a_bits < 16 else zc
            Wq = w_fake_sym_ste(torch.cat([W_e, W_h @ self.R], dim=1),
                                w_bits) if w_bits < 16 else \
                torch.cat([W_e, W_h @ self.R], dim=1)
            return F.linear(zq, Wq, bias).half()

    dfc = DiffRotFc()
    qdraft.fc = dfc

    g = torch.Generator().manual_seed(args.seed * 7919)
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logp = args.out + ".log.jsonl"
    lf = open(logp, "a")
    for step in range(1, args.steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        idxs = [ids_pool[int(torch.randint(len(ids_pool), (1,),
                                           generator=g))]
                for _ in range(args.bs)]
        opt.zero_grad(set_to_none=True)
        loss_acc = v_acc = p_acc = 0.0
        for i in idxs:
            r = rows[i]
            ids = r["input_ids"].long()[None].to(dev)
            h = teacher.get(i).half()[None]          # (1,T,4096)
            L = min(h.shape[1] - 1, ids.shape[1] - 1)
            hin = h[:, :L]
            iin = ids[:, 1:1 + L]
            am = torch.ones(1, L, dtype=torch.long, device=dev)
            with torch.no_grad():
                f_ref = ref(hin, input_ids=iin, attention_mask=am)
            Amat = A_p - A_p.T
            dfc.R = (H @ cayley(Amat)).to(torch.float32)
            f_q = qdraft(hin, input_ids=iin, attention_mask=am)
            vloss = F.smooth_l1_loss(f_q.float(), f_ref.float())
            with torch.no_grad():
                tp = torch.softmax(
                    F.linear(f_ref.half(), headW).float(), -1)
            lq = torch.log_softmax(
                F.linear(f_q.half(), headW).float(), -1)
            ploss = -(tp * lq).sum(-1).mean()
            loss = vloss + 0.1 * ploss
            (loss / args.bs).backward()
            loss_acc += float(loss) / args.bs
            v_acc += float(vloss) / args.bs
            p_acc += float(ploss) / args.bs
        torch.nn.utils.clip_grad_norm_([A_p], 0.5)
        opt.step()
        if step % 25 == 0:
            rec = dict(step=step, loss=round(loss_acc, 5),
                       vloss=round(v_acc, 5), ploss=round(p_acc, 5),
                       lr=lr_at(step),
                       mins=round((time.time() - t0) / 60, 1))
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
            if step % 100 == 0:
                print(f"[rd] {rec}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            with torch.no_grad():
                Rf = (H.double() @ cayley(
                    (A_p - A_p.T).double())).cpu()
            torch.save(dict(R_D=Rf, step=step, seed=args.seed,
                            mode=args.mode,
                            meta=dict(init="hadamard_4096",
                                      param="cayley_residual",
                                      objective="LK self-distill",
                                      lr=args.lr, bs=args.bs)),
                       f"{args.out}.step{step:04d}.pt")
    torch.save(dict(R_D=Rf, step=args.steps, seed=args.seed,
                    mode=args.mode), args.out)
    print(f"[rd] DONE {args.out} {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
