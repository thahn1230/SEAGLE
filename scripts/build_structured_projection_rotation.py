#!/usr/bin/env python
"""Build / optimize structured projection rotations (spec §5-§6, §14).

Subcommands:
  dump      write metadata (hashes, block, seed, order) for a fixed
            StructuredRotation spec
  optimize  R6: calibration-optimized structured rotation. Frozen
            model weights; learns ONLY a blockwise-Cayley orthogonal
            refinement Q2 composed AFTER the best fixed cross rotation
            (Q = Q_fixed Q2, still orthogonal by construction; Cayley
            Q_b = (I-A)(I+A)^{-1} per 64-ch block, A skew).
            Objective: deployed-quantizer STE projection-output NMSE
            on the held-out calibration tensors. 3 seeds.
            This is calibration-optimized PTQ, NOT calibration-free.
"""
import argparse, json, math, os, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 scale_vec)

D = 4096


class BlockCayley(torch.nn.Module):
    def __init__(self, n=8192, block=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n, self.block = n, block
        self.nb = n // block
        self.a = torch.nn.Parameter(
            0.01 * torch.randn(self.nb, block, block))

    def Q(self):
        A = self.a - self.a.transpose(1, 2)          # skew
        eye = torch.eye(self.block, device=A.device,
                        dtype=A.dtype).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)  # (I+A)^-1 (I-A)

    def apply(self, x):
        Q = self.Q().to(x.dtype)
        y = x.reshape(*x.shape[:-1], self.nb, self.block)
        y = torch.einsum("...bi,bij->...bj", y, Q)
        return y.reshape(*x.shape)


def ste_wq(w, fq):
    with torch.no_grad():
        wq = fq._weight_fake_quant(w.detach().half(), 4).float()
    return w + (wq - w).detach()


def ste_aq(x, fq):
    with torch.no_grad():
        aq = fq._act_quantizer(4)
        aq.find_params(x.detach().half())
        xq = aq(x.detach().half()).float()
        aq.free()
    return x + (xq - x).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["dump", "optimize"])
    ap.add_argument("--spec", default=None, help="JSON rotation spec")
    ap.add_argument("--path", default="first",
                    choices=["first", "rec"])
    ap.add_argument("--beta", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    if args.mode == "dump":
        rot = StructuredRotation(json.loads(args.spec))
        out = rot.meta()
        os.makedirs(os.path.join(rd, "candidates"), exist_ok=True)
        p = os.path.join(rd, "candidates",
                         f"rotation_meta_{args.path}.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out))
        return 0

    dev = "cuda:0"
    from eagle_spinquant import fake_w4a4_draft as fq
    tens = torch.load(os.path.join(rd, "tensors", "calib_int4.pt"),
                      map_location="cpu", weights_only=False)
    key = "X_first" if args.path == "first" else "X_rec_all"
    X = tens[key].float().to(dev)
    W0 = tens["W"].float().to(dev)
    bias = tens.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    fixed = StructuredRotation(json.loads(args.spec), device=dev)
    m = float(D ** args.beta)
    s = scale_vec(2 * D, m, dev)
    Xs = fixed.apply(X * s)                  # fixed part precomputed
    Wf = fixed.apply(W0 / s)
    Yref = X @ W0.t() + (b_t if b_t is not None else 0)

    cay = BlockCayley(seed=args.seed).to(dev)
    opt = torch.optim.Adam(cay.parameters(), lr=args.lr)
    t0 = time.time()
    j0 = None
    for it in range(args.steps):
        opt.zero_grad()
        Xt = cay.apply(Xs)
        Wt = cay.apply(Wf)
        Y = ste_aq(Xt, fq) @ ste_wq(Wt, fq).t()
        if b_t is not None:
            Y = Y + b_t
        loss = ((Y - Yref) ** 2).sum() / (Yref ** 2).sum()
        if j0 is None:
            j0 = float(loss)
        loss.backward()
        opt.step()
        if it % 25 == 0:
            print(f"[cayley {args.path} s{args.seed}] it{it} "
                  f"J={float(loss):.4f}", flush=True)
    with torch.no_grad():
        Q = cay.Q()
        orth = float((torch.einsum("bij,bkj->bik", Q, Q)
                      - torch.eye(cay.block, device=dev)
                      ).norm() / math.sqrt(Q.numel()))
        Xt = cay.apply(Xs)
        Wt = cay.apply(Wf)
        Wq = fq._weight_fake_quant(Wt.half(), 4).float()
        aq = fq._act_quantizer(4)
        aq.find_params(Xt.half())
        Xq = aq(Xt.half()).float()
        aq.free()
        Y = Xq @ Wq.t() + (b_t if b_t is not None else 0)
        jf = float(((Y - Yref) ** 2).sum() / (Yref ** 2).sum())
    out = dict(path=args.path, seed=args.seed, base_spec=args.spec,
               beta=args.beta, steps=args.steps, lr=args.lr,
               optimizer="Adam", j_initial=j0, j_final=jf,
               orthogonality_error=orth,
               wall_seconds=round(time.time() - t0, 1),
               param_count=sum(p.numel()
                               for p in cay.parameters()),
               block=cay.block)
    os.makedirs(os.path.join(rd, "candidates"), exist_ok=True)
    torch.save(dict(a=cay.a.detach().cpu(), **out),
               os.path.join(rd, "candidates",
                            f"cayley_{args.path}_s{args.seed}.pt"))
    json.dump(out, open(os.path.join(
        rd, "candidates",
        f"cayley_{args.path}_s{args.seed}.json"), "w"), indent=1)
    print(f"[cayley] {args.path} s{args.seed}: J {j0:.4f} -> "
          f"{jf:.4f} orth {orth:.2e} "
          f"({out['wall_seconds']}s, {out['param_count']} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
