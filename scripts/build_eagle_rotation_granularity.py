#!/usr/bin/env python
"""Rotation-granularity sweep G0-G7 (study §10) on the held-out
projection calibration tensors with the deployment quantizers.

Granularities (all orthogonal by construction, identity-initialized
learned forms evaluated UNTRAINED here — this stage isolates the
STRUCTURAL capacity; trained variants come from the training grid):

  G0 identity | G1 branchwise dual (structured seed-best)
  G2 pairwise (e_i,h_i) Givens — evaluated at Hadamard-2 angle 45deg
  G3 cross-branch blocks b in {2..512} (structured FWHT, seed-best of 8)
  G4 channel-group-shared block rotation (one 32-block shared)
  G5 full structured 8192 | G6 Householder K in {1..64} (random)
  G7 butterfly (staged pairwise = log2 rounds of stride perms + G2)

Metric: deployment W4A4 projection-output NMSE (j_out), A4/W4 NMSE,
params, transform FLOPs/token. Writes tables/granularity_grid.json.
"""
import argparse, json, math, os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))
from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 LearnedRotation,
                                                 interleave_perm)
import importlib.util
_sp = importlib.util.spec_from_file_location(
    "cal", os.path.join(ROOT, "scripts", "calibrate_rotated_ep3p.py"))
cal = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(cal)

D = 4096


class Pairwise45:
    """G2: fixed 45-degree rotation of every (e_i, h_i) pair == block-2
    Hadamard on the interleaved layout (the smallest cross-branch mix)."""

    def __init__(self, device):
        self.inner = StructuredRotation(
            dict(family="cross", block=2, seed=0,
                 interleave_chunk=1), device=device)

    def apply(self, x):
        return self.inner.apply(x)


class Butterfly:
    """G7: log2(n) staged block-2 rotations with stride perms."""

    def __init__(self, stages, device):
        self.stages = stages
        self.rots = [StructuredRotation(
            dict(family="cross", block=2, seed=s,
                 interleave_chunk=1 << s), device=device)
            for s in range(stages)]

    def apply(self, x):
        for r in self.rots:
            x = r.apply(x)
        return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rep3p-run", default=None)
    args = ap.parse_args()
    rd = args.run_dir
    rep = args.rep3p_run or os.path.join(
        ROOT, open(os.path.join(ROOT, "runs",
                                "REP3P_RUN_DIR")).read().strip())
    dev = "cuda:0"
    from eagle_spinquant import fake_w4a4_draft as fq
    tens = torch.load(os.path.join(rep, "tensors", "calib_int4.pt"),
                      map_location="cpu", weights_only=False)
    W0 = tens["W"].float().to(dev)
    bias = tens.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    rows = []

    def flops(kind, b=None, K=None):
        n = 2 * D
        if kind == "identity":
            return 0
        if kind == "hh":
            return 4 * n * K
        if kind == "butterfly":
            return 2 * n * int(math.log2(n))
        return 2 * n * int(math.log2(b))     # blockwise FWHT adds

    def ev(name, rot, params, fl, path="first", beta=0.40):
        X = tens["X_first" if path == "first"
                 else "X_rec_all"].float().to(dev)
        r = cal.eval_cand(X, W0, b_t, beta, rot, "SR", fq, dev)
        rows.append(dict(granularity=name, path=path, params=params,
                         flops_per_token=fl, **r))
        print(f"[gran] {name:24s} {path} j_out={r['j_out']:.4f}",
              flush=True)

    for path, beta in (("first", 0.40), ("rec", 0.45)):
        ev("G0_identity", StructuredRotation(dict(family="identity")),
           0, 0, path, beta)
        best = None
        for seed in range(8):
            r = StructuredRotation(dict(family="dual", block=256,
                                        seed=seed), device=dev)
            X = tens["X_first" if path == "first"
                     else "X_rec_all"].float().to(dev)
            m = cal.eval_cand(X, W0, b_t, beta, r, "SR", fq, dev)
            if best is None or m["j_out"] < best[1]["j_out"]:
                best = (r, m, seed)
        rows.append(dict(granularity="G1_branchwise", path=path,
                         params=0, flops_per_token=flops("f", 256),
                         seed=best[2], **best[1]))
        print(f"[gran] G1_branchwise {path} "
              f"j_out={best[1]['j_out']:.4f}")
        ev("G2_pairwise", Pairwise45(dev), 0, flops("f", 2), path,
           beta)
        for b in (2, 4, 8, 16, 32, 64, 128, 256, 512):
            best = None
            for seed in range(8):
                r = StructuredRotation(
                    dict(family="cross", block=b, seed=seed,
                         interleave_chunk=1), device=dev)
                X = tens["X_first" if path == "first"
                         else "X_rec_all"].float().to(dev)
                m = cal.eval_cand(X, W0, b_t, beta, r, "SR", fq, dev)
                if best is None or m["j_out"] < best[1]["j_out"]:
                    best = (r, m, seed)
            rows.append(dict(granularity=f"G3_cross_b{b}", path=path,
                             params=0, flops_per_token=flops("f", b),
                             seed=best[2], **best[1]))
            print(f"[gran] G3_cross_b{b:4d} {path} "
                  f"j_out={best[1]['j_out']:.4f}")
        ev("G5_full8192", StructuredRotation(
            dict(family="full", block=8192, seed=7), device=dev),
           0, flops("f", 8192), path, beta)
        for K in (1, 2, 4, 8, 16, 32, 64):
            lr = LearnedRotation("householder", K=K, device=dev)
            ev(f"G6_householder_K{K}", lr, lr.n_params(),
               flops("hh", K=K), path, beta)
        ev("G7_butterfly", Butterfly(13, dev), 0,
           flops("butterfly"), path, beta)
    json.dump(rows, open(os.path.join(
        rd, "tables", "granularity_grid.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
