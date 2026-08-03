#!/usr/bin/env python
"""R-EP3-P calibration engine (spec §12-§13).

Stages (all on the staged held-out calibration tensors, official
deployment quantizers, never MT-Bench):

  s1     identity-rotation beta grids (reproduce EP3-P optimum)
  s2     rotation proxy sweep at fixed EP3-P betas (families x block
         sizes x >=16 seeds x SR/RS) + rotation-only (beta=0) row for
         seed 0 of every config; shardable across GPUs
  s3     beta recalibration (coarse 0.05 grid + fine 0.01 local) for
         the best config of each rotation family
  s4     25-pair joint search (top-5 first x top-5 recurrent by J)

Ranking J = J_output + 0.1 * J_cos + 0.01 * normalized(J_max)
(primary J_output = W4A4 projection-output NMSE vs FP reference,
bias included on both sides; same quantizers as deployment).
Results land in candidates/*.jsonl and tables/rep3p_search.json.
"""
import argparse, itertools, json, math, os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 transform_xw)

D = 4096
BETA0 = dict(first=0.40, rec=0.45)


def nmse(y, ref):
    y = y.double(); ref = ref.double()
    return float(((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-30))


@torch.no_grad()
def eval_cand(X_raw, W0, bias, beta, rot, order, fq, dev):
    """X_raw: raw unscaled [N, 8192] fp32 on dev. Returns metric dict."""
    m = float(D ** beta) if beta > 0 else 1.0
    Xt, Wt = transform_xw(X_raw, W0, m, rot, order)
    Yref = X_raw @ W0.t()
    if bias is not None:
        Yref = Yref + bias
    Wq = fq._weight_fake_quant(Wt.half(), 4).float()
    aq = fq._act_quantizer(4)
    aq.find_params(Xt.half())
    Xq = aq(Xt.half()).float()
    aq.free()
    Y = Xq @ Wq.t()
    if bias is not None:
        Y = Y + bias
    d = (Y - Yref)
    j_out = nmse(Y, Yref)
    j_cos = 1.0 - float(torch.nn.functional.cosine_similarity(
        Y.flatten(), Yref.flatten(), dim=0))
    j_max = float(d.abs().amax(dim=1).mean())
    ref_scale = float(Yref.abs().amax(dim=1).mean()) + 1e-12
    a4 = nmse(Xq, Xt)
    w4 = nmse(Wq, Wt)
    return dict(beta=round(beta, 4), m=round(m, 4), order=order,
                j_out=j_out, j_cos=j_cos,
                j_max_norm=j_max / ref_scale, a4_nmse=a4, w4_nmse=w4,
                J=j_out + 0.1 * j_cos + 0.01 * (j_max / ref_scale))


def candidates():
    """(family, block, seed, order) grid for s2."""
    out = []
    for fam in ("e_only", "h_only", "dual"):
        for blk in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            for seed in range(16):
                out.append((fam, blk, seed, "SR"))
    for fam in ("full", "cross"):
        for blk in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                    8192):
            for seed in range(16):
                for order in ("SR", "RS"):
                    out.append((fam, blk, seed, order))
    return out


def load_X(tens, path, dev):
    key = "X_first" if path == "first" else "X_rec_all"
    return tens[key].float().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["s1", "s2", "s3", "s4"])
    ap.add_argument("--target", default="int4",
                    choices=["int4", "fp16"])
    ap.add_argument("--path", default="first",
                    choices=["first", "rec"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    from eagle_spinquant import fake_w4a4_draft as fq
    tens = torch.load(os.path.join(
        rd, "tensors", f"calib_{args.target}.pt"),
        map_location="cpu", weights_only=False)
    W0 = tens["W"].float().to(dev)
    bias = tens.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    os.makedirs(os.path.join(rd, "candidates"), exist_ok=True)

    if args.stage == "s1":
        X = load_X(tens, args.path, dev)
        rot = StructuredRotation(dict(family="identity"))
        rows = []
        hi = 61 if args.path == "first" else 66
        for bi in range(20, hi, 5):
            rows.append(eval_cand(X, W0, b_t, bi / 100, rot, "SR",
                                  fq, dev))
        best = min(rows, key=lambda r: r["J"])["beta"]
        for dl in range(-5, 6):
            bb = round(best + dl / 100, 4)
            if 0 < bb <= 1 and not any(r["beta"] == bb for r in rows):
                rows.append(eval_cand(X, W0, b_t, bb, rot, "SR", fq,
                                      dev))
        rows.sort(key=lambda r: r["J"])
        out = os.path.join(rd, "candidates",
                           f"s1_{args.target}_{args.path}.json")
        json.dump(rows, open(out, "w"), indent=1)
        print(f"[s1] {args.target}/{args.path} best "
              f"beta={rows[0]['beta']} J={rows[0]['J']:.4f} "
              f"j_out={rows[0]['j_out']:.4f}")
        return 0

    if args.stage == "s2":
        X = load_X(tens, args.path, dev)
        cands = candidates()[args.shard::args.nshards]
        beta = BETA0[args.path]
        out_p = os.path.join(
            rd, "candidates",
            f"s2_{args.target}_{args.path}_shard{args.shard}.jsonl")
        done = set()
        if os.path.exists(out_p):
            for ln in open(out_p):
                r = json.loads(ln)
                done.add((r["family"], r["block"], r["seed"],
                          r["order"], r["beta"]))
        f = open(out_p, "a")
        for fam, blk, seed, order in cands:
            rot = StructuredRotation(
                dict(family=fam, block=blk, seed=seed,
                     interleave_chunk=1),
                device=dev)
            jobs = [beta] + ([0.0] if seed == 0 and order == "SR"
                             else [])
            for bb in jobs:
                if (fam, blk, seed, order, round(bb, 4)) in done:
                    continue
                r = eval_cand(X, W0, b_t, bb, rot, order, fq, dev)
                r.update(family=fam, block=blk, seed=seed)
                f.write(json.dumps(r) + "\n")
                f.flush()
        f.close()
        print(f"[s2] shard {args.shard}/{args.nshards} done")
        return 0

    if args.stage == "s3":
        import glob
        rows = []
        for p in glob.glob(os.path.join(
                rd, "candidates",
                f"s2_{args.target}_{args.path}_shard*.jsonl")):
            rows += [json.loads(x) for x in open(p)]
        rows = [r for r in rows if r["beta"] > 0]
        best_per_family = {}
        for r in sorted(rows, key=lambda r: r["J"]):
            best_per_family.setdefault(r["family"], r)
        X = load_X(tens, args.path, dev)
        hi = 61 if args.path == "first" else 66
        out = {}
        for fam, r0 in best_per_family.items():
            rot = StructuredRotation(
                dict(family=fam, block=r0["block"], seed=r0["seed"],
                     interleave_chunk=1), device=dev)
            grid = [eval_cand(X, W0, b_t, bi / 100, rot, r0["order"],
                              fq, dev) for bi in range(20, hi, 5)]
            bb = min(grid, key=lambda r: r["J"])["beta"]
            for dl in range(-5, 6):
                v = round(bb + dl / 100, 4)
                if 0 < v <= 1 and not any(g["beta"] == v
                                          for g in grid):
                    grid.append(eval_cand(X, W0, b_t, v, rot,
                                          r0["order"], fq, dev))
            grid.sort(key=lambda g: g["J"])
            out[fam] = dict(config=dict(family=fam,
                                        block=r0["block"],
                                        seed=r0["seed"],
                                        order=r0["order"]),
                            grid=grid, best=grid[0])
            print(f"[s3] {args.path}/{fam} b{r0['block']} "
                  f"s{r0['seed']} {r0['order']}: beta "
                  f"{BETA0[args.path]}->{grid[0]['beta']} "
                  f"j_out {r0['j_out']:.4f}->"
                  f"{grid[0]['j_out']:.4f}")
        json.dump(out, open(os.path.join(
            rd, "candidates",
            f"s3_{args.target}_{args.path}.json"), "w"), indent=1)
        return 0

    if args.stage == "s4":
        picks = {}
        for path in ("first", "rec"):
            d = json.load(open(os.path.join(
                rd, "candidates", f"s3_{args.target}_{path}.json")))
            flat = [dict(path=path, family=f, **v["best"],
                         config=v["config"])
                    for f, v in d.items()]
            picks[path] = sorted(flat,
                                 key=lambda r: r["J"])[:5]
        Xf = load_X(tens, "first", dev)
        Xr = load_X(tens, "rec", dev)
        pairs = []
        for cf, cr in itertools.product(picks["first"],
                                        picks["rec"]):
            rf = StructuredRotation(dict(interleave_chunk=1,
                                         **cf["config"]), device=dev)
            rr = StructuredRotation(dict(interleave_chunk=1,
                                         **cr["config"]), device=dev)
            mf = eval_cand(Xf, W0, b_t, cf["beta"], rf,
                           cf["config"]["order"], fq, dev)
            mr = eval_cand(Xr, W0, b_t, cr["beta"], rr,
                           cr["config"]["order"], fq, dev)
            pairs.append(dict(first=dict(cf["config"],
                                         beta=cf["beta"]),
                              rec=dict(cr["config"], beta=cr["beta"]),
                              J_first=mf["J"], J_rec=mr["J"],
                              j_out_first=mf["j_out"],
                              j_out_rec=mr["j_out"],
                              J_sum=mf["J"] + mr["J"]))
        pairs.sort(key=lambda p: p["J_sum"])
        json.dump(dict(top5=picks, pairs25=pairs,
                       top9=pairs[:9], top3_by_proxy=pairs[:3]),
                  open(os.path.join(
                      rd, "candidates",
                      f"s4_{args.target}_pairs.json"), "w"),
                  indent=1)
        print(f"[s4] best pair: {json.dumps(pairs[0])[:300]}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
