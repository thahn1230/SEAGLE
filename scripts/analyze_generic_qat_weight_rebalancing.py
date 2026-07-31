#!/usr/bin/env python
"""Does generic QAT learn P3-like geometry? (study §8)

For anchor, every GQAT step-ckpt, and the P3 references: measure
RMS/absmax of W_e and W_h, their ratio, and per-block W4 NMSE of each
slice under the deployed fold (identity or gamma_R1 basis). Writes
tables/gqat_rebalancing_<variant>.json with the ratio trajectory.
"""
import argparse, glob, json, os, re, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import fake_w4a4_draft as fq

D = 4096


def stats(sd, dev="cuda:0", alpha=None):
    W = sd["fc.weight"].float()
    We, Wh = W[:, :D], W[:, D:]
    if alpha:
        We = We / alpha
    r = {}
    for k, w in (("W_e", We), ("W_h", Wh)):
        r[f"rms_{k}"] = float(w.pow(2).mean().sqrt())
        r[f"absmax_{k}"] = float(w.abs().max())
    r["rms_ratio"] = r["rms_W_e"] / r["rms_W_h"]
    Wcat = torch.cat([We, Wh], dim=1).half().to(dev)
    Wq = fq._weight_fake_quant(Wcat, 4).float().cpu()
    ref = torch.cat([We, Wh], dim=1)
    for k, sl in (("W_e", slice(0, D)), ("W_h", slice(D, 2 * D))):
        num = (Wq[:, sl] - ref[:, sl]).pow(2).sum()
        r[f"w4_nmse_{k}"] = float(num / (ref[:, sl].pow(2).sum()
                                         + 1e-12))
    return r


def load_sd(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return sd.get("draft_state_dict", sd.get("model", sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True, choices=["T0", "T1"])
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--lp3-alpha", type=float, required=True,
                    help="legacy P3 alpha for the LP3 reference row")
    args = ap.parse_args()
    rd = args.run_dir
    out = dict(variant=args.variant, rows=[])
    out["rows"].append(dict(name="anchor", step=0,
                            **stats(load_sd(args.anchor))))
    out["rows"].append(dict(
        name="LP3_effective", step=0,
        **stats(load_sd(args.anchor), alpha=args.lp3_alpha)))
    for s in (0, 1, 2):
        cks = sorted(
            glob.glob(os.path.join(
                rd, "ckpts", f"GQAT_{args.variant}_s{s}_step*.pt")),
            key=lambda p: int(re.search(r"_step(\d+)", p).group(1)))
        last = os.path.join(rd, "ckpts",
                            f"GQAT_{args.variant}_s{s}_last.pt")
        if os.path.exists(last):
            cks.append(last)
        for p in cks:
            m = re.search(r"_step(\d+)", p)
            step = int(m.group(1)) if m else 3000
            out["rows"].append(dict(name=f"GQAT_s{s}", step=step,
                                    **stats(load_sd(p))))
    path = os.path.join(rd, "tables",
                        f"gqat_rebalancing_{args.variant}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    a = out["rows"][0]["rms_ratio"]
    fin = [r for r in out["rows"] if r["name"].startswith("GQAT")
           and r["step"] >= 3000]
    print(f"[rebal] {args.variant}: anchor ratio {a:.3f} -> final "
          f"{[round(r['rms_ratio'],3) for r in fin]} "
          f"(LP3-effective {out['rows'][1]['rms_ratio']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
