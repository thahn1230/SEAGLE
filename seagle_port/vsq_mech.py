"""§19-20 counterfactual: does Vanilla SpinQuant already Gaussianize H_t?

Computes H_t statistics (kurtosis, absmax/RMS, top-0.1% channel energy,
A4 zero-code/entropy/NMSE) under three deployed constructions on the SAME
cached target hiddens (hcache, deployed W4A4 s1 basis):

  M1-analog : stock fc (no target fold, no draft rotation)  [naive basis]
  M3        : R1_T-folded fc, H_t before any R_C            [VSQ basis]
  M5        : M3 then H_t @ R_C (= R1_T)                    [R_C basis]

Writes tables/mechanism_stats.csv. Verdict: if M3 H_t kurtosis ~3 and low
A4 NMSE, prior R_C novelty is weakened; if M3 leaves the pathology while
target PPL is good, the fused-context rotation loop is genuinely outside
vanilla SpinQuant's model-local closure.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .dkva_capture import act_quant_detail

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"


def stats(x):
    f = x.flatten().float()
    ctr = f - f.mean()
    var = ctr.pow(2).mean()
    am = f.abs().max().item()
    rms = f.pow(2).mean().sqrt().item()
    energy = x.float().pow(2).sum(0)
    e = energy.sort(descending=True).values
    top = e[:max(1, int(len(e) * 0.001))].sum() / e.sum()
    det = act_quant_detail(x.float())
    h = np.bincount(det["codes"].cpu().numpy().flatten(),
                    minlength=16)[:16]
    hp = h / h.sum()
    ent = float(-(hp[hp > 0] * np.log2(hp[hp > 0])).sum())
    return {"kurtosis": (ctr.pow(4).mean() / (var ** 2 + 1e-30)).item(),
            "absmax": am, "rms": rms, "absmax_over_rms": am / (rms + 1e-9),
            "top01pct_energy": top.item(),
            "a4_zero_code": det["zero_int_ratio"],
            "a4_entropy_bits": ent,
            "a4_nmse": float(det["tok_nmse"].mean())}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-rows", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    d0 = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sq.load_rbin(args.rbin)["R1"]
    fc_stock = d0.fc.weight.data.float().to(dev)
    gam = d0.hidden_norm.weight.data.float().to(dev)
    dF = interfaces.fold_wc(d0, R1)
    fc_fold = dF.fc.weight.data.float().to(dev)
    R1d = R1.float().to(dev)
    del d0, dF

    def rms_norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gam

    files = sorted(glob.glob(f"{args.cache_dir}/row*.npz"))[:args.n_rows]
    acc = {k: [] for k in ("M1_naiveHt", "M3_vsqHt", "M5_rcHt")}
    for f in files:
        z = np.load(f)
        # cached hidden IS the deployed rotated-target basis (H R1_T)
        Hrot = torch.tensor(z["hidden"], dtype=torch.float32,
                            device=dev)
        Horig = (Hrot.reshape(-1, 5, 4096) @ R1d.t()).reshape(-1, 20480)
        acc["M1_naiveHt"].append(rms_norm(Horig @ fc_stock.t()))
        Ht3 = rms_norm(Hrot @ fc_fold.t())
        acc["M3_vsqHt"].append(Ht3)
        acc["M5_rcHt"].append(Ht3 @ R1d)
    rows = []
    for k, v in acc.items():
        X = torch.cat(v)
        rows.append({"tensor": k, "n_rows": X.shape[0], **stats(X)})
        print(rows[-1], flush=True)
    with open(os.path.join(args.run_dir, "tables",
                           "mechanism_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[vsq_mech] DONE")


if __name__ == "__main__":
    main()
