#!/usr/bin/env python
"""Rotation geometry + quantization-error analysis (study spec 17).

For every LK checkpoint (and the shared R_T reference): geodesic/Frobenius
distance from R_T, max element change, orthogonality error, and — through
the exact fold — transformed-weight kurtosis, W4 clip fraction, occupied
W4 levels, weight quantization NMSE per layer. CPU/GPU cheap (one fold).

Writes <run>/tables/rotation_geometry.csv.
"""
import argparse, csv, glob, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.residual_rotation import (FullRotation,
                                               rotation_geometry)
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant import fake_w4a4_draft as fq

KIND = "learned_chat_w4a4kv16"


def wstats(w):
    wf = w.float()
    mu = wf.mean()
    sd = wf.std().clamp_min(1e-12)
    kurt = float((((wf - mu) / sd) ** 4).mean())
    q = fq._weight_fake_quant(w.detach(), 4).float()
    nmse = float(((q - wf) ** 2).sum() / (wf ** 2).sum())
    absmax = wf.abs().amax(dim=1, keepdim=True)
    clipped = float((q.abs() >= 0.999 * q.abs().amax(dim=1, keepdim=True))
                    .float().mean())
    levels = torch.unique((q / (q.abs().amax() / 7)).round()).numel()
    return kurt, nmse, clipped, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    from safetensors.torch import load_file, safe_open
    sfp = os.path.join(paths["draft_path"], "model.safetensors")
    sd = load_file(sfp) if os.path.exists(sfp) else torch.load(
        os.path.join(paths["draft_path"], "pytorch_model.bin"),
        map_location="cpu", weights_only=True)
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    rows = []
    cands = [("SHARED_RT", None)]
    cands += [(os.path.basename(p)[:-3], p) for p in sorted(glob.glob(
        os.path.join(args.run_dir, "rotations", "LK_*.pt")))]
    for tag, ckp in cands:
        if ckp is None:
            R_D, alpha = R_T.clone(), 32.0
            core = None
        else:
            ck = torch.load(ckp, map_location="cpu", weights_only=False)
            R_D, alpha = ck["R_D"].float(), float(ck.get("alpha", 32.0))
            core = ck.get("core_weights")
        geo = rotation_geometry(R_D.double(), R_T.double())
        m = ExactQATRotatedDraft(sd, R_T, gamma, W_lm,
                                 FullRotation(R_D).to(dev),
                                 alpha_init=alpha, device=dev,
                                 first_fold_R=R_T)
        if core is not None:
            with torch.no_grad():
                for n in core:
                    getattr(m, n).copy_(core[n].float().to(dev))
        tw = m.transformed_weights(exact=True)
        r = dict(tag=tag, alpha=alpha,
                 **{k: round(v, 6) for k, v in geo.items()})
        for lname in ("W_rec", "q", "o", "down"):
            kurt, nmse, clip, lev = wstats(tw[lname])
            r[f"{lname}_kurtosis"] = round(kurt, 3)
            r[f"{lname}_w4_nmse"] = round(nmse, 6)
            r[f"{lname}_clip_frac"] = round(clip, 5)
            r[f"{lname}_levels"] = lev
        rows.append(r)
        print(f"[geom] {tag}: geo={geo['geodesic_dist']:.4f} "
              f"Wrec_nmse={r['W_rec_w4_nmse']}", flush=True)
        del m
        torch.cuda.empty_cache()
    out = os.path.join(args.run_dir, "tables", "rotation_geometry.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[geom] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
