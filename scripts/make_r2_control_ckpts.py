#!/usr/bin/env python
"""Build diagnostic/control rotation checkpoints for the GS R1/R2 study.

--mode composed : A4 COMPOSED_NO_JOINT_TRAINING — R_D from the selected
                  A1 checkpoint + R2_D from the selected A2 checkpoint,
                  combined WITHOUT joint fine-tuning.
--mode random   : matched random R2 controls — n random residual Cayley
                  perturbations R2 = R2_B C(B_rand) with ||B_rand||_F
                  matched EXACTLY to the learned generator's Frobenius
                  norm (matched GENERATOR Frobenius norm only; geodesic /
                  eigenangle magnitudes are recorded but NOT matched).
                  R_D is copied from --base-ckpt if given, else R1_T.

Outputs eval-ready checkpoints (schema: R_D, R2_D, R2_W, alpha, meta).
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.pmg_configs import EP3G
from eagle_spinquant.residual_rotation import (cayley, rotation_geometry)

KIND = "learned_chat_w4a4kv16"
ALPHA_GS = float(EP3G["int4"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["composed", "random"])
    ap.add_argument("--a1-ckpt", help="composed: source of R_D")
    ap.add_argument("--a2-ckpt", help="composed/random: source of R2_D / "
                                      "matched generator norm")
    ap.add_argument("--base-ckpt", default=None,
                    help="random: copy R_D from this ckpt (default R1_T)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    R2_B = fq.baseline_r2(0)

    if args.mode == "composed":
        a1 = torch.load(args.a1_ckpt, map_location="cpu",
                        weights_only=False)
        a2 = torch.load(args.a2_ckpt, map_location="cpu",
                        weights_only=False)
        out = dict(R_D=a1["R_D"], R2_D=a2["R2_D"], R2_W=a2.get("R2_W"),
                   alpha=ALPHA_GS, alpha_rec=None,
                   meta=dict(mode="composed_no_joint_training",
                             a1_ckpt=args.a1_ckpt, a2_ckpt=args.a2_ckpt,
                             r1_geometry=rotation_geometry(
                                 a1["R_D"].float(), R_T),
                             r2_geometry=rotation_geometry(
                                 a2["R2_D"].float(), R2_B.float())))
        p = os.path.join(args.out_dir,
                         f"{args.prefix}COMPOSED_A1R1_A2R2.pt")
        torch.save(out, p)
        print(f"[ctrl] composed -> {p}")
        return 0

    a2 = torch.load(args.a2_ckpt, map_location="cpu", weights_only=False)
    Bl = a2["R2_W"] - a2["R2_W"].t()
    target_norm = float(Bl.norm())
    R_D_src = (torch.load(args.base_ckpt, map_location="cpu",
                          weights_only=False)["R_D"]
               if args.base_ckpt else R_T)
    for i in range(args.n):
        g = torch.Generator().manual_seed(7000 + i)
        W = torch.randn(128, 128, generator=g)
        B = W - W.t()
        B = B * (target_norm / float(B.norm()))
        C = cayley(B)                       # fp32, ResidualR2 convention
        R2_D = R2_B @ C.double()
        geo = rotation_geometry(R2_D.float(), R2_B.float())
        out = dict(R_D=R_D_src, R2_D=R2_D, R2_W=(B / 2).clone(),
                   alpha=ALPHA_GS, alpha_rec=None,
                   meta=dict(mode="random_r2_matched_generator_norm",
                             seed=7000 + i,
                             matched_generator_frob=target_norm,
                             actual_generator_frob=float(B.norm()),
                             learned_ckpt=args.a2_ckpt,
                             base_ckpt=args.base_ckpt,
                             r2_geometry=geo))
        p = os.path.join(args.out_dir,
                         f"{args.prefix}RANDR2_s{7000 + i}.pt")
        torch.save(out, p)
        print(f"[ctrl] random {i}: ||B||={float(B.norm()):.4f} "
              f"(target {target_norm:.4f}) geo={geo['geodesic_dist']:.4f}"
              f" -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
