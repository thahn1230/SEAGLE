#!/usr/bin/env python
"""Random-R6 control: 3 random residual Cayley perturbations matched to
the learned R6 in GENERATOR Frobenius norm (||A||_F of the skew
generator; geodesic/eigenangle magnitudes are reported but NOT matched).
Writes ckpts rotations/R6_RAND_s{k}.pt (R_D = frozen R5, R6 = random)
ready for the standard rot-cfg eval.

Usage: _r6_randctl.py <run_dir> --learned <R6 ckpt> --r5-ckpt <R5>
"""
import argparse, hashlib, json, os, sys
import torch
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from eagle_spinquant.residual_rotation import cayley  # noqa: E402

HD = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--learned", required=True)
    ap.add_argument("--r5-ckpt", required=True)
    args = ap.parse_args()
    ck = torch.load(args.learned, map_location="cpu",
                    weights_only=False)
    R6 = ck["R6"].double()
    r5ck = torch.load(args.r5_ckpt, map_location="cpu",
                      weights_only=False)
    g0 = torch.Generator().manual_seed(0)
    R2B = torch.linalg.qr(torch.randn(HD, HD, generator=g0,
                                      dtype=torch.float64))[0]
    # recover the learned generator: Q = R2B^T R6 = C(A) with
    # C(A) = (I - A/2)^{-1} (I + A/2)  =>  A = 2 (Q - I) (Q + I)^{-1}
    Q = R2B.t() @ R6
    I = torch.eye(HD, dtype=torch.float64)
    A_learned = 2.0 * (Q - I) @ torch.linalg.inv(Q + I)
    A_learned = 0.5 * (A_learned - A_learned.t())   # numerical re-skew
    fro = float(A_learned.norm())
    # sanity: reconstruction
    rec = float((R2B @ cayley(A_learned) - R6).abs().max())
    out = dict(learned_generator_fro=fro, reconstruction_maxabs=rec,
               controls=[])
    # learned R6 is stored fp32; recovery through the double-precision
    # inverse Cayley lands at fp32-rounding scale (~1e-7)
    assert rec < 1e-5, f"generator recovery failed ({rec})"
    for s in (101, 102, 103):
        g = torch.Generator().manual_seed(s)
        Ar = torch.randn(HD, HD, generator=g, dtype=torch.float64)
        Ar = Ar - Ar.t()
        Ar = Ar * (fro / float(Ar.norm()))
        R6r = R2B @ cayley(Ar)
        geo = float(torch.linalg.matrix_norm(
            torch.linalg.matrix_exp(torch.zeros(1)).new_tensor(0.0)
        )) if False else None  # geodesic not matched by design
        save = dict(R_D=r5ck["R_D"], alpha=r5ck.get("alpha"),
                    alpha_rec=r5ck.get("alpha_rec"),
                    R6=R6r.float(),
                    meta=dict(kind="random_r6_control", seed=s,
                              matched="generator_fro", fro=fro))
        p = os.path.join(args.run_dir, "rotations",
                         f"R6_RAND_s{s}.pt")
        torch.save(save, p)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        eig = torch.linalg.eigvals(cayley(Ar))
        ang = torch.angle(eig).abs()
        out["controls"].append(dict(
            seed=s, path=p, sha256=sha,
            generator_fro=float(Ar.norm()),
            max_eigenangle=float(ang.max()),
            mean_eigenangle=float(ang.mean())))
        print(f"[randctl] {p} fro={float(Ar.norm()):.4f}")
    eigL = torch.linalg.eigvals(cayley(A_learned))
    angL = torch.angle(eigL).abs()
    out["learned"] = dict(max_eigenangle=float(angL.max()),
                          mean_eigenangle=float(angL.mean()))
    dst = os.path.join(args.run_dir, "geometry", "r6_randctl.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"[randctl] -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
