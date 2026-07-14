"""Commutator diagnostic: C = diag(gamma_f)·R1 − R1·diag(gamma_f) ≠ 0, hence
(n·R1)·diag(γ) ≠ n·diag(γ)·R1 — multiplying gamma directly in the rotated basis
is NOT equivalent. Writes artifacts/b2_split_study/commutator.json using the
REAL target R1 + target_final_rms_gamma when available, else random."""

import json
import os

import torch

from b2_common import PROJECT_ROOT, rand_orthogonal, rand_gamma, rms_normalize

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")


def _load_real_R1_gamma():
    rbin = os.path.join(PROJECT_ROOT, "outputs", "rotations",
                        "random_hadamard", "R.bin")
    if not os.path.isfile(rbin):
        return None, None
    R = torch.load(rbin, map_location="cpu", weights_only=False)
    R1 = R["R1"].double()
    # gamma: cached stash if present, else None (random fallback)
    for cand in ("target_final_rms_gamma.pt",):
        p = os.path.join(PROJECT_ROOT, "outputs", cand)
        if os.path.isfile(p):
            return R1, torch.load(p, weights_only=False).double()
    return R1, None


def commutator_stats(R1, gamma):
    Dg = torch.diag(gamma)
    C = Dg @ R1 - R1 @ Dg
    ref = Dg @ R1
    return dict(fro_norm=float(C.norm()),
                spectral_norm=float(torch.linalg.matrix_norm(C, 2)),
                fro_ratio=float(C.norm() / ref.norm()))


def test_commutator_nonzero_and_gamma_order_matters():
    R1r, gamma_r = _load_real_R1_gamma()
    R1 = R1r if R1r is not None else rand_orthogonal(4096, seed=0)
    gamma = gamma_r if gamma_r is not None else rand_gamma(R1.shape[0], seed=1)
    st = commutator_stats(R1, gamma)
    assert st["fro_ratio"] > 1e-3, f"commutator unexpectedly tiny: {st}"

    # (n R) diag(γ) vs n diag(γ) R on real vectors
    n = rms_normalize(torch.randn(8, R1.shape[0], dtype=torch.float64))
    wrong = (n @ R1) * gamma
    right = (n * gamma) @ R1
    rel = float((wrong - right).norm() / right.norm())
    assert rel > 1e-3, "gamma commuted with R1?!"
    st["gamma_order_rel_l2"] = rel
    st["source"] = ("real_R1" if R1r is not None else "random_R1") + \
        ("+real_gamma" if gamma_r is not None else "+random_gamma")

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "commutator.json"), "w") as f:
        json.dump(st, f, indent=2)


if __name__ == "__main__":
    test_commutator_nonzero_and_gamma_order_matters()
    print("OK", open(os.path.join(ART, "commutator.json")).read())
