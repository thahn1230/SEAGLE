"""Mint the R_C control matrices used by the audit-Q3 arms (V_RCrand/V_RChad).

Deterministic: random orthogonal = QR of a seed-1234 gaussian with the
sign-fixed diagonal (unique factor); Hadamard = SpinQuant
random_hadamard_matrix under torch seed 1234. The .pt files are not
committed (67 MB each) — re-mint with this script; shas printed for the
provenance record.
"""
import argparse
import hashlib

import torch

from . import SPINQUANT_ROOT  # noqa: F401  (sys.path side effect)
from utils.hadamard_utils import random_hadamard_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    A = torch.randn(4096, 4096, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))
    err = (Q @ Q.t() - torch.eye(4096, dtype=torch.float64)).abs().max()
    torch.save({"R_C": Q.float(),
                "meta": {"kind": "random_orthogonal_qr", "seed": args.seed,
                         "orth_err": err.item()}},
               f"{args.out_dir}/rc_random.pt")
    H = random_hadamard_matrix(4096, "cpu").float()
    errh = (H @ H.t() - torch.eye(4096)).abs().max()
    torch.save({"R_C": H, "meta": {"kind": "random_hadamard",
                                   "seed": args.seed,
                                   "orth_err": errh.item()}},
               f"{args.out_dir}/rc_hadamard.pt")
    for n in ("rc_random.pt", "rc_hadamard.pt"):
        sha = hashlib.sha256(open(f"{args.out_dir}/{n}", "rb").read())
        print(n, sha.hexdigest()[:16])


if __name__ == "__main__":
    main()
