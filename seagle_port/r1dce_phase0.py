"""R1DCE Phase-0 — audit every candidate context-rotation matrix (§4/§33).

Loads R1_T, R1_D, current R_C and every stored candidate; prints per-matrix
identity (path, file sha256, tensor sha, dtype, shape, orthogonality
residual, det sign) and pairwise similarity (||A-B||_F/sqrt(d),
tr(A^T B)/d, allclose). Mints the frozen per-arm {"R_C": ...} ckpts into
<run-dir>/rotations/ (pre-registered random seeds) and writes
tables/rotation_matrix_similarity.csv. CPU fp64; no model load.
"""
import argparse
import csv
import hashlib
import json
import os

import torch

WS = "/home/thahn1230/dflash_workspace"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
FIDI_RD = (f"{WS}/dflash/runs/"
           "dflash_full_interface_distribution_intervention_20260811_074001")
DFST_RD = f"{WS}/dflash/runs/dflash_seagle_transfer_20260807_180238"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt.best"
D = 4096
RANDOM_SEEDS = (101, 102, 103)      # pre-registered (configs/preregistration)


def fsha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def tsha(t):
    return hashlib.sha256(
        t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]


def load_rbin_r1(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("R1", "R"):
            if k in ck:
                return ck[k]
    raise KeyError(f"no R1 in {path}: {list(ck.keys())[:8]}")


def load_rc(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for k in ("R_C", "R"):
            if k in ck:
                return ck[k]
        raise KeyError(f"no R_C in {path}: {list(ck.keys())[:8]}")
    return ck


def audit_one(name, t, src):
    x = t.detach().to(torch.float64)
    I = torch.eye(D, dtype=torch.float64)
    orth = torch.linalg.norm(x.t() @ x - I, ord="fro").item()
    sign, _ = torch.linalg.slogdet(x)
    return {"name": name, "source": src, "dtype": str(t.dtype),
            "shape": tuple(t.shape), "tensor_sha": tsha(t),
            "orth_fro": orth, "det_sign": int(sign.item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    os.makedirs(f"{rd}/rotations", exist_ok=True)
    os.makedirs(f"{rd}/tables", exist_ok=True)

    mats, meta = {}, {}

    def add(name, t, src):
        mats[name] = t.detach().to(torch.float64)
        meta[name] = audit_one(name, t, src)
        m = meta[name]
        print(f"[{name:14s}] src={src}")
        print(f"    dtype={m['dtype']} shape={m['shape']} "
              f"tensor_sha={m['tensor_sha']} ||R^TR-I||_F={m['orth_fro']:.3e}"
              f" det_sign={m['det_sign']:+d}")

    print("=" * 70)
    print("PHASE-0: CANDIDATE CONTEXT-ROTATION MATRIX AUDIT")
    print("=" * 70)
    print(f"R1_T file  : {S1RBIN}  sha={fsha(S1RBIN)}")
    add("R1_T", load_rbin_r1(S1RBIN), S1RBIN)

    print(f"R1_D file  : {R1D_CKPT}  sha={fsha(R1D_CKPT)}")
    ck = torch.load(R1D_CKPT, map_location="cpu", weights_only=False)
    add("R1_D", ck["R1_D"], R1D_CKPT)
    r2 = ck.get("R2_D")
    print(f"    R2_D: {len(r2)} per-layer [{tuple(r2[0].shape)}] "
          f"(UNTOUCHED by this study, §18)")

    # current deployed R_C: the M5/M6 chain used eval_al --vsq-rc rt,
    # i.e. rc_matrix_buf := R1_T (verified against scheduler job configs).
    add("RC_current", mats["R1_T"].clone(), "vsq-rc=rt (R1 of s1 R.bin)")

    for name, path in [("Hadamard", f"{FIDI_RD}/tables/rc_hadamard.pt"),
                       ("Random_fidi", f"{FIDI_RD}/tables/rc_random.pt"),
                       ("RC1_reuseRT", f"{DFST_RD}/rotations/RC1_reuseRT.pt"),
                       ("RC_L0", f"{DFST_RD}/rotations/RC_L0.pt")]:
        if os.path.exists(path):
            print(f"{name} file: {path}  sha={fsha(path)}")
            add(name, load_rc(path).float(), path)
        else:
            print(f"[{name}] MISSING: {path}")

    # pre-registered fresh random orthogonal seeds (QR of gaussian, det +1)
    for s in RANDOM_SEEDS:
        g = torch.Generator().manual_seed(s)
        A = torch.randn(D, D, generator=g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R))[None, :]   # unique QR, det±1
        if torch.linalg.slogdet(Q)[0] < 0:
            Q[:, 0] = -Q[:, 0]                            # force det +1
        add(f"Random_s{s}", Q, f"QR(randn) seed={s}")

    # C7b combined matrix: R_combo = R1_D @ RC_current (row-vector order:
    # Ht @ R1_D @ RC == Ht @ (R1_D @ RC))
    add("R_combo", mats["R1_D"] @ mats["RC_current"],
        "R1_D @ RC_current (C7b)")

    # ---- pairwise similarity table
    names = list(mats.keys())
    sq = D ** 0.5
    rows = []
    print("\nPAIRWISE SIMILARITY  (||A-B||_F/sqrt(d) | tr(A^T B)/d | equal)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            df = torch.linalg.norm(mats[a] - mats[b], ord="fro").item() / sq
            tr = torch.trace(mats[a].t() @ mats[b]).item() / D
            eq = bool(torch.allclose(mats[a], mats[b], atol=1e-6))
            rows.append({"A": a, "B": b, "fro_over_sqrtd": df,
                         "trace_align": tr, "allclose_1e-6": eq})
            print(f"  {a:14s} vs {b:14s}: {df:8.4f} | {tr:+8.5f} | {eq}")

    hdr = ["A", "B", "fro_over_sqrtd", "trace_align", "allclose_1e-6"]
    with open(f"{rd}/tables/rotation_matrix_similarity.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    json.dump({"matrices": meta,
               "rc_current_equals_r1t": bool(
                   torch.allclose(mats["RC_current"], mats["R1_T"])),
               "rc_current_equals_r1d": bool(
                   torch.allclose(mats["RC_current"], mats["R1_D"]))},
              open(f"{rd}/tables/phase0_matrix_audit.json", "w"), indent=1,
              default=str)

    # ---- mint frozen per-arm ckpts (fp32, {"R_C": ...} for eval_al --vsq-rc)
    minted = {"rc_R1D.pt": "R1_D", "rc_R1T.pt": "R1_T",
              "rc_current.pt": "RC_current", "rc_hadamard.pt": "Hadamard",
              "rc_combo_c7b.pt": "R_combo"}
    for s in RANDOM_SEEDS:
        minted[f"rc_random_s{s}.pt"] = f"Random_s{s}"
    if "RC_L0" in mats:
        minted["rc_learned_l0.pt"] = "RC_L0"
    man = {}
    for fn, src in minted.items():
        p = f"{rd}/rotations/{fn}"
        torch.save({"R_C": mats[src].to(torch.float32),
                    "provenance": meta[src]["source"]}, p)
        man[fn] = {"from": src, "file_sha": fsha(p),
                   "tensor_sha": meta[src]["tensor_sha"]}
        print(f"minted {p}  <- {src}  sha={man[fn]['file_sha']}")
    json.dump(man, open(f"{rd}/rotations/MANIFEST.json", "w"), indent=1)

    print("\nKEY ANSWERS (§33):")
    print(f"  current R_C == R1_T ? "
          f"{bool(torch.allclose(mats['RC_current'], mats['R1_T']))}")
    print(f"  current R_C == R1_D ? "
          f"{bool(torch.allclose(mats['RC_current'], mats['R1_D']))}")
    if "RC1_reuseRT" in mats:
        print(f"  DFST RC1_reuseRT == R1_T ? "
              f"{bool(torch.allclose(mats['RC1_reuseRT'], mats['R1_T'], atol=1e-5))}")


if __name__ == "__main__":
    main()
