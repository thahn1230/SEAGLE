"""R1DCE §6 — H_t candidate-rotation battery on the SAME paired tokens.

Builds FP-path H_t (bare-RMS of folded-W_c output; FIDI convention) from the
frozen hcache, reservoir-samples 8000 tokens (seed 0), then for every
candidate R computes X_R = H_t @ R and the full §6 stat set + A4
diagnostics.  The token sample is saved to captures/ht_sample.npz and
REUSED by r1dce_kv_awdecomp and the figures, so all comparisons are
token-paired.  Writes tables/ht_candidate_stats.csv +
tables/ht_a4_candidate_stats.csv.
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
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
CAP = 8000
SEED = 0


def rms_bare(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def candidates(rd, dev):
    """name -> fp32 rotation (or None for identity)."""
    rot = lambda f: torch.load(f"{rd}/rotations/{f}", map_location=dev,
                               weights_only=False)["R_C"].float()
    fidi = (f"{WS}/dflash/runs/"
            "dflash_full_interface_distribution_intervention_20260811_074001")
    out = {}
    # trained arms (present only after C8/C9 have run)
    for name, fn in (("R1D_DeltaR", f"{rd}/rotations/rc_R1D_deltaR.pt"),
                     ("R1DC", f"{rd}/rotations/R1DC_l10.pt.best")):
        if os.path.exists(fn):
            out[name] = torch.load(fn, map_location=dev,
                                   weights_only=False)["R_C"].float()
    out.update({
        "none": None,
        "R1_D": rot("rc_R1D.pt"),
        "R1_T_currentRC": rot("rc_current.pt"),
        "Hadamard": torch.load(f"{fidi}/tables/rc_hadamard.pt",
                               map_location=dev,
                               weights_only=False)["R_C"].float(),
        "Random_s101": rot("rc_random_s101.pt"),
        "Random_s102": rot("rc_random_s102.pt"),
        "Random_s103": rot("rc_random_s103.pt"),
        "Random_s1234": torch.load(f"{fidi}/tables/rc_random.pt",
                                   map_location=dev,
                                   weights_only=False)["R_C"].float(),
        "RC_L0_learned": rot("rc_learned_l0.pt"),
        "R_combo_c7b": rot("rc_combo_c7b.pt"),
    })
    return out


@torch.inference_mode()
def build_ht_sample(dev):
    """Reservoir CAP tokens of FP-path H_t (original basis), seed SEED."""
    from dflash.model import DFlashDraftModel
    d0 = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    R1 = sq.load_rbin(S1RBIN)["R1"].float().to(dev)
    dF = interfaces.fold_wc(d0, R1)
    Wfc = dF.fc.weight.data.float()             # [4096, 20480] folded
    rng = np.random.default_rng(SEED)
    res = torch.zeros(CAP, 4096, dtype=torch.float32, device=dev)
    seen = 0
    for f in sorted(glob.glob(f"{VSQ_RD}/hcache_w4a4_s1/row*.npz")):
        H = torch.tensor(np.load(f)["hidden"], dtype=torch.float32,
                         device=dev)
        Ht = rms_bare(H @ Wfc.t())
        for j in range(Ht.shape[0]):
            if seen < CAP:
                res[seen] = Ht[j]
            else:
                k = rng.integers(0, seen + 1)
                if k < CAP:
                    res[k] = Ht[j]
            seen += 1
    del d0, dF
    torch.cuda.empty_cache()
    return res, seen


def stat_row(name, X):
    """§6 distribution stats on fp32 [N,4096]."""
    a = X.abs()
    flat = a.flatten().float()
    rms = X.pow(2).mean().sqrt().item()
    fl = flat.cpu().numpy()
    p99, p999, p9999 = np.percentile(fl, [99, 99.9, 99.99])
    ctr = X - X.mean()
    kurt = (ctr.pow(4).mean() / (ctr.pow(2).mean() ** 2 + 1e-30)).item()
    e = X.pow(2).sum(0)
    es, _ = torch.sort(e, descending=True)
    tot = es.sum().item() + 1e-30
    top = lambda f: es[:max(1, int(len(es) * f))].sum().item() / tot
    return {"candidate": name, "n_tokens": X.shape[0],
            "rms": rms, "absmax": a.max().item(),
            "p99": float(p99), "p99.9": float(p999), "p99.99": float(p9999),
            "kurtosis": kurt, "max_over_rms": a.max().item() / (rms + 1e-30),
            "top0.01pct_energy": top(0.0001), "top0.1pct_energy": top(0.001),
            "top0.5pct_energy": top(0.005), "top1pct_energy": top(0.01)}


def a4_row(name, X):
    """§6 A4 diagnostics on fp32 [N,4096]."""
    d = act_quant_detail(X)
    codes = torch.as_tensor(d["codes"]).to(X.device).long()
    tok_nmse = torch.as_tensor(d["tok_nmse"]).float().cpu()
    N, C = codes.shape
    hist = torch.bincount(codes.flatten(), minlength=16).float()
    p = hist / hist.sum()
    ent_marg = -(p[p > 0] * p[p > 0].log2()).sum().item()
    off = codes + torch.arange(N, device=X.device)[:, None] * 16
    per = torch.bincount(off.flatten(), minlength=N * 16).view(N, 16).float()
    pp = per / per.sum(1, keepdim=True)
    ent_tok = (-(pp * (pp + 1e-12).log2()).sum(1)).mean().item()
    uniq = (per > 0).sum(1).float().mean().item()
    scale = torch.as_tensor(d["scale"]).flatten().float().cpu().numpy()
    dq = torch.as_tensor(d["dequant"]).to(X.device)
    pooled = ((dq - X) ** 2).sum().item() / (X.pow(2).sum().item() + 1e-30)
    return {"candidate": name,
            "a4_nmse_mean": tok_nmse.mean().item(),
            "a4_nmse_median": tok_nmse.median().item(),
            "a4_nmse_p90": np.percentile(tok_nmse.numpy(), 90).item(),
            "a4_nmse_pooled": pooled,
            "sqnr_db": -10 * np.log10(pooled + 1e-30),
            "exact_zero_ratio": d["exact_zero_dequant_ratio"],
            "zero_code_ratio": d["zero_int_ratio"],
            "sat_ratio": d["sat_low"] + d["sat_high"],
            "code_entropy_marginal_bits": ent_marg,
            "code_entropy_per_token_bits": ent_tok,
            "unique_codes_per_token": uniq,
            "scale_median": float(np.median(scale)),
            "scale_p99": float(np.percentile(scale, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    torch.manual_seed(SEED)

    os.makedirs(f"{args.run_dir}/captures", exist_ok=True)
    samp_f = f"{args.run_dir}/captures/ht_sample.npz"
    if os.path.exists(samp_f):
        z = np.load(samp_f)
        Ht = torch.tensor(z["rows"], dtype=torch.float32, device=dev)
        seen = int(z["seen"])
        print(f"[ht] reusing sample {samp_f} ({Ht.shape[0]} tokens of "
              f"{seen})")
    else:
        Ht, seen = build_ht_sample(dev)
        np.savez_compressed(samp_f, rows=Ht.cpu().numpy().astype(np.float16),
                            seen=seen,
                            meta=json.dumps({"cap": CAP, "seed": SEED,
                                             "basis": "original_fp_path"}))
        print(f"[ht] sampled {Ht.shape[0]} of {seen} tokens -> {samp_f}")

    rows_s, rows_a = [], []
    for name, R in candidates(args.run_dir, dev).items():
        X = Ht if R is None else Ht @ R
        rows_s.append(stat_row(name, X))
        rows_a.append(a4_row(name, X))
        r = rows_s[-1]
        a = rows_a[-1]
        print(f"[{name:16s}] kurt={r['kurtosis']:9.2f} absmax="
              f"{r['absmax']:7.2f} top0.1%E={r['top0.1pct_energy']:.4f} "
              f"A4-NMSE(mean)={a['a4_nmse_mean']:.5f} "
              f"SQNR={a['sqnr_db']:5.2f}dB ent/tok="
              f"{a['code_entropy_per_token_bits']:.3f}b", flush=True)

    for fn, rows in [("ht_candidate_stats.csv", rows_s),
                     ("ht_a4_candidate_stats.csv", rows_a)]:
        with open(f"{args.run_dir}/tables/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote tables/{fn}")


if __name__ == "__main__":
    main()
