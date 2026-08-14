"""R1DCE §8 — ctx K/V activation/weight error decomposition per candidate,
plus the §19 gamma-contract / weight-view-unification audit.

Uses the token-paired H_t sample from r1dce_ht_stats (captures/ht_sample.npz).
For each candidate rotation R and each draft layer:
    X = H_t @ R,  W_K = (W_k γ_hid) @ R,  W_V = R2out(W_v γ_hid) @ R
    FP    = X W^T          A-only = Q_A4(X) W^T
    W-only= X Q_W4(W)^T    A+W    = Q_A4(X) Q_W4(W)^T
NMSE denominators use the candidate-local FP output (== stock FP output up
to fp32 rounding; parity recorded per row).  Writes
tables/ctx_kv_aw_decomposition.csv, tables/ctx_kv_layerwise_nmse.csv,
tables/weight_view_unification.json.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .rc import rtn_sym_perchannel, act_fake_ste
from .r1dce_ht_stats import candidates

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"


def nmse(y, ref):
    return (((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-12)).item()


def cosf(y, ref):
    return torch.nn.functional.cosine_similarity(
        y.flatten(), ref.flatten(), dim=0).item()


def verdict(a, w, aw):
    if a >= 4 * w and aw < 2 * a:
        return "activation_driven"
    if w >= 4 * a:
        return "weight_driven"
    if aw > 2 * (a + w):
        return "interaction"
    return "mixed"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft

    z = np.load(f"{args.run_dir}/captures/ht_sample.npz")
    Ht = torch.tensor(z["rows"], dtype=torch.float32, device=dev)
    print(f"[kv] H_t sample {tuple(Ht.shape)}")

    d0 = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    R1T = sq.load_rbin(S1RBIN)["R1"].float().to(dev)
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1D = ck["R1_D"].float().to(dev)
    R2D = [t.float().to(dev) for t in ck["R2_D"]]
    base = interfaces.fold_wc(d0, R1T)
    rq = RotQuantDraft(base, w_bits=16, a_bits=16, use_r2=True,
                       train_rotations=False, device=dev)
    rq.R1 = lambda: R1D
    rq.R2 = lambda i: R2D[i]

    # ---- §19 gamma-contract audit
    gam_h = d0.hidden_norm.weight.data.float()
    g19 = {"gamma_ctx=hidden_norm": {
        "mean": gam_h.mean().item(), "std": gam_h.std().item()}}
    per = []
    for i in range(rq.n_layers):
        gi = d0.layers[i].input_layernorm.weight.data.float()
        per.append({
            "layer": i,
            "cos(gamma_ctx,gamma_draft)": torch.nn.functional
            .cosine_similarity(gam_h, gi, dim=0).item(),
            "rel_l2_diff": ((gam_h - gi).norm() / gi.norm()).item(),
            "ratio_min": (gam_h / gi).min().item(),
            "ratio_max": (gam_h / gi).max().item()})
        wk_ctx = getattr(rq, f"wk_ctx_{i}")
        wk_noise = getattr(rq, f"wk_noise_{i}")
        per[-1]["ctx_vs_draft_K_view_rel_diff_postR1D"] = (
            ((wk_ctx @ R1D) - (wk_noise @ R1D)).norm()
            / (wk_noise @ R1D).norm()).item()
    g19["per_layer"] = per
    g19["same_basis_under_C1"] = True
    g19["same_gamma"] = False
    g19["same_exact_tensor"] = False
    g19["view_sharing_possible"] = ("NO — W_K,ctx=(W_K D_gamma_hid)R1_D vs "
                                    "W_K,draft=(W_K D_gamma_in,i)R1_D differ "
                                    "by the diagonal gamma contract; equal "
                                    "only if gamma_hid==gamma_in,i")
    nb = sum(getattr(rq, f"wk_ctx_{i}").numel() +
             getattr(rq, f"wv_ctx_{i}").numel()
             for i in range(rq.n_layers)) * 2   # bf16 deployed
    g19["ctx_view_memory_bytes_bf16"] = nb
    json.dump(g19, open(f"{args.run_dir}/tables/"
                        "weight_view_unification.json", "w"), indent=1)
    print(f"[§19] gamma cos(ctx,draft) per layer: "
          f"{[round(p['cos(gamma_ctx,gamma_draft)'], 4) for p in per]}")

    # ---- §8 quadrants
    cands = candidates(args.run_dir, dev)
    rows = []
    for name, R in cands.items():
        X = Ht if R is None else Ht @ R
        Xq = act_fake_ste(X, 4)
        for i in range(rq.n_layers):
            wk = getattr(rq, f"wk_ctx_{i}").float()
            wv = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                              R2D[i], "out")
            if R is not None:
                wk = wk @ R
                wv = wv @ R
            for proj, W in (("K", wk), ("V", wv)):
                fp = X @ W.t()
                ref0 = Ht @ (getattr(rq, f"wk_ctx_{i}").float().t()
                             if proj == "K" else
                             rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                                          R2D[i], "out").t())
                a_ = Xq @ W.t()
                w_ = X @ rtn_sym_perchannel(W, 4).t()
                aw = Xq @ rtn_sym_perchannel(W, 4).t()
                an, wn, awn = nmse(a_, fp), nmse(w_, fp), nmse(aw, fp)
                rows.append({
                    "candidate": name, "layer": i, "proj": proj,
                    "fp_parity_rel": ((fp - ref0).abs().max()
                                      / (ref0.abs().max() + 1e-30)).item(),
                    "A_only_nmse": an, "W_only_nmse": wn, "AW_nmse": awn,
                    "interaction_nmse": awn - an - wn,
                    "AW_cosine": cosf(aw, fp),
                    "AW_sqnr_db": -10 * np.log10(awn + 1e-30),
                    "max_abs_err_AW": (aw - fp).abs().max().item(),
                    "verdict": verdict(an, wn, awn),
                    "n_rows": X.shape[0]})
        k = [r for r in rows if r["candidate"] == name and r["proj"] == "K"]
        v = [r for r in rows if r["candidate"] == name and r["proj"] == "V"]
        print(f"[{name:16s}] K AW-NMSE mean="
              f"{np.mean([r['AW_nmse'] for r in k]):.5f} "
              f"V AW-NMSE mean={np.mean([r['AW_nmse'] for r in v]):.5f} "
              f"(A-only K {np.mean([r['A_only_nmse'] for r in k]):.5f})",
              flush=True)

    with open(f"{args.run_dir}/tables/ctx_kv_aw_decomposition.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # layerwise AW-NMSE pivot for §11 diagnosis
    with open(f"{args.run_dir}/tables/ctx_kv_layerwise_nmse.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "proj"] + [f"L{i}" for i in range(5)])
        for name in cands:
            for proj in ("K", "V"):
                sel = [r["AW_nmse"] for r in rows
                       if r["candidate"] == name and r["proj"] == proj]
                w.writerow([name, proj] + [f"{x:.6f}" for x in sel])
    print("wrote tables/ctx_kv_aw_decomposition.csv + "
          "ctx_kv_layerwise_nmse.csv")


if __name__ == "__main__":
    main()
