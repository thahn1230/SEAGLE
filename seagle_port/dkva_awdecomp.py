"""DKVA §16-17: activation x weight error decomposition for K/V projections.

For every draft layer i and X in {H_t (C2 reservoir), H_t@R_C (C3
reservoir), H_d layer-i (C2 reservoir)}:

  Y_fp = X W^T          Y_A = Q_A4(X) W^T
  Y_W  = X Q_W4(W)^T    Y_AW = Q_A4(X) Q_W4(W)^T

with the weight quantized under (a) deployed policy for that branch
(mseclip on W for RC-off ctx/draft; RTN on W@R_C for RC-on views) and
(b) an RTN-on-W control so the clip-search difference cannot masquerade
as the R_C effect. H0/H1 (§17): qparams are computed per token per branch
call (independent) — the comparison quantifies within-token difficulty.

Writes tables/aw_error_decomposition.csv + tables/kv_projection_error.csv.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils
from .rc import rtn_sym_perchannel
from .dkva_capture import act_quant_detail

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
PREV = "runs/dflash_seagle_transfer_20260807_180238"


def load_rows(rd, cfg, tensor, cap=8000):
    rows = []
    for p in sorted(glob.glob(
            f"{rd}/raw/activations/{cfg}__*__{tensor}.npz")):
        rows.append(np.load(p)["rows"])
    if not rows:
        return None
    x = np.concatenate(rows)
    rng = np.random.default_rng(0)
    if len(x) > cap:
        x = x[rng.choice(len(x), cap, replace=False)]
    return torch.from_numpy(x).float()


def w4_mseclip(w):
    q = quant_utils.WeightQuantizer()
    q.configure(4, perchannel=True, sym=True, mse=True)
    wf = w.float().cuda()
    q.find_params(wf)
    return q.quantize(wf).cpu()


def nmse(a, b):
    return ((a - b).pow(2).mean() / (b.pow(2).mean() + 1e-12)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rd = args.run_dir
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    RC1 = torch.load(f"{PREV}/rotations/RC1_reuseRT.pt",
                     weights_only=False)["R_C"].float()

    Ht = load_rows(rd, "C2", "C_Ht")
    Htr = load_rows(rd, "C3", "C_Ht_rot")
    assert Ht is not None and Htr is not None, "capture shards missing"
    rows = []
    for li in range(5):
        Hd = load_rows(rd, "C2", f"D_draft_k_l{li}")
        att = draft.layers[li].self_attn
        for pname, proj in (("K", att.k_proj), ("V", att.v_proj)):
            W = proj.weight.data.float()
            Wr = (W.double() @ RC1.double()).float()
            wq = {"mseclip_W": w4_mseclip(W),
                  "rtn_W": rtn_sym_perchannel(W, 4),
                  "rtn_WRC": rtn_sym_perchannel(Wr, 4)}
            branches = [
                ("ctx_RCoff", Ht, W, "mseclip_W"),
                ("ctx_RCoff_rtnctl", Ht, W, "rtn_W"),
                ("ctx_RC1", Htr, Wr, "rtn_WRC"),
                ("draft", Hd, W, "mseclip_W"),
            ]
            for bname, X, Wfp, wkey in branches:
                if X is None:
                    continue
                Xd = X.to(dev)
                Wd = Wfp.to(dev)
                Wq = wq[wkey].to(dev)
                Xq = act_quant_detail(Xd)["dequant"]
                Yfp = Xd @ Wd.t()
                r = {"layer": li, "proj": pname, "branch": bname,
                     "weight_quantizer": wkey, "n_rows": len(X),
                     "A_only_nmse": nmse(Xq @ Wd.t(), Yfp),
                     "W_only_nmse": nmse(Xd @ Wq.t(), Yfp),
                     "AW_nmse": nmse(Xq @ Wq.t(), Yfp),
                     "AW_cosine": cos(Xq @ Wq.t(), Yfp),
                     "out_rms_fp": Yfp.pow(2).mean().sqrt().item()}
                r["interaction"] = (r["AW_nmse"] - r["A_only_nmse"]
                                    - r["W_only_nmse"])
                rows.append(r)
                print(r, flush=True)
                del Xd, Wd, Wq, Xq, Yfp
                torch.cuda.empty_cache()
    with open(f"{rd}/tables/aw_error_decomposition.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # compact kv projection error table (deployed policies only)
    dep = [r for r in rows if r["weight_quantizer"] != "rtn_W"]
    with open(f"{rd}/tables/kv_projection_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dep[0].keys()))
        w.writeheader(); w.writerows(dep)
    # H0/H1 verdict material: independent qparams by construction;
    # summarize ctx-vs-draft AW gap under deployed policy
    verdict = {}
    for li in range(5):
        c = next(r for r in dep if r["layer"] == li and r["proj"] == "K"
                 and r["branch"] == "ctx_RCoff")
        d = next(r for r in dep if r["layer"] == li and r["proj"] == "K"
                 and r["branch"] == "draft")
        verdict[f"l{li}_K_ctx_over_draft_AW"] = c["AW_nmse"] / \
            max(d["AW_nmse"], 1e-12)
    json.dump({"qparams_shared": False,
               "note": "per-token qparams computed independently per call "
                       "site (H0 rejected by construction + measurement)",
               "ctx_over_draft_ratio": verdict},
              open(f"{rd}/tables/h0h1_verdict.json", "w"), indent=1)
    print("[awdecomp] DONE", len(rows), "rows")


if __name__ == "__main__":
    main()
