"""Phase 4: Multi-P3 (MP3) source-wise scale calibration (§11).

m_i = D^{beta_i}, gauge mean(beta)=0 (geomean(m)=1). Proxy objective:
fc-output NMSE under deployed quantizer granularity (concat per-token asym
A4 + per-row sym MSE-clip W4 on the scaled/folded weight). Search:
  1. init A: beta=0; init B: RMS-equalized (m_i ∝ 1/rms(H_i), gauged)
  2. coordinate descent around the better init (deltas 0.06/0.03/0.015,
     2 sweeps each) on calib half; report NMSE on held-out half.
Emits every evaluated point to tables/mp3_calibration.csv and the top-3
gauged candidates to tables/mp3_candidates.json (validation-AL jobs pick
these up).

--basis rot uses H_i @ R1 and the R1-folded fc (Gate B2-verified identity);
--basis orig uses raw hiddens and stock fc.
"""
import argparse
import csv
import json
import os

import torch

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils
from . import spinquant_target as sq
from .wc_stats import calib_prompts, a4_fake, w4_fake

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]
D = 4096


@torch.inference_mode()
def gather(dev, n, rbin, basis):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, "fp16", device=dev)
    Hs = {l: [] for l in SRC}
    for p in calib_prompts(n):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=768).input_ids.to(dev)
        o = target(ids, output_hidden_states=True, use_cache=False)
        for l in SRC:
            Hs[l].append(o.hidden_states[l + 1][0].float().cpu())
    del target
    torch.cuda.empty_cache()
    draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    W = draft.fc.weight.data.float()
    del draft
    H = {l: torch.cat(v) for l, v in Hs.items()}
    R1 = sq.load_rbin(rbin)["R1"].float()
    if basis == "rot":
        H = {l: H[l] @ R1 for l in SRC}
        W2 = W.clone()
        for i in range(5):
            W2[:, i * D:(i + 1) * D] = \
                (W[:, i * D:(i + 1) * D].double() @ R1.double()).float()
        W = W2
    return H, W


@torch.inference_mode()
def nmse_for_beta(Hcat_dev, W_dev, beta, y_fp, y_norm):
    m = torch.pow(torch.tensor(float(D)), beta - beta.mean()).to(Hcat_dev.device)
    x = Hcat_dev.reshape(-1, 5, D) * m.view(1, 5, 1)
    xq, _, _ = a4_fake(x.reshape(-1, 5 * D))
    Wm = W_dev.clone().reshape(-1, 5, D) / m.view(1, 5, 1)
    Wq, _ = w4_fake(Wm.reshape(-1, 5 * D))
    y = xq @ Wq.t()
    return ((y - y_fp).pow(2).mean() / y_norm).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--basis", default="rot", choices=["orig", "rot"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-prompts", type=int, default=24)
    args = ap.parse_args()
    dev = args.device
    H, W = gather(dev, args.n_prompts, args.rbin, args.basis)
    Hcat = torch.cat([H[l] for l in SRC], dim=-1)
    n = Hcat.shape[0]
    tr, te = Hcat[: n // 2].to(dev), Hcat[n // 2:].to(dev)
    W_dev = W.to(dev)
    rows = []

    def evaluate(tag, beta):
        b = beta - beta.mean()
        r = {"tag": tag, "basis": args.basis,
             **{f"beta{i}": round(b[i].item(), 5) for i in range(5)}}
        for split, X in (("calib", tr), ("holdout", te)):
            y_fp = X @ W_dev.t()
            r[f"nmse_{split}"] = nmse_for_beta(X, W_dev, b, y_fp,
                                               y_fp.pow(2).mean())
        rows.append(r)
        print(r, flush=True)
        return r["nmse_calib"]

    beta0 = torch.zeros(5)
    evaluate("naive_beta0", beta0)
    rms = torch.tensor([H[l].pow(2).mean().sqrt() for l in SRC])
    beta_rms = -torch.log(rms) / torch.log(torch.tensor(float(D)))
    evaluate("rms_equalized", beta_rms)

    best = min(rows, key=lambda r: r["nmse_calib"])
    beta = torch.tensor([best[f"beta{i}"] for i in range(5)])
    cur = best["nmse_calib"]
    for delta in (0.06, 0.03, 0.015):
        for sweep in range(2):
            improved = False
            for i in range(5):
                for sgn in (1, -1):
                    cand = beta.clone()
                    cand[i] += sgn * delta
                    v = evaluate(f"cd_d{delta}_s{sweep}_b{i}{'+' if sgn>0 else '-'}",
                                 cand)
                    if v < cur - 1e-6:
                        beta, cur, improved = cand - cand.mean(), v, True
            if not improved:
                break

    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables",
                           f"mp3_calibration_{args.basis}.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    uniq = {}
    for r in sorted(rows, key=lambda r: r["nmse_holdout"]):
        key = tuple(round(r[f"beta{i}"], 4) for i in range(5))
        if key not in uniq:
            uniq[key] = r
        if len(uniq) == 3:
            break
    top = list(uniq.values())
    mfin = [[float(D) ** r[f"beta{i}"] for i in range(5)] for r in top]
    json.dump({"basis": args.basis, "top3": top, "m_top3": mfin,
               "final_beta": [round(b, 5) for b in
                              [top[0][f"beta{i}"] for i in range(5)]],
               "final_m": mfin[0]},
              open(os.path.join(args.run_dir, "tables",
                                f"mp3_candidates_{args.basis}.json"), "w"),
              indent=1)
    print("[mp3] DONE best holdout NMSE:", top[0]["nmse_holdout"],
          "m:", [round(x, 4) for x in mfin[0]])


if __name__ == "__main__":
    main()
