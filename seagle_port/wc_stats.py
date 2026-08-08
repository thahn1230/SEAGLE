"""Phase 3 instrumentation: W_c branch statistics + quantization audit (§5-6).

Captures, on calibration prompts (gsm8k train offset-500 pool, SEAGLE
convention), for each source branch i in [1,8,15,22,29]:
  activations H_i      : RMS, absmax, p99, p99.9, kurtosis, A4 zero-code ratio
  weights W_i          : RMS, absmax, per-row-scale share, W4 zero-code ratio
  contribution y_i     : RMS share of fc output
  naive-W4A4 output    : NMSE total + per-branch contribution NMSE
  branch deletion      : NMSE of fc output with branch i zeroed (FP)
Both bases: original (stock target) and rotated (H R1, valid per Gate B2).

Writes tables/wc_branch_stats.csv, tables/wc_quantization_stats.csv,
plots/wc_stats_arrays.npz.
"""
import argparse
import json
import os

import numpy as np
import torch

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils
from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]
D = 4096


def calib_prompts(n=24):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return ["Solve the following math problem step by step.\n\nQuestion: "
            + ds[i]["question"] + "\nAnswer:" for i in range(500, 500 + n)]


def act_stats(x):
    xf = x.float()
    flat = xf.flatten()
    return {
        "rms": xf.pow(2).mean().sqrt().item(),
        "absmax": xf.abs().max().item(),
        "p99": torch.quantile(flat.abs(), 0.99).item(),
        "p999": torch.quantile(flat.abs(), 0.999).item(),
        "kurtosis": ((xf - xf.mean()).pow(4).mean() /
                     (xf.var() ** 2 + 1e-12)).item(),
    }


def a4_fake(x, bits=4):
    q = quant_utils.ActQuantizer()
    q.configure(bits=bits, groupsize=-1, sym=False, clip_ratio=1.0)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).float()
    q.find_params(x2)
    out = q(x2)
    scale = q.scale.clone()
    q.free()
    zero_ratio = (out == 0).float().mean().item()
    return out.reshape(shp), zero_ratio, scale


def w4_fake(w, bits=4):
    q = quant_utils.WeightQuantizer()
    q.configure(bits, perchannel=True, sym=True, mse=True)
    wf = w.float()
    q.find_params(wf)
    dq = q.quantize(wf)
    zero_ratio = (dq == 0).float().mean().item()
    return dq, zero_ratio


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-prompts", type=int, default=24)
    args = ap.parse_args()
    dev = args.device

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    target = sq.build_target(MODEL, "fp16", device=dev)
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, dtype=torch.bfloat16).to(dev).eval()
    R1 = sq.load_rbin(args.rbin)["R1"].float()

    # gather H_i over calib prompts (prefill hiddens, teacher-forced)
    Hs = {l: [] for l in SRC}
    for p in calib_prompts(args.n_prompts):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=768).input_ids.to(dev)
        o = target(ids, output_hidden_states=True, use_cache=False)
        for l in SRC:
            Hs[l].append(o.hidden_states[l + 1][0].float().cpu())
    H = {l: torch.cat(v) for l, v in Hs.items()}          # [N, D]
    del target
    torch.cuda.empty_cache()

    W = draft.fc.weight.data.float().cpu()                 # [D, 5D]
    rows, qrows = [], []
    arrays = {}
    for basis in ("orig", "rot"):
        Hb = {l: (H[l] @ R1 if basis == "rot" else H[l]) for l in SRC}
        Wb = W.clone()
        if basis == "rot":
            for i in range(5):
                Wb[:, i * D:(i + 1) * D] = \
                    (W[:, i * D:(i + 1) * D].double() @ R1.double()).float()
        Hcat = torch.cat([Hb[l] for l in SRC], dim=-1)
        y_fp = Hcat @ Wb.t()
        y_fp_norm = y_fp.pow(2).mean()

        # per-token concat A4 (deployed quantizer granularity) + W4
        Hq_cat, az_cat, _ = a4_fake(Hcat)
        # branch-independent A4 (P2 proxy)
        Hq_branch = torch.cat([a4_fake(Hb[l])[0] for l in SRC], dim=-1)
        Wq, wz_all = w4_fake(Wb)
        y_naive = Hq_cat @ Wq.t()
        y_p2 = Hq_branch @ Wq.t()
        qrows.append(dict(
            basis=basis, arm="naive_concatA4_W4",
            out_nmse=((y_naive - y_fp).pow(2).mean() / y_fp_norm).item(),
            act_zero_ratio=az_cat, w_zero_ratio=wz_all))
        qrows.append(dict(
            basis=basis, arm="p2_branchA4_W4",
            out_nmse=((y_p2 - y_fp).pow(2).mean() / y_fp_norm).item(),
            act_zero_ratio=float("nan"), w_zero_ratio=wz_all))
        qrows.append(dict(
            basis=basis, arm="a4only_concat",
            out_nmse=((Hq_cat @ Wb.t() - y_fp).pow(2).mean()
                      / y_fp_norm).item(),
            act_zero_ratio=az_cat, w_zero_ratio=0.0))
        qrows.append(dict(
            basis=basis, arm="w4only",
            out_nmse=((Hcat @ Wq.t() - y_fp).pow(2).mean()
                      / y_fp_norm).item(),
            act_zero_ratio=0.0, w_zero_ratio=wz_all))

        for bi, l in enumerate(SRC):
            Wi = Wb[:, bi * D:(bi + 1) * D]
            hst = act_stats(Hb[l])
            _, az, _ = a4_fake(Hb[l])
            _, wz = w4_fake(Wi)
            y_i = Hb[l] @ Wi.t()
            # deletion: zero branch bi
            Hdel = Hcat.clone()
            Hdel[:, bi * D:(bi + 1) * D] = 0
            del_nmse = ((Hdel @ Wb.t() - y_fp).pow(2).mean()
                        / y_fp_norm).item()
            # per-branch contribution error under naive quant
            yq_i = Hq_cat[:, bi * D:(bi + 1) * D] @ \
                Wq[:, bi * D:(bi + 1) * D].t()
            rows.append(dict(
                basis=basis, layer=l,
                act_rms=hst["rms"], act_absmax=hst["absmax"],
                act_p99=hst["p99"], act_p999=hst["p999"],
                act_kurtosis=hst["kurtosis"],
                act_a4_zero_ratio=az,
                w_rms=Wi.pow(2).mean().sqrt().item(),
                w_absmax=Wi.abs().max().item(),
                w_w4_zero_ratio=wz,
                contrib_rms=y_i.pow(2).mean().sqrt().item(),
                deletion_nmse=del_nmse,
                branch_q_nmse=((yq_i - y_i).pow(2).mean()
                               / y_i.pow(2).mean()).item(),
            ))
            arrays[f"{basis}_absrow_H{l}"] = \
                Hb[l].abs().amax(0).numpy()[None, :]
    import csv
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "plots"), exist_ok=True)
    for fn, rws in (("wc_branch_stats.csv", rows),
                    ("wc_quantization_stats.csv", qrows)):
        with open(os.path.join(args.run_dir, "tables", fn), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader()
            w.writerows(rws)
    np.savez(os.path.join(args.run_dir, "plots", "wc_stats_arrays.npz"),
             **arrays)
    # console summary: imbalance ratios (§6)
    for basis in ("orig", "rot"):
        rs = [r for r in rows if r["basis"] == basis]
        arms = {r["layer"]: r for r in rs}
        rms = [r["act_rms"] for r in rs]
        wrms = [r["w_rms"] for r in rs]
        print(f"[{basis}] act RMS ratio max/min = {max(rms)/min(rms):.3f} "
              f"({ {l: round(arms[l]['act_rms'],2) for l in SRC} })")
        print(f"[{basis}] W_i RMS ratio max/min = {max(wrms)/min(wrms):.3f}")
    for q in qrows:
        print(q)
    print("[wc_stats] DONE")


if __name__ == "__main__":
    main()
