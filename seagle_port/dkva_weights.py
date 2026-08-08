"""DKVA §12-15: weight distributions + W4 quantization for every weight in
the DFlash draft dataflow.

Covers: W_c (stock / R_T-folded, full + 5 source blocks), all draft-layer
q/k/v/o/gate/up/down, ctx-specific K/V views (R_C OFF / R_C=R_T / learned),
shared embedding + lm_head (stats only), all norm gammas.

W4 uses the PROJECT quantizers: WeightQuantizer (per-channel sym, MSE clip)
for base weights — the deployed policy for fc/QKVO/MLP — and
rtn_sym_perchannel for ctx views (deployed RC policy). Both are reported.

Writes tables/weight_*.csv + raw/weights/*.npz (per-row/col arrays,
downsampled abs grids for 3D plots).
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from . import SPINQUANT_ROOT  # noqa: F401
from utils import quant_utils
from . import spinquant_target as sq
from . import interfaces
from .rc import rtn_sym_perchannel

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
LRN = "/home/thahn1230/dflash_workspace/outputs/rotations/llama31_w16a4kv16/R.bin"
PREV = "runs/dflash_seagle_transfer_20260807_180238"
SRC = [1, 8, 15, 22, 29]


def wstats(w):
    wf = w.float()
    absf = wf.abs().flatten().numpy()
    ctr = wf.flatten() - wf.mean()
    var = ctr.pow(2).mean()
    return {"rms": wf.pow(2).mean().sqrt().item(),
            "absmax": float(absf.max()),
            "mean_abs": float(absf.mean()),
            "std": var.sqrt().item(),
            "kurtosis": (ctr.pow(4).mean() / (var ** 2 + 1e-30)).item(),
            "p99_abs": float(np.percentile(absf, 99)),
            "p999_abs": float(np.percentile(absf, 99.9)),
            "max_over_rms": float(absf.max()) /
                            (wf.pow(2).mean().sqrt().item() + 1e-30)}


def axis_stats(w, dim):
    wf = w.float()
    a = wf.abs().numpy()
    return {"rms": wf.pow(2).mean(dim).sqrt(),
            "absmax": wf.abs().amax(dim),
            "p99": torch.from_numpy(
                np.percentile(a, 99, axis=dim).astype(np.float32))}


def w4_mse(w):
    q = quant_utils.WeightQuantizer()
    q.configure(4, perchannel=True, sym=True, mse=True)
    wf = w.float().cuda()
    q.find_params(wf)
    dq = q.quantize(wf)
    scale = q.scale.flatten().cpu()
    return dq.cpu(), scale


def quant_report(w, dq):
    wf, dqf = w.float(), dq.float()
    err = dqf - wf
    return {"w4_nmse": (err.pow(2).mean() / wf.pow(2).mean()).item(),
            "w4_cosine": torch.nn.functional.cosine_similarity(
                wf.flatten(), dqf.flatten(), dim=0).item(),
            "w4_zero_code_ratio": (dqf == 0).float().mean().item(),
            "w4_sat_ratio": float(((dqf.abs() -
                                    dqf.abs().amax(1, True)).abs() < 1e-12)
                                  .float().mean().item())}


def grid3d(w, step=16):
    a = w.float().abs()
    return a[::max(1, a.shape[0] // 256) if a.shape[0] > 4096 else step // 2,
             ::step][:512, :2048].numpy().astype(np.float16)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    od = args.run_dir
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sq.load_rbin(LRN)["R1"]
    folded = interfaces.fold_wc(draft, R1)
    RC1 = torch.load(f"{PREV}/rotations/RC1_reuseRT.pt",
                     weights_only=False)["R_C"].float()
    RCL = torch.load(f"{PREV}/rotations/RC_L0.pt",
                     weights_only=False)["R_C"].float()

    rows, rowaxis, colaxis, qrows = [], [], [], []
    raw = {}

    def emit(name, w, quantizer="mseclip"):
        st = wstats(w)
        st["name"] = name
        st["shape"] = "x".join(map(str, w.shape))
        st["quantizer"] = quantizer
        if quantizer == "mseclip":
            dq, scale = w4_mse(w)
        else:
            dq = rtn_sym_perchannel(w.float(), 4)
            scale = w.float().abs().amax(1) / 7
        st.update(quant_report(w, dq))
        rows.append(st)
        for dim, buf in ((1, rowaxis), (0, colaxis)):
            ax = axis_stats(w, dim)
            buf.append({"name": name,
                        **{f"{k}_mean": float(v.mean()) for k, v in
                           ax.items()},
                        **{f"{k}_max": float(v.max()) for k, v in
                           ax.items()}})
            raw[f"{name}__axis{dim}_rms"] = ax["rms"].numpy().astype(
                np.float32)
            raw[f"{name}__axis{dim}_absmax"] = ax["absmax"].numpy().astype(
                np.float32)
        raw[f"{name}__scale"] = np.asarray(scale, dtype=np.float32)
        raw[f"{name}__grid"] = grid3d(w)
        raw[f"{name}__grid_q"] = grid3d(dq)
        print("[w]", name, "rms %.4f absmax %.3f nmse %.5f" %
              (st["rms"], st["absmax"], st["w4_nmse"]), flush=True)

    # A. W_c
    for tag, m in (("Wc_stock", draft), ("Wc_foldedRT", folded)):
        W = m.fc.weight.data
        emit(tag, W)
        for bi, l in enumerate(SRC):
            emit(f"{tag}_blk{l}", W[:, bi * 4096:(bi + 1) * 4096])
    # B. draft layers
    for i, layer in enumerate(draft.layers):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            emit(f"l{i}_{nm}", getattr(layer.self_attn, nm).weight.data)
        for nm in ("gate_proj", "up_proj", "down_proj"):
            emit(f"l{i}_{nm}", getattr(layer.mlp, nm).weight.data)
    # C. ctx views
    for i, layer in enumerate(draft.layers):
        for nm in ("k_proj", "v_proj"):
            W = getattr(layer.self_attn, nm).weight.data.float()
            emit(f"l{i}_{nm}_ctxview_RCoff", W, quantizer="rtn")
            emit(f"l{i}_{nm}_ctxview_RC1", W @ RC1, quantizer="rtn")
            emit(f"l{i}_{nm}_ctxview_RCL", W @ RCL, quantizer="rtn")
    # D. shared boundaries (stats only, no quant — kept bf16 in deploy)
    tgt = sq.load_target(MODEL, device="cpu")
    emb = tgt.model.embed_tokens.weight.data
    head = tgt.lm_head.weight.data
    tied = bool(emb.data_ptr() == head.data_ptr())
    for nm, W in (("shared_embedding", emb), ("shared_lm_head", head)):
        st = wstats(W)
        st.update({"name": nm, "shape": "x".join(map(str, W.shape)),
                   "quantizer": "none(bf16 deployed)", "w4_nmse": "",
                   "w4_cosine": "", "w4_zero_code_ratio": "",
                   "w4_sat_ratio": ""})
        rows.append(st)
    del tgt
    # E. norms
    norms = {"hidden_norm": draft.hidden_norm.weight,
             "final_norm": draft.norm.weight}
    for i, layer in enumerate(draft.layers):
        norms[f"l{i}_input_ln"] = layer.input_layernorm.weight
        norms[f"l{i}_post_ln"] = layer.post_attention_layernorm.weight
        norms[f"l{i}_q_norm"] = layer.self_attn.q_norm.weight
        norms[f"l{i}_k_norm"] = layer.self_attn.k_norm.weight
    for nm, g in norms.items():
        raw[f"norm__{nm}"] = g.data.float().numpy().astype(np.float32)

    os.makedirs(f"{od}/tables", exist_ok=True)
    for fn, rws in (("weight_global_stats.csv", rows),
                    ("weight_row_stats.csv", rowaxis),
                    ("weight_column_stats.csv", colaxis)):
        keys = sorted({k for r in rws for k in r})
        with open(f"{od}/tables/{fn}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rws)
    np.savez_compressed(f"{od}/raw/weights/weights_arrays.npz", **raw)
    json.dump({"embedding_head_tied": tied},
              open(f"{od}/tables/weight_meta.json", "w"))
    print(f"[dkva_weights] DONE {len(rows)} weights, tied={tied}")


if __name__ == "__main__":
    main()
