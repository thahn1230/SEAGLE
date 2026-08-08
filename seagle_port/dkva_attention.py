"""DKVA §20: attention-geometry diagnostic (pre-attention error tracking).

Fixed subsample (8 prompts x first 2 cycles, mtbench+gsm8k). For each draft
layer, replay the SAME cycle through three paths and compare against the
C1-equivalent FP reference (rotated target, fp16 draft weights):

  ref    : FP16 ctx/noise K,V (post k_norm + RoPE), FP16 Q
  w4a4   : C2 path (QLinear-quantized fc/K/V, R_C OFF)
  rc1    : C3 path (RCContextKV views, R_C = R_T)

Metrics per layer: K_ctx / V_ctx NMSE (post-norm+RoPE), attention score
S = QK^T/sqrt(d) NMSE + cosine, softmax KL/TV, top-1 attended-position
agreement. Attention itself runs in bf16 in ALL arms — recorded explicitly:
the error exists BEFORE attention.

Writes tables/attention_error.csv.
"""
import argparse
import csv
import json
import os

import torch

from . import spinquant_target as sq
from . import interfaces
from .rc import load_rc_draft

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
LRN = "/home/thahn1230/dflash_workspace/outputs/rotations/llama31_w16a4kv16/R.bin"
PREV = "runs/dflash_seagle_transfer_20260807_180238"
SRC = [1, 8, 15, 22, 29]


def kv_for(layer, x_ctx, x_noise, pos_emb, head_dim):
    """K (post k_norm + RoPE) and V for given ctx/noise inputs."""
    from dflash.model import apply_rotary_pos_emb
    att = layer.self_attn
    k = torch.cat([att.k_proj(x_ctx), att.k_proj(x_noise)], dim=1)
    v = torch.cat([att.v_proj(x_ctx), att.v_proj(x_noise)], dim=1)
    b, s = k.shape[:2]
    k = att.k_norm(k.view(b, s, -1, head_dim)).transpose(1, 2)
    v = v.view(b, s, -1, head_dim).transpose(1, 2)
    cos_, sin_ = pos_emb
    kr = (k * cos_.unsqueeze(1)) + (
        _rot_half(k) * sin_.unsqueeze(1))
    return kr, v


def _rot_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def nmse(a, b):
    return ((a.float() - b.float()).pow(2).mean() /
            (b.float().pow(2).mean() + 1e-12)).item()


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
    from dflash.model import DFlashDraftModel, apply_rotary_pos_emb
    from . import draft_quant

    target = sq.build_target(MODEL, "w4a4", rbin_path=LRN, device=dev)
    base_fp = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    base_fp = interfaces.fold_wc(base_fp, sq.load_rbin(LRN)["R1"])
    q_draft = draft_quant.quantize_draft(base_fp, 4, 4,
                                        fc_branch_dims=[4096] * 5)
    rc1 = load_rc_draft(q_draft, base_fp,
                        rc_ckpt=f"{PREV}/rotations/RC1_reuseRT.pt").to(dev)
    hd = 128

    rows = []
    for ds in ("mtbench", "gsm8k"):
        cyc = [json.loads(l) for l in open(
            f"{rd}/cycles/cyc__DKVA_C0__{ds}.jsonl")]
        turns, order = {}, []
        for r in cyc:
            k = (r["prompt_id"], r["turn"])
            if r.get("type") == "turn_header":
                turns[k] = {"ids": r["input_ids"], "cycles": []}
                order.append(k)
            else:
                turns[k]["cycles"].append(r)
        for key in order[:8]:
            t = turns[key]
            flat = list(t["ids"])
            for c in t["cycles"]:
                flat += c["block"][:c["tau"]]
            full = torch.tensor([flat], device=dev)
            out = target(full, output_hidden_states=True, use_cache=False)
            Hcat = torch.cat([out.hidden_states[l + 1][0] for l in SRC],
                             dim=-1)
            del out
            for c in t["cycles"][:2]:
                pl = c["prefix_len"]
                th = Hcat[:pl].unsqueeze(0).to(torch.bfloat16)
                block = torch.tensor([c["block"]], device=dev)
                ne = target.model.embed_tokens(block)
                B = block.shape[1]
                pos = torch.arange(pl + B, device=dev).unsqueeze(0)
                pe = base_fp.rotary_emb(ne, pos)

                # shared FP draft-side stream (isolate ctx-path effects)
                hs = ne
                for li, layer in enumerate(base_fp.layers):
                    x_noise = layer.input_layernorm(hs)
                    Ht_fp = base_fp.hidden_norm(base_fp.fc(th))
                    Ht_q = q_draft.hidden_norm(q_draft.fc(th))
                    att = layer.self_attn
                    q = att.q_norm(att.q_proj(x_noise).view(
                        1, B, -1, hd)).transpose(1, 2)
                    cos_, sin_ = pe
                    q = (q * cos_[..., -B:, :].unsqueeze(1)) + (
                        _rot_half(q) * sin_[..., -B:, :].unsqueeze(1))

                    K_ref, V_ref = kv_for(layer, Ht_fp, x_noise, pe, hd)
                    qatt = q_draft.layers[li].self_attn
                    Kq = torch.cat([qatt.k_proj(Ht_q),
                                    qatt.k_proj(x_noise)], dim=1)
                    Vq = torch.cat([qatt.v_proj(Ht_q),
                                    qatt.v_proj(x_noise)], dim=1)
                    Kq = att.k_norm(Kq.view(1, pl + B, -1, hd)).transpose(
                        1, 2)
                    Kq = (Kq * cos_.unsqueeze(1)) + (
                        _rot_half(Kq) * sin_.unsqueeze(1))
                    Vq = Vq.view(1, pl + B, -1, hd).transpose(1, 2)
                    kc, vc = rc1.ctx_kv[li](Ht_fp, rc1.rc_matrix().to(dev))
                    Krc = torch.cat([kc, qatt.k_proj(x_noise)], dim=1)
                    Vrc = torch.cat([vc, qatt.v_proj(x_noise)], dim=1)
                    Krc = att.k_norm(Krc.view(1, pl + B, -1, hd)).transpose(
                        1, 2)
                    Krc = (Krc * cos_.unsqueeze(1)) + (
                        _rot_half(Krc) * sin_.unsqueeze(1))
                    Vrc = Vrc.view(1, pl + B, -1, hd).transpose(1, 2)

                    def score(K):
                        Ke = K.repeat_interleave(4, dim=1)
                        return (q.float() @ Ke.float().transpose(-1, -2)
                                ) / hd ** 0.5
                    S_ref = score(K_ref)
                    for arm, K, V in (("w4a4", Kq, Vq), ("rc1", Krc, Vrc)):
                        S = score(K)
                        P_ref = torch.softmax(S_ref, -1)
                        P = torch.softmax(S, -1)
                        kl = (P_ref * ((P_ref + 1e-9).log()
                                       - (P + 1e-9).log())).sum(-1)
                        rows.append({
                            "ds": ds, "prompt": key[0], "cycle_prefix": pl,
                            "layer": li, "arm": arm,
                            "K_ctx_nmse": nmse(K[:, :, :pl],
                                               K_ref[:, :, :pl]),
                            "V_ctx_nmse": nmse(V[:, :, :pl],
                                               V_ref[:, :, :pl]),
                            "score_nmse": nmse(S, S_ref),
                            "softmax_kl_mean": kl.mean().item(),
                            "softmax_tv_mean": 0.5 * (P - P_ref).abs()
                                .sum(-1).mean().item(),
                            "top1_agree": (S.argmax(-1) ==
                                           S_ref.argmax(-1)).float()
                                .mean().item(),
                            "attention_dtype": "bf16 (all arms)"})
                    # advance shared FP stream via reference layer forward
                    hs = _layer_fp(layer, hs, Ht_fp, pe, pl, B)
            del Hcat
            torch.cuda.empty_cache()

    with open(f"{rd}/tables/attention_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[attention] DONE", len(rows), "rows")


def _layer_fp(layer, hs, Ht, pe, pl, B):
    from .rc import _layer_forward_ctxkv
    att = layer.self_attn
    k_ctx = att.k_proj(Ht)
    v_ctx = att.v_proj(Ht)
    return _layer_forward_ctxkv(layer, hs, k_ctx, v_ctx, None, None,
                                False, pe, {})


if __name__ == "__main__":
    main()
