#!/usr/bin/env python
"""Task 5: optimize an EAGLE-AWARE R1.

Skeptical framing: in the EXACT fp16 pure-R1 design, any orthogonal R1 gives the
SAME (exact) interface, so acceptance is R1-INVARIANT in fp16 — R1 is a gauge.
R1 only matters once QUANTIZATION is added, where it shapes how well the rotated
hidden quantizes AND how well the quantized hidden still drives the draft. This
script optimizes R1 for exactly that, on cached final-layer hidden activations.

Objective (cached final hidden h from the fp16 target; frozen draft):
  L = L_target_quant  : W4A4 fake-quant rel error of h_R = h @ R1
    + lam_kurt * mean channel excess-kurtosis of h_R   (outlier flatness)
    + lam_draft * draft-feature drift ||f(Q(h_R)) - f(h_R)|| (EAGLE-aware)

R1 is optimized on the Stiefel manifold via QR-retraction (R1_orth = qr(P)[0],
backprop through qr), Adam. Only R1 is changed; R2/R3 are copied from the
random-Hadamard R.bin so the full SpinQuant pipeline still runs.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/train_eagle_aware_r1.py \
      --steps 150 --num-seq 32 --seqlen 128 --out-name eagle_aware
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import experiment, rotation_aware as ra, study  # noqa: E402

DEV = "cuda:0"


def fake_quant_ste(x, bits=4):
    """Per-token symmetric W4A4-style activation fake-quant with straight-through
    round. s = max|x|/qmax per row."""
    qmax = 2 ** (bits - 1) - 1
    s = (x.detach().abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(x / s), -qmax - 1, qmax)
    q_ste = x / s + (q - x / s).detach()               # straight-through
    return q_ste * s


def excess_kurtosis(x):
    m = x.mean(0, keepdim=True)
    xc = x - m
    var = xc.pow(2).mean(0)
    k = (xc.pow(4).mean(0) / (var.pow(2) + 1e-8)) - 3.0
    return k.mean()


@torch.no_grad()
def cache_hidden(paths, cfg, num_seq, seqlen, rotations_root):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    tok = AutoTokenizer.from_pretrained(paths["target_path"], use_fast=False)
    m = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                low_cpu_mem_usage=True).to(DEV).eval()
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids
    H, TOK = [], []
    for i in range(num_seq):
        c = ids[:, i * seqlen:(i + 1) * seqlen]
        if c.shape[1] < seqlen:
            break
        h = m.model(input_ids=c.to(DEV))[0][0]          # [seqlen, 4096] original h
        H.append(h.float().cpu()); TOK.append(c[0].cpu())
    W_lm = m.lm_head.weight.detach().float().cpu()
    del m; torch.cuda.empty_cache()
    return torch.cat(H, 0), W_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--num-seq", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-kurt", type=float, default=0.05)
    ap.add_argument("--lam-draft", type=float, default=1.0)
    ap.add_argument("--out-name", default="eagle_aware")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    torch.manual_seed(0)
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")

    print(f"[r1] caching {args.num_seq}x{args.seqlen} hidden vectors...", flush=True)
    H, W_lm = cache_hidden(paths, cfg, args.num_seq, args.seqlen, rotations_root)
    H = H.to(DEV)                                        # [N, 4096]
    print(f"[r1] cached {H.shape}", flush=True)

    # frozen draft (for the EAGLE-aware term): use its fc + layer as f(.)
    draft = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    for p in draft.parameters():
        p.requires_grad = False
    D = 4096

    # init R1 from random-Hadamard R.bin (keep its R2/R3 for the saved R.bin)
    base_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                                rotations_root)
    R = torch.load(base_bin, map_location="cpu", weights_only=False)
    R1_init = R["R1"].to(DEV).float()
    P = torch.nn.Parameter(R1_init.clone())
    opt = torch.optim.Adam([P], lr=args.lr)

    # a fixed token context for the draft term (dummy ids; the draft's fc input
    # is dominated by h, embeddings are frozen). Use a small batch of rows.
    N = H.shape[0]
    idx = torch.randperm(N, device=DEV)[:2048]
    Hs = H[idx]                                          # calibration hidden rows

    def draft_first_feat(hid):
        # single fc forward: fc(cat([e=0, hid])) -> act -> layer0 (approx via fc
        # only, cheap & differentiable). We use the fc+MLP path signal: fc output.
        z = torch.cat([torch.zeros_like(hid), hid], -1)
        return draft.fc(z)                              # [n,4096], depends on R1 via hid

    log = []
    for step in range(args.steps):
        R1 = torch.linalg.qr(P)[0]                       # orthogonal retraction
        h_R = Hs @ R1                                    # rotated hidden
        hq = fake_quant_ste(h_R)
        l_quant = ((hq - h_R).norm() / (h_R.norm() + 1e-8))
        l_kurt = excess_kurtosis(h_R)
        f_clean = draft_first_feat(h_R)
        f_q = draft_first_feat(hq)
        l_draft = ((f_q - f_clean).norm() / (f_clean.norm() + 1e-8))
        loss = l_quant + args.lam_kurt * l_kurt.abs() + args.lam_draft * l_draft
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 15 == 0 or step == args.steps - 1:
            row = dict(step=step, loss=loss.item(), l_quant=l_quant.item(),
                       l_kurt=l_kurt.item(), l_draft=l_draft.item())
            log.append(row)
            print(f"  step {step:4d} loss={loss.item():.4f} quant={l_quant.item():.4f} "
                  f"kurt={l_kurt.item():.3f} draft={l_draft.item():.4f}", flush=True)

    with torch.no_grad():
        R1_final = torch.linalg.qr(P)[0].cpu()
    orth_err = (R1_final.double().t() @ R1_final.double()
                - torch.eye(D, dtype=torch.float64)).abs().max().item()
    out_dir = os.path.join(rotations_root or os.path.join(PROJECT_ROOT, "outputs", "rotations"),
                           args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    R_out = dict(R)
    R_out["R1"] = R1_final                               # replace R1, keep R2/R3
    torch.save(R_out, os.path.join(out_dir, "R.bin"))
    meta = dict(steps=args.steps, num_seq=args.num_seq, seqlen=args.seqlen,
                lr=args.lr, lam_kurt=args.lam_kurt, lam_draft=args.lam_draft,
                orth_err=orth_err, init="random_hadamard_seed0",
                first_loss=log[0], final_loss=log[-1], gpu=gpu,
                note="only R1 optimized; R2/R3 copied from random_hadamard R.bin")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"orth_err": orth_err, "final": log[-1],
                      "saved": os.path.join(out_dir, "R.bin")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
