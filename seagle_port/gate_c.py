"""Gate C: explicit inverse-rotation reference == folded-W_c interface.

Tensor level (uses gate_b captures, real rotated hiddens):
  fp64 / fp32 / bf16 parity of fc output and hidden_norm output between
    (a) explicit: per-branch H_rot @ R1^T -> stock fc
    (b) folded:   H_rot -> fc with W_i @ R1 blocks
Also verifies Gate D precursor: MP3 fold with random m_i is FP-invariant.

End-to-end level: rot_fp16 target + stock-vs-folded interface on 3 mtbench
prompts -> identical greedy blocks + acceptance sequences (bf16 tolerance:
report agreement, require accept-sequence equality).
"""
import argparse
import json
import os

import torch

from . import spinquant_target as sq
from . import interfaces

SRC = [1, 8, 15, 22, 29]


def tensor_level(rbin, run_dir, draft):
    rot = torch.load(os.path.join(run_dir, "tables", "gate_b_rot_fp16.pt"),
                     weights_only=False)
    R1 = sq.load_rbin(rbin)["R1"]
    fc_w = draft.fc.weight.data
    hn_w = draft.hidden_norm.weight.data.float()
    eps = 1e-6
    res = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32),
                        ("bf16", torch.bfloat16)):
        R = R1.to(dt)
        W = fc_w.to(dt)
        Wf = W.clone()
        for i in range(5):
            Wf[:, i * 4096:(i + 1) * 4096] = \
                W[:, i * 4096:(i + 1) * 4096].to(torch.float64) @ \
                R1.to(torch.float64)
        Wf = Wf.to(dt)
        errs, herrs = [], []
        for cap in rot["capture"][:4]:
            Hrot = torch.cat([cap["hidden"][l + 1] for l in SRC],
                             dim=-1).to(dt)
            x = Hrot.reshape(-1, 5, 4096) @ R.t()
            y_exp = x.reshape(-1, 5 * 4096) @ W.t()
            y_fold = Hrot @ Wf.t()
            errs.append(((y_exp - y_fold).norm() /
                         (y_exp.norm() + 1e-9)).item())
            for y in (y_exp, y_fold):
                pass
            he = (y_exp.float(), y_fold.float())
            n = lambda t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True)
                                          + eps) * hn_w
            herrs.append(((n(he[0]) - n(he[1])).norm() /
                          (n(he[0]).norm() + 1e-9)).item())
        res[f"fc_relerr_{dt_name}"] = max(errs)
        res[f"norm_relerr_{dt_name}"] = max(herrs)
    # MP3 invariance (fp64): random m_i, geomean 1
    torch.manual_seed(0)
    beta = torch.randn(5) * 0.3
    beta -= beta.mean()
    m = (4096.0 ** beta).tolist()
    W64 = fc_w.to(torch.float64)
    y_ref, y_mp3 = [], []
    for cap in rot["capture"][:2]:
        Hrot = torch.cat([cap["hidden"][l + 1] for l in SRC],
                         dim=-1).to(torch.float64)
        y_ref.append(Hrot @ W64.t())
        xs = Hrot.reshape(-1, 5, 4096).clone()
        Wm = W64.clone()
        for i in range(5):
            xs[:, i] *= m[i]
            Wm[:, i * 4096:(i + 1) * 4096] /= m[i]
        y_mp3.append(xs.reshape(-1, 20480) @ Wm.t())
    res["mp3_fp64_relerr"] = max(
        ((a - b).norm() / (a.norm() + 1e-9)).item()
        for a, b in zip(y_ref, y_mp3))
    res["mp3_scales"] = m
    return res


def e2e_level(rbin, run_dir, device):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from dflash.benchmark import _apply_chat_template
    from transformers import AutoTokenizer
    from .generate import dflash_generate_hooked

    tok = AutoTokenizer.from_pretrained(sq.__dict__.get(
        "MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    target = sq.build_target("meta-llama/Llama-3.1-8B-Instruct", "rot_fp16",
                             rbin_path=rbin, device=device)
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    R1 = sq.load_rbin(rbin)["R1"]
    ef, hf = interfaces.make_embed_head_restore(target, R1)
    ct_exp = interfaces.make_ctx_transform("explicit", R1=R1)
    d_fold = interfaces.fold_wc(draft, R1)

    rows = [json.loads(l) for l in open(os.path.join(
        os.path.dirname(__file__), "..", "cache", "mt-bench.jsonl"))][:3]
    same_accept, agree = [], []
    for inst in rows:
        msgs = [{"role": "user", "content": inst["turns"][0]}]
        ids = tok.encode(_apply_chat_template(tok, msgs, False),
                         return_tensors="pt").to(device)
        o1 = dflash_generate_hooked(draft, target, ids, 256,
                                    [tok.eos_token_id], 0.0,
                                    ctx_transform=ct_exp, embed_fn=ef,
                                    head_fn=hf, record_cycles=True)
        o2 = dflash_generate_hooked(d_fold, target, ids, 256,
                                    [tok.eos_token_id], 0.0,
                                    ctx_transform=None, embed_fn=ef,
                                    head_fn=hf, record_cycles=True)
        same_accept.append(o1.acceptance_lengths == o2.acceptance_lengths)
        n = min(o1.output_ids.shape[1], o2.output_ids.shape[1])
        agree.append((o1.output_ids[0, :n] == o2.output_ids[0, :n])
                     .float().mean().item())
    return {"e2e_accept_seq_equal": same_accept,
            "e2e_token_agree": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat", dtype=torch.bfloat16)
    res = tensor_level(args.rbin, args.run_dir, draft)
    del draft
    res.update(e2e_level(args.rbin, args.run_dir, args.device))
    json.dump(res, open(os.path.join(args.run_dir, "tables",
                                     "gate_c.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    ok = (res["fc_relerr_fp64"] < 1e-10 and res["fc_relerr_fp32"] < 1e-4
          and res["mp3_fp64_relerr"] < 1e-10
          and sum(res["e2e_token_agree"]) / len(res["e2e_token_agree"]) > 0.95)
    print("GATE C:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
