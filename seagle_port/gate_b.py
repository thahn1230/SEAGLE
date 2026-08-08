"""Gate B: SpinQuant target standalone correctness on Llama-3.1-8B-Instruct.

Memory-safe: one process per arm (fp16 / rot_fp16 / w8a8 / w4a4) saves
captures + PPL to run_dir/tables/gate_b_<arm>.pt; `--arm compare` merges.

Checks:
  B1 rotation invariance : stock vs rot_fp16 logits (argmax agree, |dL|max)
  B2 residual basis      : H_rot[l] == H_stock[l] @ R1 for every layer 0..32
  B3 quantizer effect    : w8a8/w4a4 weights differ from rot_fp16 + act bits
  B4 wikitext2 PPL       : all four arms
"""
import argparse
import json
import os

import torch

from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SRC_LAYERS = [1, 8, 15, 22, 29]


def _prompts(n=6):
    import json as _j
    path = os.path.join(os.path.dirname(__file__), "..", "cache", "gsm8k.jsonl")
    rows = [_j.loads(l) for l in open(path)][:n]
    return [r["turns"][0][:2000] for r in rows]


@torch.inference_mode()
def capture(model, tok, prompts, device):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=512
                  ).input_ids.to(device)
        o = model(ids, output_hidden_states=True, use_cache=False)
        outs.append({"logits": o.logits[0, -32:].float().cpu(),
                     "hidden": [h[0].float().cpu() for h in o.hidden_states]})
    return outs


@torch.inference_mode()
def ppl_wikitext2(model, tok, device, n_windows=8, seqlen=2048):
    from datasets import load_dataset
    txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1",
                                   split="test")["text"])
    enc = tok(txt, return_tensors="pt").input_ids[0]
    nll, cnt = 0.0, 0
    for i in range(n_windows):
        ids = enc[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(device)
        out = model(ids, labels=ids, use_cache=False)
        nll += out.loss.item() * (ids.numel() - 1)
        cnt += ids.numel() - 1
    return float(torch.exp(torch.tensor(nll / cnt)))


def run_arm(arm, rbin, run_dir, device):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = sq.build_target(MODEL, arm, rbin_path=rbin, device=device)
    rec = {"ppl": ppl_wikitext2(m, tok, device),
           "capture": capture(m, tok, _prompts(), device)}
    if arm != "fp16":
        wq = m.model.layers[0].self_attn.q_proj
        w = (wq.module.weight if hasattr(wq, "module") else wq.weight)
        rec["q_proj_w"] = w.detach().float().cpu()
        rec["act_bits"] = int(wq.quantizer.bits) if hasattr(wq, "quantizer") \
            else 16
    torch.save(rec, os.path.join(run_dir, "tables", f"gate_b_{arm}.pt"))
    print(f"[gate_b] {arm} done ppl={rec['ppl']:.3f}")


def compare(rbin, run_dir):
    T = lambda a: torch.load(os.path.join(run_dir, "tables",
                                          f"gate_b_{a}.pt"),
                             weights_only=False)
    stock, rot = T("fp16"), T("rot_fp16")
    w8, w4 = T("w8a8"), T("w4a4")
    res = {"rbin": rbin, "model": MODEL}
    for a in ("fp16", "rot_fp16"):
        res[f"ppl_{a}"] = T(a)["ppl"]
    res["ppl_w8a8"], res["ppl_w4a4"] = w8["ppl"], w4["ppl"]

    agree, dmax = [], []
    for cs, cr in zip(stock["capture"], rot["capture"]):
        agree.append((cs["logits"].argmax(-1) == cr["logits"].argmax(-1))
                     .float().mean().item())
        dmax.append((cs["logits"] - cr["logits"]).abs().max().item())
    res["B1_argmax_agree"] = sum(agree) / len(agree)
    res["B1_logit_maxabs"] = max(dmax)

    R1 = sq.load_rbin(rbin)["R1"].to(torch.float32)
    # index 32 is the POST-final-norm entry; the rotated model fused gamma_f
    # into lm_head, so index 32 differs by construction and is never consumed
    # by the DFlash interface (sources are residual indices 2,9,16,23,30).
    layer_err = {}
    for li in range(33):
        errs = []
        for cs, cr in zip(stock["capture"], rot["capture"]):
            ref = cs["hidden"][li] @ R1
            errs.append(((cr["hidden"][li] - ref).norm() /
                         (ref.norm() + 1e-9)).item())
        layer_err[li] = sum(errs) / len(errs)
    res["B2_relerr_residual_layers_max"] = max(
        v for k, v in layer_err.items() if k < 32)
    res["B2_relerr_postnorm_idx32"] = layer_err[32]
    res["B2_relerr_per_layer"] = layer_err
    res["B2_relerr_src_layers"] = {li: layer_err[li + 1] for li in SRC_LAYERS}

    for tag, rec in (("w8a8", w8), ("w4a4", w4)):
        res[f"B3_{tag}_weight_changed"] = bool(
            not torch.equal(rec["q_proj_w"], rot["q_proj_w"]))
        res[f"B3_{tag}_act_bits"] = rec["act_bits"]

    out = os.path.join(run_dir, "tables", "gate_b.json")
    json.dump(res, open(out, "w"), indent=1)
    slim = {k: v for k, v in res.items() if k != "B2_relerr_per_layer"}
    print(json.dumps(slim, indent=1))
    ok = (res["B1_argmax_agree"] > 0.95
          and res["B2_relerr_residual_layers_max"] < 0.05
          and res["B3_w4a4_weight_changed"])
    print("GATE B:", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["fp16", "rot_fp16", "w8a8", "w4a4", "compare"])
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    if args.arm == "compare":
        compare(args.rbin, args.run_dir)
    else:
        run_arm(args.arm, args.rbin, args.run_dir, args.device)


if __name__ == "__main__":
    main()
