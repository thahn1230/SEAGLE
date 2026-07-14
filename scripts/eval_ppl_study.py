#!/usr/bin/env python
"""Wikitext-2 perplexity for a study target under (rotation, quant) settings,
using the SAME in-process build path as the acceptance study
(study.apply_rotation_quant on the EAGLE-vendored KV Llama), so ppl.csv rows
describe exactly the targets the acceptance rows ran on.

All quantized settings are FAKE quantization (QDQ + FP16 matmuls). This script
never claims real INT4 anything.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/eval_ppl_study.py \
    --run-id rotstudy_vicuna_x --config configs/vicuna_experiment.yaml \
    --settings fp16,rotonly,w4a4,w4a4kv4
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import experiment, logging_utils, study  # noqa: E402

DEV = "cuda:0"
SETTINGS = {
    #            rotation  quant
    "fp16":     ("none",   "none"),
    "rotonly":  ("full",   "none"),
    "w4a16":    ("full",   "w4a16"),
    "w4a4":     ("full",   "w4a4"),
    "w4a4kv4":  ("full",   "w4a4kv4"),
}


@torch.no_grad()
def wikitext2_ppl(model, tokenizer, seqlen=2048, device=DEV):
    """Standard GPTQ-style protocol: concat raw test set, split into seqlen
    windows, mean NLL over all predicted tokens."""
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    n_chunks = ids.shape[1] // seqlen
    nll_sum, n_tok = 0.0, 0
    for i in range(n_chunks):
        chunk = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(input_ids=chunk, use_cache=False).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = chunk[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        nll_sum += loss.item()
        n_tok += shift_labels.numel()
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--settings", default="fp16,rotonly,w4a4,w4a4kv4")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, \
        "PPL eval is a single-GPU job; launch with exactly one visible GPU"
    print("[ppl] GPU policy OK:", gpu, flush=True)

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"], use_fast=False)

    rows = []
    for name in [s.strip() for s in args.settings.split(",") if s.strip()]:
        rotation, quant = SETTINGS[name]
        print(f"[ppl] building target: {name} (rotation={rotation}, "
              f"quant={quant})", flush=True)
        from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
        m = KVLlama.from_pretrained(paths["target_path"],
                                    torch_dtype=torch.float16,
                                    low_cpu_mem_usage=True).eval()
        r_bin = None
        if rotation != "none":
            r_bin = study.r_bin_path("random_hadamard", args.seed,
                                     paths["target_path"], rotations_root)
        study.apply_rotation_quant(m, rotation, r_bin, quant,
                                   cfg["model"]["target"], DEV)
        m.to(DEV)
        t0 = time.time()
        ppl = wikitext2_ppl(m, tok, seqlen=args.seqlen)
        print(f"[ppl] {name}: {ppl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        rows.append({
            "quant": name, "model": cfg["model"]["target"],
            "w_bits": 16 if quant == "none" else 4,
            "a_bits": {"none": 16, "w4a16": 16, "w4a4": 4, "w4a4kv4": 4}[quant],
            "kv_bits": 4 if quant == "w4a4kv4" else 16,
            "rotate": rotation != "none", "w_method": "rtn",
            "learned_R": False, "ppl": ppl,
            "runtime_mode": ("fp16_bf16_baseline" if quant == "none"
                             else "fake_quant_pytorch"),
            "seqlen": args.seqlen,
            "notes": "in-process study build (study.apply_rotation_quant, "
                     "RTN fake quant); protocol: wikitext-2-raw test, "
                     f"{args.seqlen}-token windows, mean NLL",
        })
        del m
        torch.cuda.empty_cache()

    out = os.path.join(run_dir, "ppl.csv")
    logging_utils.write_csv(out, rows)
    print(f"[ppl] wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
