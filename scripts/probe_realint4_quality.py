#!/usr/bin/env python
"""Quality probe for the REAL-INT4 integrated targets: wikitext-2 PPL of the
exact integrated model (rotation + swapped kernels) + sample completions.

Purpose: decide whether the QuaRot-arm acceptance (4.18 > fp16 stock 3.58)
reflects degraded/more-predictable output rather than a genuine gain.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/probe_realint4_quality.py \
      --run-id rotstudy_realint4_<ts> --backend quarot_w4a4
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import experiment, realint4, study  # noqa: E402

DEV = "cuda:0"


@torch.no_grad()
def wikitext2_ppl(model, tok, seqlen=2048, max_chunks=40):
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    n = min(ids.shape[1] // seqlen, max_chunks)
    nll, cnt = 0.0, 0
    for i in range(n):
        c = ids[:, i * seqlen:(i + 1) * seqlen].to(DEV)
        logits = model(input_ids=c, use_cache=False).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            c[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item(); cnt += c.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / cnt))), n


@torch.no_grad()
def greedy_completion(model, tok, prompt_ids, n_new=120):
    # no KV cache: the vendored KVLlama uses EAGLE's own cache classes, and
    # this probe only needs 3 short samples — full-prefix recompute is fine.
    ids = prompt_ids.to(DEV)
    for _ in range(n_new):
        logits = model(input_ids=ids, use_cache=False).logits
        nxt = logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], 1)
    return tok.decode(ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", required=True,
                    choices=["fp16_ref", "tinygemm_w4a16", "quarot_w4a4"])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")

    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"], use_fast=False)
    m = KVLlama.from_pretrained(paths["target_path"],
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True).eval()
    if args.backend != "fp16_ref":
        r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                                 rotations_root)
        study.apply_rotation_quant(m, "r1r2", r_bin, "none",
                                   cfg["model"]["target"], DEV)
        m.to(DEV)
        realint4.swap_target_linears(m, args.backend, DEV)
    else:
        m.to(DEV)

    ppl, n_chunks = wikitext2_ppl(m, tok)
    prompts = experiment.load_mt_bench_prompts(3)
    from eagle_spinquant import eagle_bridge
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build = eagle_bridge.PROMPT_BUILDERS[tmpl]
    samples = []
    for p in prompts:
        text = greedy_completion(m, tok, build(tok, p["text"]))
        # crude degeneration metric: distinct-2 over the completion tokens
        toks = text.split()
        bigrams = list(zip(toks, toks[1:]))
        d2 = len(set(bigrams)) / max(len(bigrams), 1)
        samples.append({"question_id": p["question_id"],
                        "distinct2": round(d2, 3),
                        "completion_first_400_chars": text[:400]})
    out = {"backend": args.backend, "model": cfg["model"]["target"],
           "wikitext2_ppl": ppl, "ppl_chunks": n_chunks,
           "note": "PPL computed on the EXACT integrated target used by the "
                   "acceptance runs (rotation r1r2 + real kernels)",
           "samples": samples}
    path = os.path.join(run_dir, f"quality_probe_{args.backend}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"},
                     indent=2))
    for s in samples:
        print(f"--- q{s['question_id']} distinct2={s['distinct2']}\n"
              f"{s['completion_first_400_chars']}\n")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
