#!/usr/bin/env python
"""FP equivalence gate (task I-2). All with quantization OFF (rotate_only).

Validates the rotation PORT onto EAGLE's vendored LLaMA + the unrotation map on
the REAL 7B target, using a random-Hadamard R (any orthogonal R preserves logits
in FP, so this needs no learned rotation):

  (a) rotated-target logits  == original-target logits          (SpinQuant invariant)
  (b) unrotate(h_hat)        == original post-final-norm hidden  (draft interface)

Memory-safe: computes the original outputs, frees the model, then the rotated one.
Writes results/fp_equivalence.json. Exit 0 iff both pass.

Usage: python tests/test_fp_equivalence.py --gpu 0
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import experiment, spinquant_bridge as sb  # noqa: E402
from eagle_spinquant import rotation_interface as ri  # noqa: E402


@torch.no_grad()
def original_outputs(target_path, input_ids, dtype, device):
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    model = KVLlama.from_pretrained(target_path, torch_dtype=dtype,
                                    low_cpu_mem_usage=True).to(device).eval()
    out = model.model(input_ids=input_ids.to(device))
    h = out[0]                          # post-final-norm hidden (EAGLE's feature)
    logits = model.lm_head(h)
    h = h.float().cpu()
    logits = logits.float().cpu()
    cfg = model.config
    del model
    torch.cuda.empty_cache()
    return h, logits, cfg


@torch.no_grad()
def rotated_outputs(target_path, input_model_id, r_bin, input_ids, dtype, device):
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    model = KVLlama.from_pretrained(target_path, torch_dtype=dtype,
                                    low_cpu_mem_usage=True).to(device).eval()
    spec = sb.default_ptq_args(rotate=True, optimized_rotation_path=r_bin)
    stash = sb.apply_spinquant_pipeline(model, spec, input_model_id=input_model_id,
                                        stage="rotate_only")
    model.to(device)
    out = model.model(input_ids=input_ids.to(device))
    h_hat = out[0]
    logits = model.lm_head(h_hat)      # rotated lm_head consumes h_hat directly
    h_hat = h_hat.float().cpu()
    logits = logits.float().cpu()
    del model
    torch.cuda.empty_cache()
    return h_hat, logits, stash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seqlen", type=int, default=64)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    target_path = paths["target_path"]
    input_model_id = cfg["model"]["target"]

    # fixed prompt token ids
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target_path)
    text = ("The theory of relativity fundamentally changed our understanding of "
            "space and time, and its consequences remain central to modern physics.")
    input_ids = tok(text, return_tensors="pt").input_ids[:, :args.seqlen]

    # mint a random-Hadamard R.bin (any orthogonal R preserves logits in FP)
    from transformers import AutoConfig
    conf = AutoConfig.from_pretrained(target_path)
    r_bin = os.path.join(PROJECT_ROOT, "outputs", "rotations", "random_hadamard", "R.bin")
    sb.make_random_rotation_bin(conf, r_bin, mode="hadamard", seed=0)

    print("computing original outputs...")
    h, logits_orig, _ = original_outputs(target_path, input_ids, dtype, device)
    print("computing rotated outputs...")
    h_hat, logits_rot, stash = rotated_outputs(
        target_path, input_model_id, r_bin, input_ids, dtype, device)

    # (a) logits equivalence
    logit_abs = (logits_orig - logits_rot).abs()
    logit_max = logit_abs.max().item()
    logit_mean = logit_abs.mean().item()
    # argmax (greedy token) agreement — the property EAGLE verification depends on
    argmax_agree = (logits_orig.argmax(-1) == logits_rot.argmax(-1)).float().mean().item()

    # (b) unrotation map: unrotate(h_hat) == h
    R1 = stash["R1"].float()
    gamma_f = stash["gamma_f"].float()
    h_rec = ri.unrotate_hidden(h_hat.float(), R1, gamma_f)
    hid_abs = (h - h_rec).abs()
    hid_max = hid_abs.max().item()
    hid_scale = h.abs().max().item()
    hid_rel = hid_max / (hid_scale + 1e-8)

    results = {
        "seqlen": input_ids.shape[1], "dtype": args.dtype,
        "logit_max_abs_err": logit_max,
        "logit_mean_abs_err": logit_mean,
        "greedy_argmax_agreement": argmax_agree,
        "hidden_unrotate_max_abs_err": hid_max,
        "hidden_scale": hid_scale,
        "hidden_unrotate_rel_err": hid_rel,
        # fp16 through a 7B model accumulates error; tolerances are generous but
        # meaningful (random logits would disagree ~100% and err ~ O(10)).
        "logits_equivalent": (argmax_agree > 0.99 and logit_max < 0.5),
        "unrotation_correct": hid_rel < 0.05,
    }
    results["passed"] = results["logits_equivalent"] and results["unrotation_correct"]

    out = os.path.join(PROJECT_ROOT, "results", "fp_equivalence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"-> {out}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
