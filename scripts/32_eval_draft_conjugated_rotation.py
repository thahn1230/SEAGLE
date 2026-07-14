#!/usr/bin/env python
"""Variant B: draft-conjugated rotation (target 8 / task V-1).

Two parts:
  1. Full-precision conjugation identity check on the REAL EAGLE draft:
     stock-draft(h) vs conjugated-draft(h_hat), single forward. Writes
     results/conjugation_identity_check.json.
  2. End-to-end EAGLE generation with the fc-folded draft consuming h_hat, vs
     Variant A on the same rotated target. Writes conjugated_rotation_eval.jsonl.

Documents the recycling caveat (docs/04): the fc fold is exact for the first
draft token but the EAGLE tree recycles the draft's original-basis predictions
through the same fc, so faithful full-generation Variant B equals Variant A only
with recycled-feature compensation. We report both so the divergence is visible.

Usage:
  python scripts/32_eval_draft_conjugated_rotation.py --smoke
  python scripts/32_eval_draft_conjugated_rotation.py --stage full --num-prompts 10
"""

import argparse
import copy
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             variant_eval, spinquant_bridge as sb)
from eagle_spinquant import draft_conjugation as dc  # noqa: E402
from eagle_spinquant.rotation_interface import DraftInterfaceAdapter  # noqa: E402


def ensure_r_bin(paths, learned=False):
    if learned:
        p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "learned_w16a4kv4", "R.bin")
        if os.path.isfile(p):
            return p
    p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "random_hadamard", "R.bin")
    if not os.path.isfile(p):
        from transformers import AutoConfig
        conf = AutoConfig.from_pretrained(paths["target_path"])
        sb.make_random_rotation_bin(conf, p, mode="hadamard", seed=0)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--stage", default="rotate_only", choices=["rotate_only", "full"])
    ap.add_argument("--w-method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--kv4", action="store_true")
    ap.add_argument("--learned-r", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.smoke:
        args.num_prompts = 2
        args.max_new_tokens = 48
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    input_model_id = cfg["model"]["target"]
    r_bin = ensure_r_bin(paths, learned=args.learned_r)

    quant_config = None
    if args.stage == "full":
        quant_config = dict(cfg["quantization"])
        quant_config["w_method"] = args.w_method
        if not args.kv4:
            quant_config["k_bits"] = 16
            quant_config["v_bits"] = 16

    run_name = f"variantB_{args.stage}" + ("_kv4" if args.kv4 else "")
    run_dir = os.path.join(PROJECT_ROOT, "runs", run_name)
    logger = logging_utils.RunLogger(run_dir, run_name,
                                     config={"args": vars(args), "quant": quant_config})

    # Build rotated model with naive adapter (draft UNFOLDED for the identity check).
    print(f"building rotated target (stage={args.stage})...")
    model, adapter, stash = variant_eval.build_variant_model(
        paths, input_model_id, r_bin, variant="naive", stage=args.stage,
        quant_config=quant_config, dtype=dtype, device="cuda")
    tok = eagle_bridge.get_tokenizer(model)
    R1, gamma_f = stash["R1"], stash["gamma_f"]

    # (1) conjugation identity check on the REAL draft (single forward)
    print("running full-precision conjugation identity check on the real draft...")
    ident = dc.conjugation_identity_check(model.ea_layer, R1, gamma_f,
                                          seq=8, device="cuda", dtype=torch.float32)
    ident_path = os.path.join(PROJECT_ROOT, "results", "conjugation_identity_check.json")
    os.makedirs(os.path.dirname(ident_path), exist_ok=True)
    with open(ident_path, "w") as f:
        json.dump(ident, f, indent=2)
    print("  identity check:", json.dumps(ident, indent=2))

    prompts = experiment.load_mt_bench_prompts(args.num_prompts) or [
        {"question_id": 0, "category": "smoke", "text": "Explain how a rainbow forms."}]
    temperature = cfg["evaluation"]["temperature"]
    all_rows = []

    # (2a) Variant A reference on this rotated model (adapter swap, unfolded draft).
    # Always uninstall the current adapter before installing a new one so each
    # adapter wraps the TRUE stock topK_genrate (never a previous wrapper).
    adapter.uninstall()
    a_ref = DraftInterfaceAdapter(model, R1=R1, gamma_f=gamma_f,
                                  original_head_weight=stash["lm_head_weight"],
                                  variant="unrotate").install()
    print("evaluating Variant A reference...")
    all_rows += variant_eval.run_prompts(
        model, tok, prompts, args.max_new_tokens, temperature,
        method="spinquant_target_unrotate_interface", stage=args.stage,
        quant_config=quant_config, include_baseline=False)

    # (2b) Variant B: restore stock, fold fc, install 'conjugate' adapter
    a_ref.uninstall()
    print("folding draft fc (Variant B) and evaluating...")
    dc.conjugate_draft_fc(model.ea_layer, R1.cuda(), gamma_f.cuda())
    DraftInterfaceAdapter(
        model, R1=R1, gamma_f=gamma_f, original_head_weight=stash["lm_head_weight"],
        variant="conjugate").install()
    all_rows += variant_eval.run_prompts(
        model, tok, prompts, args.max_new_tokens, temperature,
        method="spinquant_target_conjugated_draft", stage=args.stage,
        quant_config=quant_config, include_baseline=False)

    for r in all_rows:
        logger.log(**r)
    summary = summarize(all_rows)
    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "conjugated_rotation_eval.jsonl"), all_rows)
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "conjugated_rotation_summary.csv"), summary)

    write_docs04(ident)
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['method']:42s} accept={fmt(s['mean_accept_length'])} tok/s={fmt(s['mean_tokens_per_s'])}")
    print(f"\nidentity: feat_abs_err={ident['feature_max_abs_err']:.2e} "
          f"cos={ident['feature_cosine_sim']:.6f} logit_kl={ident['logit_kl']:.2e}")
    return 0


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def summarize(rows):
    methods = []
    for r in rows:
        m = r.get("method")
        if m and m not in methods:
            methods.append(m)
    out = []
    for m in methods:
        mr = [r for r in rows if r.get("method") == m and "avg_accept_length" in r]
        if not mr:
            continue
        def mean(k):
            v = [r[k] for r in mr if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        out.append({"method": m, "n": len(mr),
                    "mean_accept_length": mean("avg_accept_length"),
                    "mean_tokens_per_s": mean("tokens_per_s"),
                    "runtime_mode": mr[0]["runtime_mode"],
                    "target_quant": mr[0].get("target_quant")})
    return out


def write_docs04(ident: dict) -> None:
    doc = f"""# 04 — Draft Conjugation Notes (Variant B)

## The transform

EAGLE draft first projection (cnets.py:592-593): `fc(cat([e, h]))`, an
`nn.Linear(2D -> D)`. In nn.Linear terms `y = x @ W^T`; with `x = [e | h]` the
embedding occupies input columns `[0:D]` and the hidden occupies `[D:2D]`.

To consume the rotated hidden `h_hat` directly (h = (h_hat @ R1^T) * gamma_f):

    W_h_new = W_h_old @ (diag(gamma_f) @ R1)        # fc.weight[:, D:2D]

The embedding block is left untouched (the draft's embedding is the original,
unrotated Llama embedding, so `e` is already in the right basis).

## Full-precision identity check (real draft, single forward)

    feature_max_abs_err = {ident['feature_max_abs_err']:.3e}
    feature_rel_err     = {ident['feature_rel_err']:.3e}
    feature_cosine_sim  = {ident['feature_cosine_sim']:.6f}
    logit_kl            = {ident['logit_kl']:.3e}

The conjugated draft consuming `h_hat` reproduces the stock draft consuming `h`
to floating-point roundoff on the first forward. Variant B == Variant A there.

## Recycling caveat (discovered from the code)

EAGLE tree drafting (cnets.py:806-817) recycles the draft's OWN predicted
features `out_hidden` back into the same `fc` for subsequent tree levels. The
draft is trained to predict ORIGINAL-basis features, so `out_hidden` is in the
original basis — but after folding, `fc`'s hidden block now expects the ROTATED
basis. Hence the fc fold is exact only for the FIRST (external) hidden; across the
recycled tree-expansion steps Variant B diverges from Variant A unless the
recycled features are also rotated to the h_hat basis (an online op of the same
cost as Variant A's unrotation).

Consequence: Variant A (explicit unrotation at the draft entry, applied once per
draft call, with the internal recycling naturally consistent in the original
basis) is the correctness reference for end-to-end generation. Variant B is the
zero-overhead form ONLY for single-step feature prediction; for full generation
it is either (a) an approximation, or (b) equivalent to A once recycled features
are compensated. The end-to-end numbers in `results/conjugated_rotation_*` show
this divergence directly.
"""
    with open(os.path.join(PROJECT_ROOT, "docs", "04_DRAFT_CONJUGATION_NOTES.md"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    sys.exit(main())
