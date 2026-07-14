#!/usr/bin/env python
"""Final experiment matrix (target 10 / task X-2,X-3).

Runs the full method x quantization comparison at batch=1 (EAGLE v1's supported
path), plus a batch-size probe. Methods:
  fp16_target_fp16_eagle              (stock EAGLE, unrotated FP16 target)
  spinquant_target_no_eagle           (vanilla decode on the rotated/quant target)
  spinquant_target_original_eagle     (NAIVE: rotated target, stock draft fed h_hat)
  spinquant_target_unrotate_interface (Variant A)
  spinquant_target_conjugated_draft   (Variant B)
  spinquant_target_retrained_rotated_draft (Variant C, if a draft ckpt exists)

Quantization settings: fp16 (rotate_only reference is separate), W4A4, W4A4KV4.
All quant timing is fake_quant (docs/00 S4): report relative speedup vs
no-EAGLE on the SAME target, not absolute deployment speed.

Writes results/final_sweep.jsonl + results/final_sweep_summary.csv.

Usage:
  python scripts/50_run_batch_sweep.py --smoke
  python scripts/50_run_batch_sweep.py --num-prompts 8 --settings fp16 w4a4 w4a4kv4
"""

import argparse
import gc
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             metrics, variant_eval, spinquant_bridge as sb)
from eagle_spinquant.rotation_interface import DraftInterfaceAdapter  # noqa: E402
from eagle_spinquant import draft_conjugation as dc  # noqa: E402

QUANT_SETTINGS = {
    "fp16": None,
    "w4a4": {"w_bits": 4, "a_bits": 4, "k_bits": 16, "v_bits": 16},
    "w4a4kv4": {"w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4},
}


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


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_fp16_stock(paths, prompts, max_new_tokens, temperature, dtype, logger):
    """Stock EAGLE on the unrotated FP16 target (the reference)."""
    rows = []
    model = eagle_bridge.load_eagle_model(paths["target_path"], paths["draft_path"],
                                          dtype=dtype, device_map="cuda")
    tok = model.get_tokenizer()
    wi = eagle_bridge.build_llama2_chat_prompt(tok, prompts[0]["text"])
    eagle_bridge.generate_and_measure(model, wi, mode="eagle", max_new_tokens=8, warmup=True)
    for p in prompts:
        ids = eagle_bridge.build_llama2_chat_prompt(tok, p["text"])
        for mode, method in [("eagle", "fp16_target_fp16_eagle"),
                             ("baseline", "fp16_target_no_eagle")]:
            r = eagle_bridge.generate_and_measure(model, ids, mode=mode,
                                                  max_new_tokens=max_new_tokens,
                                                  temperature=temperature)
            r.update({"method": method, "question_id": p["question_id"],
                      "quant_setting": "fp16", "target_quant": "none",
                      "runtime_mode": metrics.RUNTIME_FP16, "batch_size": 1,
                      "context_len": ids.shape[1]})
            logger.log(**r); rows.append(r)
    free(model)
    return rows


def run_quant_setting(paths, input_model_id, r_bin, setting, quant_config,
                      prompts, max_new_tokens, temperature, dtype, logger,
                      draft_ckpt=None):
    rows = []
    # Build once with naive adapter; evaluate naive -> unrotate (A) -> conjugate (B).
    model, adapter, stash = variant_eval.build_variant_model(
        paths, input_model_id, r_bin, variant="naive", stage="full",
        quant_config=quant_config, dtype=dtype, device="cuda")
    tok = model.get_tokenizer()
    R1, gamma_f, head_w = stash["R1"], stash["gamma_f"], stash["lm_head_weight"]

    def eval_method(method, include_baseline):
        rs = variant_eval.run_prompts(model, tok, prompts, max_new_tokens, temperature,
                                      method=method, stage="full",
                                      quant_config=quant_config,
                                      include_baseline=include_baseline)
        for r in rs:
            r["quant_setting"] = setting
        return rs

    # Save a clean copy of the ORIGINAL draft weights: needed to restore before the
    # destructive Variant-B fold, since Variant C overwrites the draft.
    import copy
    orig_draft_state = copy.deepcopy(model.ea_layer.state_dict())

    # naive (+ the no-eagle baseline on this quant target)
    rows += eval_method("spinquant_target_original_eagle", include_baseline=True)
    # Variant A
    adapter.uninstall()
    a = DraftInterfaceAdapter(model, R1=R1, gamma_f=gamma_f,
                              original_head_weight=head_w, variant="unrotate").install()
    rows += eval_method("spinquant_target_unrotate_interface", include_baseline=False)
    a.uninstall()

    # Variant C: load the retrained draft into the SAME target (NO rebuild -> no OOM).
    # It was trained in the Variant-A basis (unrotated features + original head), so
    # it also uses the 'unrotate' adapter.
    if draft_ckpt and os.path.isfile(draft_ckpt):
        sd = torch.load(draft_ckpt, map_location="cuda")
        model.ea_layer.load_state_dict(sd, strict=False)
        model.ea_layer.to(dtype)
        cadapt = DraftInterfaceAdapter(model, R1=R1, gamma_f=gamma_f,
                                       original_head_weight=head_w, variant="unrotate").install()
        rows += eval_method("spinquant_target_retrained_rotated_draft", include_baseline=False)
        cadapt.uninstall()
        model.ea_layer.load_state_dict(orig_draft_state, strict=True)  # restore for B
        model.ea_layer.to(dtype)

    # Variant B last (destructive fc fold on the restored original draft)
    dc.conjugate_draft_fc(model.ea_layer, R1.cuda(), gamma_f.cuda())
    DraftInterfaceAdapter(model, R1=R1, gamma_f=gamma_f,
                          original_head_weight=head_w, variant="conjugate").install()
    rows += eval_method("spinquant_target_conjugated_draft", include_baseline=False)
    for r in rows:
        logger.log(**r)
    free(model)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--settings", nargs="+", default=["fp16", "w4a4", "w4a4kv4"])
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--learned-r", action="store_true")
    ap.add_argument("--draft-ckpt", default=None,
                    help="Variant C draft checkpoint (defaults to variantC_draft_full)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # reduce fragmentation across the multiple full-model builds in one process
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.smoke:
        args.num_prompts = 2
        args.max_new_tokens = 48
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    input_model_id = cfg["model"]["target"]
    r_bin = ensure_r_bin(paths, learned=args.learned_r)
    temperature = cfg["evaluation"]["temperature"]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts) or [
        {"question_id": 0, "category": "smoke", "text": "Explain how a rainbow forms."}]
    draft_ckpt = args.draft_ckpt or os.path.join(
        PROJECT_ROOT, "outputs", "draft_ckpts", "variantC_draft_full", "draft_rotated.pt")

    run_dir = os.path.join(PROJECT_ROOT, "runs", "final_sweep")
    logger = logging_utils.RunLogger(run_dir, "final_sweep",
                                     config={"args": vars(args), "r_bin": r_bin})
    for setting in args.settings:
        if setting not in QUANT_SETTINGS:
            print(f"[skip] unknown setting {setting}"); continue
        print(f"\n########## setting: {setting} ##########")
        try:
            if setting == "fp16":
                run_fp16_stock(paths, prompts, args.max_new_tokens,
                               temperature, dtype, logger)
            else:
                run_quant_setting(
                    paths, input_model_id, r_bin, setting, QUANT_SETTINGS[setting],
                    prompts, args.max_new_tokens, temperature, dtype, logger,
                    draft_ckpt=draft_ckpt)
        except Exception as e:  # OOM etc. — record and continue (no silent loss)
            import traceback
            print(f"[FAIL] setting {setting}: {type(e).__name__}: {e}")
            logger.log(method=f"SETTING_FAILED:{setting}", error=f"{type(e).__name__}: {e}")
            gc.collect(); torch.cuda.empty_cache()
            traceback.print_exc()

    # Summarize from the ACCUMULATED incremental log (cumulative across reruns of
    # individual settings), so a per-setting rerun in a fresh process still yields a
    # complete final table.
    all_rows = logging_utils.read_jsonl(logger.jsonl_path)
    all_rows = [r for r in all_rows if "avg_accept_length" in r]
    summary = summarize(all_rows)
    logging_utils.write_jsonl(os.path.join(PROJECT_ROOT, "results", "final_sweep.jsonl"), all_rows)
    logging_utils.write_csv(os.path.join(PROJECT_ROOT, "results", "final_sweep_summary.csv"), summary)
    print("\n================ FINAL SWEEP SUMMARY ================")
    print(f"{'setting':10s} {'method':44s} {'accept':>7s} {'tok/s':>8s} {'rel_spd':>8s}")
    for s in summary:
        print(f"{s['quant_setting']:10s} {s['method']:44s} "
              f"{fmt(s['mean_accept_length']):>7s} {fmt(s['mean_tokens_per_s']):>8s} "
              f"{fmt(s.get('rel_speedup')):>8s}")
    return 0


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def summarize(rows):
    # group by (quant_setting, method)
    groups = {}
    for r in rows:
        if "avg_accept_length" not in r:
            continue
        key = (r.get("quant_setting"), r.get("method"))
        groups.setdefault(key, []).append(r)
    # baseline tok/s per quant setting (no-eagle) for relative speedup
    base_tps = {}
    for (setting, method), rs in groups.items():
        if method and "no_eagle" in method:
            vals = [r["tokens_per_s"] for r in rs if r.get("tokens_per_s")]
            if vals:
                base_tps[setting] = sum(vals) / len(vals)
    out = []
    for (setting, method), rs in groups.items():
        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        tps = mean("tokens_per_s")
        rel = (tps / base_tps[setting]) if (tps and base_tps.get(setting)) else None
        out.append({"quant_setting": setting, "method": method, "n": len(rs),
                    "mean_accept_length": mean("avg_accept_length"),
                    "mean_tokens_per_s": tps, "mean_ms_per_token": mean("ms_per_token"),
                    "mean_peak_mem_gib": mean("peak_mem_gib"),
                    "rel_speedup": rel, "runtime_mode": rs[0]["runtime_mode"],
                    "target_quant": rs[0].get("target_quant")})
    # stable sort: setting then descending accept
    order = {"fp16": 0, "w4a4": 1, "w4a4kv4": 2}
    out.sort(key=lambda s: (order.get(s["quant_setting"], 9),
                            -(s["mean_accept_length"] or 0)))
    return out


if __name__ == "__main__":
    sys.exit(main())
