#!/usr/bin/env python
"""Bitwidth-AL study §11 verifier path consistency, for ALL THREE targets of the
3×3 matrix: fp16 stock, fused fake-W8A8, fused fake-W4A4.

Reuses validate_concat_selective_fp.verifier_consistency: full-sequence vs
chunked-KV vs incremental-KV target logits on identical token sequences
(top-1 agreement + logit deltas over the generated region). A drop in cross-
shape top-1 agreement under quantization marks AL results as partially
path-confounded (STOP GATE B input)."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from validate_concat_selective_fp import verifier_consistency, run_gen

ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality")

TARGETS = (("fp16", "none", "none"),
           ("fake_W8A8", "full", "w8a8"),
           ("fake_W4A4", "full", "w4a4"))

def main():
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        ("6,7", "0", "1", "2", "3", "4", "5", "6")
    # NOTE 2026-07-15: physical GPU 7 dropped off the bus mid-study; run on
    # the remaining visible device (cuda:0 = physical 6). GPUs 0-5 untouched.
    assert torch.cuda.device_count() in (1, 2), torch.cuda.device_count()
    if torch.cuda.device_count() == 1:
        print("[vc] WARNING: physical GPU 7 absent; using physical GPU 6", flush=True)
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    rotation_kind = sys.argv[2] if len(sys.argv) > 2 else "random_hadamard"
    out_suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    global ART
    ART = ART + out_suffix
    os.makedirs(ART, exist_ok=True)
    cfg = experiment.load_config(None); paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(6)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    vc_rows = []
    for tag, rot, q in TARGETS:
        print(f"[vc] building {tag} target ...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, rotation_kind, q, 0, device=dev, rotations_root=rr)
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        seqs = []
        for p in prompts:
            ids = build_prompt(tok, p["text"]).to(dev)
            t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                           max_steps=52, tree_choices=tree), ids.shape[1], 48)
            seqs.append(ids[0].tolist() + t)
        verifier_consistency(model, seqs, dev, tag, vc_rows)
        del model; torch.cuda.empty_cache()
        for r in vc_rows[-6:]:
            print(f"[vc] {r}", flush=True)
    logging_utils.write_csv(os.path.join(ART, "verifier_consistency",
                                         "target_logit_consistency.csv"), vc_rows)
    print("[vc] DONE", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
