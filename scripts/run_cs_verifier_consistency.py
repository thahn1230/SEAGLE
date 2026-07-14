#!/usr/bin/env python
"""Standalone §17 verifier-consistency: full vs chunked-KV vs incremental-KV
target logits on identical sequences, for fp16 AND fake-W4A4 targets."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from validate_concat_selective_fp import verifier_consistency, run_gen

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")

def main():
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    cfg = experiment.load_config(None); paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(6)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    vc_rows = []
    for tag, rot, q in (("fp16", "none", "none"), ("fake_W4A4", "full", "w4a4")):
        print(f"[vc] building {tag} target ...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, "random_hadamard", q, 0, device=dev, rotations_root=rr)
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
        for r in vc_rows[-12:]:
            print(f"[vc] {r}", flush=True)
    logging_utils.write_csv(os.path.join(ART, "verifier_consistency",
                                         "target_logit_consistency.csv"), vc_rows)
    print("[vc] DONE", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
