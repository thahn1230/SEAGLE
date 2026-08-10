#!/usr/bin/env python
"""PPL sanity anchor for the rebuilt R1_T (GS R1/R2 study).

Official-contract wikitext-2 PPL of the in-process chat builds:
  fp16 target            (anchor ~6.9452)
  W4A4 learned R1_T      (anchor ~6.9629, prior chat learned-RTN KV16)
Exit 1 if the W4A4 PPL leaves the sanity band [6.0, 8.5] (CLAUDE.md
ballpark 6-8) — a far-off value means a broken rotation, not a result.
Writes <run>/audit/ppl_sanity.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.official_ppl import (get_official_test_tokens,
                                          official_ppl,
                                          eagle_forward_logits)

KIND = "learned_chat_w4a4kv16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    toks = get_official_test_tokens(paths["target_path"])
    out = {}
    for name, (rot, quant) in (("fp16", ("none", "none")),
                               ("w4a4_learned_r1t", ("full", "w4a4"))):
        model, _stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], rot, KIND, quant, 0,
            device=args.device, rotations_root=rr)
        ppl, ce, rows = official_ppl(eagle_forward_logits(model), toks,
                                     args.device)
        out[name] = dict(ppl=round(ppl, 4), ce=round(ce, 5),
                         n_windows=len(rows))
        print(f"[ppl] {name}: {ppl:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    out["anchors"] = dict(fp16=6.9452, w4a4_learned=6.9629,
                          band=[6.0, 8.5])
    ok = 6.0 <= out["w4a4_learned_r1t"]["ppl"] <= 8.5
    out["verdict"] = "PASS" if ok else "FAIL"
    p = os.path.join(args.run_dir, "audit", "ppl_sanity.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(f"[ppl] {out['verdict']} -> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
