#!/usr/bin/env python
"""GATE D closer: official-contract PPL of the IN-PROCESS chat rh0 W4A4 build."""
import os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from eagle_spinquant import experiment, study
from eagle_spinquant.official_ppl import (get_official_test_tokens,
                                          official_ppl, eagle_forward_logits)
torch.set_grad_enabled(False)
dev = "cuda:0"
cfg = experiment.load_config(None); paths = experiment.resolve_paths(cfg)
rr = cfg.get("paths", {}).get("rotations_root")
model, stash, _ = study.build_study_target(
    paths["target_path"], paths["draft_path"], cfg["model"]["target"],
    "full", "random_hadamard", "w4a4", 0, device=dev, rotations_root=rr)
toks = get_official_test_tokens(paths["target_path"])
ppl, ce, rows = official_ppl(eagle_forward_logits(model), toks, dev)
out = dict(config="chat_rh0_w4a4_inprocess", official_contract=True,
           ppl=round(ppl, 4), ce=round(ce, 5), n_windows=len(rows))
p = os.path.join(PROJECT_ROOT, "artifacts", "spinquant_ppl_reproduction_fix",
                 "pipeline_parity", "inprocess_official_ppl.json")
json.dump(out, open(p, "w"), indent=2)
print("[gateD]", json.dumps(out), flush=True)
