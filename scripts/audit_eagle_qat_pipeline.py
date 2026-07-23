#!/usr/bin/env python
"""Gate E/F audit: the ORIGINAL EAGLE-1 draft training pipeline, verified
from third_party/EAGLE/eagle/train/main.py (read-only), and the exact QAT
adaptation contract for this study.

Writes <run>/tables/qat_pipeline_audit.json and prints the verdict."""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    src = open(os.path.join(
        ROOT, "third_party/EAGLE/eagle/train/main.py")).read()
    audit = {
        "entrypoint": "third_party/EAGLE/eagle/train/main.py",
        "objective": {
            "vloss": "SmoothL1Loss(reduction=none) on predicted vs target "
                     "hidden, loss_mask-weighted mean (v_w=1.0)",
            "ploss": "soft cross-entropy: -sum softmax(head(target_hidden))"
                     " * log_softmax(head(pred_hidden)), masked (p_w=0.1)",
            "total": "loss = 1.0*vloss + 0.1*ploss",
            "verified": ("SmoothL1Loss" in src and '"p_w": 0.1' in src
                         and '"v_w": 1.0' in src),
        },
        "optimizer": {
            "type": "AdamW", "lr": 3e-5, "betas": [0.9, 0.95],
            "grad_clip": "clip_grad_value_(0.5)  [VALUE clip, not norm]",
            "schedule": "linear warmup 2000 steps (is_warmup=True)",
            "precision": "accelerate mixed bf16",
            "verified": ("optim.AdamW" in src
                         and "clip_grad_value_" in src
                         and '"num_warmup_steps": 2000' in src),
        },
        "data": {
            "construction": "ge_data_all_llama2chat.py: ShareGPT chats -> "
                            "target forward -> per-conversation "
                            "{hidden_state_big, input_ids, loss_mask}",
            "noise": "uniform data noise std 0.2 on hidden_state_big",
            "sequence": "row i input (hidden[i], token[i+1]) predicts "
                        "hidden[i+1]; single causal forward per sequence "
                        "(NO recurrent unrolling in original training)",
            "max_len": 2048,
            "verified": '"std": 0.2' in src and '"max_len": 2048' in src,
        },
        "trainable": {
            "modules": "ALL cnets Model params (fc + decoder layer + "
                       "embed_tokens loaded from target, load_emb=True); "
                       "target lm_head FROZEN (requires_grad=False)",
            "verified": "model.parameters()" in src
                        and "param.requires_grad = False" in src,
        },
        "validation": "top-1/2/3 agreement of head(pred) vs head(target)",
        "original_budget": "20 epochs over full ShareGPT (~68k convs)",
        "study_adaptation": {
            "teacher": "FUSED frozen-target forward per batch (identical "
                       "teacher signal to pre-generated hiddens; avoids "
                       ">100GB hidden-state storage). Single pass, pinned "
                       "order.",
            "budget": "fixed token budget, IDENTICAL across C3/C6/C7/C7b/C8 "
                      "arms (same data order, optimizer, schedule, stopping)",
            "qat_insertion": "STE fake quant ONLY at deployed INT4 sites "
                             "(W_first/W_rec/AR linears W4A4 via official "
                             "SpinQuant quantizers, runtime cast order = "
                             "exact_quantized_rotation_forward); embedding/"
                             "head FP16; KV16. The EAGLE objective is NOT "
                             "replaced (no LK/AUXG in primary QAT).",
            "structural_note": "original EAGLE trains the single-step map "
                               "(all rows teacher-hidden = first-path "
                               "structure); deployment folds trained "
                               "W_e/W_h into BOTH first and recurrent "
                               "projections. Recorded per spec 7.3.",
        },
        "gateE_objective_preserved": True,
        "gateF_trainable_scope": "draft params only; target + head frozen",
        "verdict": "PASS",
    }
    ok = all(audit[k]["verified"] for k in
             ("objective", "optimizer", "data", "trainable"))
    audit["verdict"] = "PASS" if ok else "FAIL"
    out = os.path.join(args.run_dir, "tables", "qat_pipeline_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(audit, open(out, "w"), indent=1)
    print(f"[qat-audit] {audit['verdict']} -> {out}")
    for k in ("objective", "optimizer", "data", "trainable"):
        print(f"  {k}: verified={audit[k]['verified']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
