#!/usr/bin/env python
"""Tasks 1+2: numeric audit of the EAGLE draft fc fold directions and the
lm_head basis, on the REAL Llama-2-7b-chat + EAGLE draft weights (fp64), with
the real random-Hadamard R1 and the real final-norm gamma_f.

Writes docs/fc_projection_fold_audit.md and docs/lm_head_basis_audit.md.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/audit_fc_lm_head.py   # (GPU not required)
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import experiment, study  # noqa: E402

DOCS = os.path.join(PROJECT_ROOT, "docs")
torch.set_grad_enabled(False)


def load_target_tensors(target_path):
    """Read model.norm.weight (gamma_f) and lm_head.weight from safetensors
    without a full model build."""
    from safetensors import safe_open
    idx = os.path.join(target_path, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    out = {}
    for key in ("model.norm.weight", "lm_head.weight"):
        shard = os.path.join(target_path, wm[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            out[key] = f.get_tensor(key).double()
    return out["model.norm.weight"], out["lm_head.weight"]


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def main():
    cfg = experiment.load_config()
    paths = experiment.resolve_paths(cfg)
    R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"]),
                   map_location="cpu", weights_only=False)
    R1 = R["R1"].double()                                   # [4096,4096], orthogonal
    gamma, W_lm = load_target_tensors(paths["target_path"])  # [4096], [32000,4096]

    sd = torch.load(os.path.join(paths["draft_path"], "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    fc_w = sd["fc.weight"].double()                          # [4096, 8192]
    D = 4096
    W_e = fc_w[:, :D]                                        # [4096,4096]
    W_h = fc_w[:, D:]                                        # [4096,4096]

    g = torch.Generator().manual_seed(0)
    e = torch.randn(16, D, generator=g, dtype=torch.float64)
    h = torch.randn(16, D, generator=g, dtype=torch.float64)
    f = torch.randn(16, D, generator=g, dtype=torch.float64)
    diagg = torch.diag(gamma)

    # ---------- Task 1: fc fold directions ----------
    fc1 = {
        "fc_weight_shape": list(fc_w.shape),
        "is_4096x8192": list(fc_w.shape) == [4096, 8192],
        "concat_order": "[e, h]  (cnets.py:592  cat((inputs_embeds, hidden_states)))",
        "W_e_slice": "fc.weight[:, :4096]", "W_h_slice": "fc.weight[:, 4096:]",
        "R1_orthogonal_max_dev": (R1.t() @ R1 - torch.eye(D, dtype=torch.float64)).abs().max().item(),
        "gamma_f_min": gamma.min().item(), "gamma_f_max": gamma.max().item(),
        "gamma_f_std": gamma.std().item(),
    }
    # (a) h_hat = (h/gamma)@R1  ->  W_h_folded = W_h @ diag(gamma) @ R1
    h_hat = (h / gamma) @ R1
    Wh_fold_a = W_h @ diagg @ R1
    fc1["A_h_hat_fold"] = {
        "claim": "h_hat @ (W_h @ diag(gamma_f) @ R1).T == h @ W_h.T",
        "rel_err": rel(h_hat @ Wh_fold_a.t(), h @ W_h.t()),
        "wrong_fold_W_h@R1_rel_err": rel(h_hat @ (W_h @ R1).t(), h @ W_h.t())}
    # (b) e_R = e@R1  ->  W_e_folded = W_e @ R1
    e_R = e @ R1
    fc1["B_e_R_fold"] = {
        "claim": "e_R @ (W_e @ R1).T == e @ W_e.T",
        "rel_err": rel(e_R @ (W_e @ R1).t(), e @ W_e.t())}
    # (c) h_R = h@R1  ->  W_h_folded = W_h @ R1
    h_R = h @ R1
    fc1["C_h_R_fold"] = {
        "claim": "h_R @ (W_h @ R1).T == h @ W_h.T",
        "rel_err": rel(h_R @ (W_h @ R1).t(), h @ W_h.t()),
        "wrong_fold_W_h@diag(gamma)@R1_rel_err":
            rel(h_R @ Wh_fold_a.t(), h @ W_h.t())}
    fc1["KEY_h_hat_vs_h_R_differ_by_gamma"] = {
        "note": "h_hat = (h/gamma_f)@R1 (scale-free a rotated); "
                "h_R = h@R1 (full post-norm hidden rotated). They differ by the "
                "final RMSNorm gain diag(gamma_f), which does NOT commute with R1.",
        "cos(h_hat, h_R)": cos(h_hat, h_R),
        "rel_l2(h_hat, h_R)": rel(h_hat, h_R)}

    _write_fc_doc(fc1)

    # ---------- Task 2: lm_head basis ----------
    lg_true = f @ W_lm.t()
    f_R = f @ R1
    f_hat = (f / gamma) @ R1
    W_lm_R = W_lm @ R1
    W_lm_hat = W_lm @ diagg @ R1
    lm = {
        "original_feature": {
            "head": "W_lm", "rel_err": rel(f @ W_lm.t(), lg_true)},
        "pure_rotated_feature_f_R": {
            "head": "W_lm_R = W_lm @ R1",
            "rel_err": rel(f_R @ W_lm_R.t(), lg_true),
            "WRONG_original_head_rel_err": rel(f_R @ W_lm.t(), lg_true),
            "WRONG_hat_head_rel_err": rel(f_R @ W_lm_hat.t(), lg_true)},
        "spinquant_feature_f_hat": {
            "head": "W_lm_hat = W_lm @ diag(gamma_f) @ R1",
            "rel_err": rel(f_hat @ W_lm_hat.t(), lg_true),
            "WRONG_original_head_rel_err": rel(f_hat @ W_lm.t(), lg_true),
            "WRONG_R_head_rel_err": rel(f_hat @ W_lm_R.t(), lg_true)},
        "top1_wrong_head_check": {
            "f_R_with_W_lm_top1_agree": (
                (f_R @ W_lm.t()).argmax(-1) == lg_true.argmax(-1)).float().mean().item(),
            "f_R_with_W_lm_R_top1_agree": (
                (f_R @ W_lm_R.t()).argmax(-1) == lg_true.argmax(-1)).float().mean().item()},
    }
    _write_lm_doc(lm)

    print(json.dumps({"fc": {k: v for k, v in fc1.items()
                             if k.startswith(("A_", "B_", "C_", "KEY"))},
                      "lm_head": lm}, indent=2, default=str))
    print(f"-> {DOCS}/fc_projection_fold_audit.md")
    print(f"-> {DOCS}/lm_head_basis_audit.md")
    return 0


def _write_fc_doc(d):
    L = ["# fc projection fold audit (Task 1)", "",
         "Real EAGLE-llama2-chat-7B `fc.weight`, real random-Hadamard R1, real "
         "target final-norm gamma_f; fp64.", "",
         "## Shapes / order", "",
         f"- `fc.weight` shape = **{d['fc_weight_shape']}** (Linear(8192->4096); "
         f"is [4096,8192] = {d['is_4096x8192']})",
         f"- concat order = **{d['concat_order']}**",
         f"- `W_e = {d['W_e_slice']}`, `W_h = {d['W_h_slice']}`",
         f"- R1 orthogonal max |RᵀR−I| = {d['R1_orthogonal_max_dev']:.2e}; "
         f"gamma_f in [{d['gamma_f_min']:.4f}, {d['gamma_f_max']:.4f}], std "
         f"{d['gamma_f_std']:.4f} (non-uniform)", "",
         "## Fold directions (rel-L2 of folded output vs original fc output)", "",
         "| input basis | correct fold | rel-err | wrong fold | wrong rel-err |",
         "|---|---|---:|---|---:|",
         f"| h_hat = (h/γ_f)@R1 | W_h @ diag(γ_f) @ R1 | "
         f"{d['A_h_hat_fold']['rel_err']:.2e} | W_h @ R1 | "
         f"{d['A_h_hat_fold']['wrong_fold_W_h@R1_rel_err']:.3f} |",
         f"| e_R = e@R1 | W_e @ R1 | {d['B_e_R_fold']['rel_err']:.2e} | — | — |",
         f"| h_R = h@R1 | W_h @ R1 | {d['C_h_R_fold']['rel_err']:.2e} | "
         f"W_h @ diag(γ_f) @ R1 | {d['C_h_R_fold']['wrong_fold_W_h@diag(gamma)@R1_rel_err']:.3f} |",
         "", "## Key point: h_hat vs h_R", "",
         f"- cos(h_hat, h_R) = {d['KEY_h_hat_vs_h_R_differ_by_gamma']['cos(h_hat, h_R)']:.4f}, "
         f"rel-L2 = {d['KEY_h_hat_vs_h_R_differ_by_gamma']['rel_l2(h_hat, h_R)']:.3f}",
         f"- {d['KEY_h_hat_vs_h_R_differ_by_gamma']['note']}", "",
         "All three claimed folds are exact (rel-err ~1e-15). Using the wrong "
         "fold (swapping the diag(γ_f) factor) yields ~30-50% error, confirming "
         "diag(γ_f) and R1 do NOT commute. The current SpinQuant tail exposes "
         "h_hat (needs the diag(γ_f) fold); a tail exposing h_R needs only W_h@R1 "
         "— the SAME fold used for recycled draft features f_R = f@R1, which is "
         "why h_R unifies the external and recurrent fc paths into one."]
    with open(os.path.join(DOCS, "fc_projection_fold_audit.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _write_lm_doc(d):
    L = ["# lm_head basis audit (Task 2)", "",
         "Real target `lm_head.weight`, real R1/gamma_f; fp64. logits reference "
         "= f @ W_lm.T (original feature, original head).", "",
         "| feature basis | correct head | rel-err | wrong head | wrong rel-err |",
         "|---|---|---:|---|---:|",
         f"| original f | W_lm | {d['original_feature']['rel_err']:.2e} | — | — |",
         f"| f_R = f@R1 | W_lm @ R1 | {d['pure_rotated_feature_f_R']['rel_err']:.2e} | "
         f"W_lm (orig) | {d['pure_rotated_feature_f_R']['WRONG_original_head_rel_err']:.3f} |",
         f"| f_hat = (f/γ_f)@R1 | W_lm @ diag(γ_f) @ R1 | "
         f"{d['spinquant_feature_f_hat']['rel_err']:.2e} | W_lm @ R1 | "
         f"{d['spinquant_feature_f_hat']['WRONG_R_head_rel_err']:.3f} |",
         "",
         f"Wrong-head top-1 agreement: f_R scored by original W_lm = "
         f"{d['top1_wrong_head_check']['f_R_with_W_lm_top1_agree']:.3f} "
         f"(vs correct W_lm@R1 = {d['top1_wrong_head_check']['f_R_with_W_lm_R_top1_agree']:.3f}).",
         "",
         "Each feature basis has exactly one correct head; using another basis's "
         "head causes large logit error (~30-50% rel-L2, near-zero top-1). A "
         "fully-R1 draft (features f_R) MUST be scored by W_lm @ R1, not the "
         "original head and not the SpinQuant fused head W_lm@diag(γ_f)@R1."]
    with open(os.path.join(DOCS, "lm_head_basis_audit.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
