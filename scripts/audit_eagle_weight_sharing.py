#!/usr/bin/env python
"""Weight-sharing audit: measure (never assume) whether the EAGLE draft shares
tensors with the target, and what happens to each after target rotation.

Answers with evidence:
  1. draft embed_tokens vs target embed_tokens: same object / same storage /
     equal values? (before AND after rotation)
  2. lm_head: physically shared into the draft, or passed at call time?
     (module inspection + the exact code lines)
  3. after target rotation, did draft embedding change? (max abs diff vs
     pre-rotation snapshot)
  4. after target rotation, is there any draft-owned lm_head to rotate?
  5. gamma_f = target.model.norm.weight: shape/min/max/mean/std/first 8.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/audit_eagle_weight_sharing.py \
      --out-dir runs/rotation_aware_audit_<ts>
"""

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import eagle_bridge, experiment, study  # noqa: E402

DEV = "cuda:0"


def tstats(t):
    t = t.detach().float()
    return {"shape": list(t.shape), "min": t.min().item(), "max": t.max().item(),
            "mean": t.mean().item(), "std": t.std().item(),
            "first8": [round(v, 6) for v in t.flatten()[:8].tolist()]}


def share_report(a, b):
    return {
        "same_python_object": a is b,
        "same_storage_ptr": a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr(),
        "data_ptr_a": hex(a.data_ptr()), "data_ptr_b": hex(b.data_ptr()),
        "same_shape": list(a.shape) == list(b.shape),
        "max_abs_diff": (a.detach().float().cpu() - b.detach().float().cpu())
                        .abs().max().item() if list(a.shape) == list(b.shape) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    out = {"gpu": gpu, "model": cfg["model"]["target"],
           "draft": cfg["model"]["eagle_draft"]}

    # ---- draft checkpoint keys (what the draft PHYSICALLY owns) ----
    ck = os.path.join(paths["draft_path"], "pytorch_model.bin")
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    out["draft_checkpoint_keys"] = {k: list(v.shape) for k, v in sd.items()}
    out["draft_owns_lm_head_in_checkpoint"] = any("lm_head" in k for k in sd)
    out["draft_owns_embed_in_checkpoint"] = "embed_tokens.weight" in sd

    # ---- live model ----
    model = eagle_bridge.load_eagle_model(paths["target_path"],
                                          paths["draft_path"],
                                          dtype=torch.float16, device_map=DEV)
    tgt_emb = model.base_model.model.embed_tokens.weight
    drf_emb = model.ea_layer.embed_tokens.weight
    lm_head = model.base_model.lm_head.weight
    norm_w = model.base_model.model.norm.weight

    out["embedding_sharing_BEFORE_rotation"] = share_report(tgt_emb, drf_emb)
    out["draft_has_lm_head_module"] = hasattr(model.ea_layer, "lm_head")
    out["draft_modules"] = [n for n, _ in model.ea_layer.named_children()]
    out["gamma_f_stats_BEFORE_rotation"] = tstats(norm_w)
    out["gamma_f_identity"] = "target.base_model.model.norm.weight (final RMSNorm scale of the TARGET; the draft has no such tensor)"

    # snapshots for the after-rotation diff
    snap = {"tgt_emb": tgt_emb.detach().clone().cpu(),
            "drf_emb": drf_emb.detach().clone().cpu(),
            "lm_head": lm_head.detach().clone().cpu(),
            "norm_w": norm_w.detach().clone().cpu()}

    # ---- the code path where lm_head reaches the draft (evidence) ----
    ea = os.path.join(PROJECT_ROOT, "third_party", "EAGLE", "eagle", "model", "ea_model.py")
    cn = os.path.join(PROJECT_ROOT, "third_party", "EAGLE", "eagle", "model", "cnets.py")
    lines = []
    for path, pats in ((ea, ("topK_genrate", "lm_head")),
                       (cn, ("def topK_genrate", "head(", "headweight"))):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if any(p in line for p in pats) and "def sample" not in line:
                    lines.append(f"{os.path.basename(path)}:{i}: {line.rstrip()[:110]}")
    out["lm_head_code_path"] = lines[:24]

    # ---- rotate the TARGET only (full rotate-only pipeline, quant off) ----
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             cfg.get("paths", {}).get("rotations_root"))
    study.apply_rotation_quant(model.base_model, "full", r_bin, "none",
                               cfg["model"]["target"], DEV)
    model.base_model.to(DEV)

    tgt_emb2 = model.base_model.model.embed_tokens.weight
    drf_emb2 = model.ea_layer.embed_tokens.weight
    lm_head2 = model.base_model.lm_head.weight
    norm_w2 = model.base_model.model.norm.weight

    out["AFTER_rotation"] = {
        "target_embed_changed_max_abs": (tgt_emb2.detach().cpu().float()
                                         - snap["tgt_emb"].float()).abs().max().item(),
        "draft_embed_changed_max_abs": (drf_emb2.detach().cpu().float()
                                        - snap["drf_emb"].float()).abs().max().item(),
        "target_lm_head_changed_max_abs": (lm_head2.detach().cpu().float()
                                           - snap["lm_head"].float()).abs().max().item(),
        "target_norm_weight_now": tstats(norm_w2),
        "note_norm": "SpinQuant fuse_layer_norms sets model.norm.weight to ones "
                     "after folding gamma_f into lm_head; gamma_f must be "
                     "stashed BEFORE rotation (stash_original_tensors does).",
        "embedding_sharing_after": share_report(tgt_emb2, drf_emb2),
    }
    out["ANSWERS"] = {
        "1_draft_embed_physically_shared": bool(
            out["embedding_sharing_BEFORE_rotation"]["same_storage_ptr"]),
        "2_lm_head_shared_or_passed": ("draft owns no lm_head module/checkpoint "
                                       "key; the TARGET's head is passed as the "
                                       "`head` argument to topK_genrate at call "
                                       "time (see lm_head_code_path)"),
        "3_draft_embed_auto_rotated": bool(
            out["AFTER_rotation"]["draft_embed_changed_max_abs"] > 0),
        "4_draft_lm_head_auto_rotated": "n/a - draft owns no lm_head; the "
                                        "PASSED head is the target's (rotated) "
                                        "one unless an adapter substitutes it",
        "5_gamma_f": out["gamma_f_stats_BEFORE_rotation"],
    }
    p = os.path.join(args.out_dir, "weight_sharing_audit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ANSWERS"], indent=2))
    print(json.dumps(out["embedding_sharing_BEFORE_rotation"], indent=2))
    print(json.dumps(out["AFTER_rotation"], indent=2, default=str)[:1200])
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
