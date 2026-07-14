#!/usr/bin/env python
"""Mathematical validation of the rotation interface (Phase 3 gate).

Validates, on the REAL Llama-2-7B-chat + EAGLE-1 draft (quantization OFF,
random-Hadamard R, seed 0):

  1. Orthogonality of R1, all per-layer R2, and the head_dim Hadamard (R3 form).
  2. QK invariance under shared R3.
  3. Interface inverse on real hiddens: unrotate WITH vs WITHOUT gamma_f,
     including downstream draft-feature/logit effects (gamma is not optional).
  4. Variant B single-forward identity.
  5. Level-wise diagnostics for B and B2 during real tree expansion
     (external level 1 vs recycled levels 2..5) -> hidden_diagnostics.csv.
     This is the direct test of the recycling explanation (H6) and the
     two-path fix (H7).

Outputs:
  runs/<run_id>/rotation_interface_validation.json
  runs/<run_id>/hidden_diagnostics.csv

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/validate_rotation_interface.py --run-id <id>
"""

import argparse
import copy
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, hadamard_shim,  # noqa: E402
                             logging_utils, spinquant_bridge as sb, study)

DEV = "cuda:0"
DTYPE = torch.float16


def t_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.double().flatten(); b = b.double().flatten()
    diff = (a - b).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "relative_l2_error": (torch.norm(a - b) / (torch.norm(a) + 1e-12)).item(),
        "cosine_similarity": F.cosine_similarity(a, b, dim=0).item(),
    }


def head_agreement(head, fa, fb) -> dict:
    """draft_logits_kl is the MEAN per-position KL (review fix: batchmean with
    batch=1 summed over sequence positions, making levels incomparable)."""
    la = head(fa.float()); lb = head(fb.float())
    pa = torch.log_softmax(la, -1); pb = torch.log_softmax(lb, -1)
    n_pos = int(la.numel() // la.shape[-1])
    kl = (F.kl_div(pb, pa, log_target=True, reduction="sum") / max(n_pos, 1)).item()
    top1 = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    return {"draft_logits_kl": kl, "draft_top1_agreement": top1}


@torch.no_grad()
def original_hiddens(target_path, prompts_ids):
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    m = KVLlama.from_pretrained(target_path, torch_dtype=DTYPE,
                                low_cpu_mem_usage=True).to(DEV).eval()
    hs, next_tok = [], []
    for ids in prompts_ids:
        out = m.model(input_ids=ids.to(DEV))
        h = out[0]
        logits = m.lm_head(h)
        hs.append(h.float().cpu())
        next_tok.append(int(logits[0, -1].argmax()))
    del m; torch.cuda.empty_cache()
    return hs, next_tok


def build_standalone_draft(draft_path, device, dtype):
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    cfg = EConfig.from_pretrained(os.path.join(draft_path, "config.json"))
    d = Model(cfg, bias=True)
    sd = torch.load(os.path.join(draft_path, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    d.load_state_dict(sd, strict=True)
    d = d.to(device=device, dtype=dtype).eval()
    d.diff_device = False
    d.init_tree()
    d.reset_kv()
    return d


@torch.no_grad()
def level_capture(draft, hidden, input_ids_full, head):
    """Run one topK_genrate; return list of per-call draft output features.
    Call 1 = external hidden; calls 2..L = recycled features."""
    feats = []
    orig_fwd = draft.forward

    def rec_fwd(hs, *a, **k):
        out = orig_fwd(hs, *a, **k)
        o = out[0] if isinstance(out, tuple) else out
        feats.append(o.detach().float().cpu())
        return out
    draft.forward = rec_fwd
    draft.reset_kv(); draft.reset()
    draft.topK_genrate(hidden, input_ids_full, head, None)
    draft.forward = orig_fwd
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--config", default=None,
                    help="experiment yaml (default: configs/default_experiment.yaml)")
    ap.add_argument("--chat-template", default=None,
                    choices=["llama2", "vicuna"])
    args = ap.parse_args()
    # weights are mutated under inference_mode during rotation; any later
    # grad-recording forward would raise -> disable autograd globally.
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    print("[validate] GPU policy OK:", gpu, flush=True)

    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    chat_template = (args.chat_template
                     or cfg.get("model", {}).get("chat_template", "llama2"))
    build_prompt = eagle_bridge.PROMPT_BUILDERS[chat_template]
    results = {}

    # ---- 1. orthogonality ----
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             rotations_root)
    R = torch.load(r_bin, map_location="cpu", weights_only=False)
    I = torch.eye(4096, dtype=torch.float64)
    o = {"R1": (R["R1"].double().T @ R["R1"].double() - I).abs().max().item()}
    r2errs = []
    I128 = torch.eye(128, dtype=torch.float64)
    for k, v in R.items():
        if "R2" in k:
            r2errs.append((v.double().T @ v.double() - I128).abs().max().item())
    o["R2_max_over_layers"] = max(r2errs)
    H = hadamard_shim._sylvester_hadamard(128, "cpu") / (128 ** 0.5)  # R3 form
    o["R3_hadamard_128"] = (H.T @ H - I128).abs().max().item()
    results["orthogonality_max_abs_dev_from_I"] = o

    # ---- 2. QK invariance under shared R3 ----
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(64, 128, generator=g, dtype=torch.float64)
    K = torch.randn(64, 128, generator=g, dtype=torch.float64)
    results["qk_invariance_max_abs"] = ((Q @ H) @ (K @ H).T - Q @ K.T).abs().max().item()

    # ---- prompts / original hiddens ----
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"])
    ids_list = [build_prompt(tok, p["text"]) for p in prompts]
    print("[validate] computing ORIGINAL hiddens...", flush=True)
    h_orig, next_toks = original_hiddens(paths["target_path"], ids_list)

    # ---- rotated target (rotate-only, full components) ----
    print("[validate] building ROTATED target (quant OFF)...", flush=True)
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=DTYPE,
                                  low_cpu_mem_usage=True).to(DEV).eval()
    stash = study.apply_rotation_quant(rot, "full", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    R1 = stash["R1"].to(DEV).float()
    gamma = stash["gamma_f"].to(DEV).float()
    from eagle_spinquant.rotation_interface import build_original_head
    head = build_original_head(stash["lm_head_weight"], DEV, torch.float32)

    # ---- 3. interface inverse with vs without gamma ----
    inv_rows = {"with_gamma": [], "no_gamma": [], "naive": []}
    hhat_list = []
    for ids, h in zip(ids_list, h_orig):
        hh = rot.model(input_ids=ids.to(DEV))[0]
        hhat_list.append(hh.float().cpu())
        hh = hh.float()
        rec_g = (hh @ R1.t()) * gamma
        rec_n = hh @ R1.t()
        href = h.to(DEV)
        inv_rows["with_gamma"].append(t_stats(href, rec_g))
        inv_rows["no_gamma"].append(t_stats(href, rec_n))
        inv_rows["naive"].append(t_stats(href, hh))
    def agg(rows):
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    results["interface_inverse"] = {k: agg(v) for k, v in inv_rows.items()}

    # downstream draft-feature effect of gamma (single forward through draft)
    print("[validate] draft-side gamma ablation + B identity + level-wise...", flush=True)
    draft_ref = build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    ids0 = ids_list[0].to(DEV)
    full_ids0 = torch.cat([ids0, torch.tensor([[next_toks[0]]], device=DEV)], 1)
    hh0 = hhat_list[0].to(DEV)
    h0 = h_orig[0].to(DEV)

    def first_feat(d, hidden):
        d.reset_kv(); d.reset()
        out, _ = d(hidden.float(), input_ids=full_ids0[:, 1:], use_cache=True)
        return out[:, -1]
    f_true = first_feat(draft_ref, h0)
    f_g = first_feat(draft_ref, (hh0 @ R1.t()) * gamma)
    f_n = first_feat(draft_ref, hh0 @ R1.t())
    results["gamma_ablation_draft_side"] = {
        "with_gamma_vs_true": {**t_stats(f_true, f_g), **head_agreement(head, f_true, f_g)},
        "no_gamma_vs_true": {**t_stats(f_true, f_n), **head_agreement(head, f_true, f_n)},
    }

    # ---- 4. Variant B single-forward identity ----
    draft_B = copy.deepcopy(draft_ref)
    M = study.fold_matrix(R1, gamma).to(DEV)
    D = 4096
    draft_B.fc.weight.data[:, D:] = (draft_B.fc.weight.data[:, D:].double() @ M.double()).float()
    f_B = first_feat(draft_B, hh0)
    results["variantB_single_forward_identity"] = {
        **t_stats(f_g, f_B), **head_agreement(head, f_g, f_B)}

    # ---- 5. level-wise diagnostics (B, B2 vs reference A) ----
    diag_rows = []
    for pi in range(min(4, len(ids_list))):
        ids = ids_list[pi].to(DEV)
        full_ids = torch.cat([ids, torch.tensor([[next_toks[pi]]], device=DEV)], 1)
        hh = hhat_list[pi].to(DEV)
        h_A = (hh.float() @ R1.t()) * gamma
        feats_A = level_capture(draft_ref, h_A, full_ids, head)
        feats_B = level_capture(draft_B, hh.float(), full_ids, head)
        # B2 = two-path: folded for external call, original for recycled.
        draft_B2 = copy.deepcopy(draft_ref)
        Wo = draft_B2.fc.weight.data
        Wf = Wo.clone(); Wf[:, D:] = (Wo[:, D:].double() @ M.double()).float()
        state = {"first": True}
        orig_fwd = draft_B2.forward
        def b2_fwd(hs, *a, **k):
            draft_B2.fc.weight.data = Wf if state["first"] else Wo
            state["first"] = False
            return orig_fwd(hs, *a, **k)
        draft_B2.forward = b2_fwd
        feats_B2 = level_capture(draft_B2, hh.float(), full_ids, head)
        draft_B2.forward = orig_fwd

        for name, feats in (("B", feats_B), ("B2", feats_B2)):
            for lvl, (fa, fb) in enumerate(zip(feats_A, feats), start=1):
                if fa.shape != fb.shape:
                    diag_rows.append({"prompt_id": prompts[pi]["question_id"],
                                      "variant": name, "tree_level": lvl,
                                      "notes": f"shape mismatch {fa.shape} vs {fb.shape}"})
                    continue
                st = t_stats(fa, fb)
                ha = head_agreement(head, fa.to(DEV), fb.to(DEV))
                diag_rows.append({
                    "prompt_id": prompts[pi]["question_id"],
                    "feature_source": "external_target_hidden" if lvl == 1 else "recycled_draft_feature",
                    "variant": name, "tree_level": lvl,
                    "cosine_to_reference_hidden": st["cosine_similarity"],
                    "relative_l2_to_reference_hidden": st["relative_l2_error"],
                    "max_abs_error": st["max_abs_error"],
                    "mean_abs_error": st["mean_abs_error"],
                    "draft_logits_kl": ha["draft_logits_kl"],
                    "draft_top1_agreement": ha["draft_top1_agreement"],
                    "target_logits_kl": "",  # N/A in single-round diagnostics
                    "target_top1_agreement": "",
                    "notes": ("reference = Variant A features on same prompt"
                              + ("; CONFOUNDED: token paths diverge after B's "
                                 "level-2 corruption, so levels>=3 mix token-"
                                 "selection divergence with basis error"
                                 if (name == "B" and lvl >= 3) else "")),
                })
    logging_utils.write_csv(os.path.join(run_dir, "hidden_diagnostics.csv"), diag_rows)

    # verdicts
    wg = results["interface_inverse"]["with_gamma"]["relative_l2_error"]
    ng = results["interface_inverse"]["no_gamma"]["relative_l2_error"]
    results["verdicts"] = {
        # 1e-6 = fp32 storage roundoff scale: SpinQuant's per-layer R2 matrices
        # are generated/stored in fp32 (measured 3.4e-8); R1/R3 are fp64 (0.0 /
        # 1e-16). Orthogonal "at storage precision".
        "orthogonality_ok": max(o["R1"], o["R2_max_over_layers"], o["R3_hadamard_128"]) < 1e-6,
        "qk_invariance_ok": results["qk_invariance_max_abs"] < 1e-9,
        "gamma_required": ng > 10 * wg,
        "B_single_forward_exact": results["variantB_single_forward_identity"]["relative_l2_error"] < 1e-3,
        "B_recycled_levels_diverge": _lvl_verdict(diag_rows, "B"),
        "B2_recycled_levels_match": _lvl_verdict(diag_rows, "B2", expect_match=True),
    }
    out = os.path.join(run_dir, "rotation_interface_validation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["verdicts"], indent=2))
    print(f"[validate] -> {out}")
    print(f"[validate] -> {run_dir}/hidden_diagnostics.csv ({len(diag_rows)} rows)")
    return 0


def _lvl_verdict(rows, variant, expect_match=False):
    """Divergence verdict uses ONLY level 2: it is the clean comparison (same
    token path, same input features, only the fc basis differs). Levels >= 3
    are confounded by token-path divergence once level 2 corrupts (annotated
    in the CSV) and must not drive the verdict."""
    lv1 = [r for r in rows if r.get("variant") == variant and r.get("tree_level") == 1
           and isinstance(r.get("cosine_to_reference_hidden"), float)]
    lv2 = [r for r in rows if r.get("variant") == variant and r.get("tree_level") == 2
           and isinstance(r.get("cosine_to_reference_hidden"), float)]
    lv2p = [r for r in rows if r.get("variant") == variant and (r.get("tree_level") or 0) >= 2
            and isinstance(r.get("cosine_to_reference_hidden"), float)]
    if not lv1 or not lv2:
        return None
    c1 = sum(r["cosine_to_reference_hidden"] for r in lv1) / len(lv1)
    c2 = sum(r["cosine_to_reference_hidden"] for r in lv2) / len(lv2)
    c2p = sum(r["cosine_to_reference_hidden"] for r in lv2p) / len(lv2p)
    if expect_match:
        return bool(c1 > 0.999 and c2p > 0.99)  # B2: all levels clean by design
    return bool(c1 > 0.999 and c2 < 0.9)


if __name__ == "__main__":
    sys.exit(main())
