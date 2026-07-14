#!/usr/bin/env python
"""Basis ledger: PROVE which tensors live in which basis (spec §3).

For n prompts, collects: h, h_hat, T_h_inv(h_hat), draft external input,
level-1 output feature, level-2/3 recycled inputs — for (a) the reference
original pipeline and (b) the Variant-B configuration (folded fc, NO recycled
conversion). Every tensor row gets cosine/rel-L2 probes against the original-
basis and rotated-basis references plus top-1 probes under both heads.

Also verifies numerically that the stash-built rotated head
(W @ diag(gamma_f) @ R1) equals the actual post-fusion rotated model lm_head.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_basis_ledger.py \
      --out-dir runs/rotation_aware_audit_<ts> --num-prompts 4
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             rotation_aware as ra, study)
from eagle_spinquant.rotation_interface import build_original_head  # noqa: E402

DEV = "cuda:0"
DTYPE = torch.float16


def cos(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def rel_l2(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return (torch.norm(a - b) / (torch.norm(b) + 1e-12)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    ra.verify_fold_algebra()
    print("[ledger] fold algebra verified; GPU:", gpu, flush=True)

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"])
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    ids_list = [eagle_bridge.PROMPT_BUILDERS[tmpl](tok, p["text"]) for p in prompts]

    # ---- original target hiddens ----
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    print("[ledger] original hiddens...", flush=True)
    m0 = KVLlama.from_pretrained(paths["target_path"], torch_dtype=DTYPE,
                                 low_cpu_mem_usage=True).to(DEV).eval()
    h_list, next_toks, W_head_orig = [], [], m0.lm_head.weight.detach().cpu().clone()
    for ids in ids_list:
        h = m0.model(input_ids=ids.to(DEV))[0]
        next_toks.append(int(m0.lm_head(h)[0, -1].argmax()))
        h_list.append(h.float().cpu())
    del m0
    torch.cuda.empty_cache()

    # ---- rotated target ----
    print("[ledger] rotated target...", flush=True)
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             cfg.get("paths", {}).get("rotations_root"))
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=DTYPE,
                                  low_cpu_mem_usage=True).eval()
    stash = study.apply_rotation_quant(rot, "full", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    rot.to(DEV)
    R1 = stash["R1"].cpu().double()
    gamma = stash["gamma_f"].cpu().double()
    hhat_list = [rot.model(input_ids=ids.to(DEV))[0].float().cpu()
                 for ids in ids_list]
    W_head_rotated_actual = rot.lm_head.weight.detach().cpu().clone()
    del rot
    torch.cuda.empty_cache()

    # rotated-head consistency: stash-built vs actual fused+rotated lm_head
    W_rot_built = ra.in_fold(W_head_orig, R1, gamma)
    head_consistency = {
        "max_abs_diff": (W_rot_built - W_head_rotated_actual.double())
                        .abs().max().item(),
        "rel_l2": (W_rot_built - W_head_rotated_actual.double()).norm().item()
                  / W_head_rotated_actual.double().norm().item(),
    }

    # heads on GPU (fp32)
    head_o = build_original_head(W_head_orig, DEV, torch.float32)
    head_r = ra.build_rotated_head(W_head_orig, R1, gamma, DEV, torch.float32)

    # ---- drafts: reference (original) and B-config (folded, no conversion) ----
    draft_ref = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    draft_B = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    M = study.fold_matrix(R1.float().to(DEV), gamma.float().to(DEV)).float()
    Dh = 4096
    draft_B.fc.weight.data[:, Dh:] = (draft_B.fc.weight.data[:, Dh:].double()
                                      @ M.double()).float()

    rows = []

    def add_row(name, source, expected, t, prompt_id, refs, f_ref=None,
                head_probe=None, note=""):
        """refs = dict with h, h_hat (matching shape slices)."""
        r = {"prompt_id": prompt_id, "name": name, "source": source,
             "expected_basis": expected, "shape": str(list(t.shape)),
             "norm": t.double().norm().item()}
        r["cosine_to_h"] = cos(t, refs["h"]) if refs.get("h") is not None and t.shape == refs["h"].shape else ""
        r["cosine_to_h_hat"] = cos(t, refs["h_hat"]) if refs.get("h_hat") is not None and t.shape == refs["h_hat"].shape else ""
        r["rel_l2_to_h"] = rel_l2(t, refs["h"]) if r["cosine_to_h"] != "" else ""
        r["rel_l2_to_h_hat"] = rel_l2(t, refs["h_hat"]) if r["cosine_to_h_hat"] != "" else ""
        if f_ref is not None and t.shape == f_ref.shape:
            r["cosine_to_f_ref"] = cos(t, f_ref)
            r["cosine_to_T_h_f_ref"] = cos(t, ra.t_h(f_ref, R1, gamma).float())
            co, cr = r["cosine_to_f_ref"], r["cosine_to_T_h_f_ref"]
            r["actual_basis_probe"] = ("original" if co > max(0.9, cr) else
                                       "rotated" if cr > max(0.9, co) else
                                       "NEITHER/corrupted")
        else:
            co = r["cosine_to_h"] or -9
            cr = r["cosine_to_h_hat"] or -9
            if co != -9 or cr != -9:
                r["actual_basis_probe"] = ("original" if co > max(0.9, cr if cr != -9 else -9)
                                           else "rotated" if cr != -9 and cr > 0.9
                                           else "NEITHER/corrupted")
            else:
                r["actual_basis_probe"] = "n/a"
        if head_probe is not None:
            x = head_probe.to(DEV).float()
            ref_tok = head_probe_ref
            r["top1_agreement_original_head"] = float(
                (head_o(x).argmax(-1) == ref_tok).float().mean().item())
            r["top1_agreement_rotated_head"] = float(
                (head_r(x).argmax(-1) == ref_tok).float().mean().item())
        r["notes"] = note
        rows.append(r)

    for pi, (ids, h, hh) in enumerate(zip(ids_list, h_list, hhat_list)):
        pid = prompts[pi]["question_id"]
        full_ids = torch.cat([ids, torch.tensor([[next_toks[pi]]])], 1).to(DEV)
        refs = {"h": h, "h_hat": hh}
        rec = ra.t_h_inv(hh, R1, gamma).float()

        # reference levels (original everything)
        ins_ref, outs_ref = ra.level_capture(draft_ref, h.to(DEV), full_ids, head_o)
        # ground-truth token per position of level-1 output
        global head_probe_ref
        head_probe_ref = head_o(outs_ref[0].to(DEV).float()).argmax(-1)

        add_row("h", "original target model.norm output", "original",
                h, pid, refs, note="reference")
        add_row("h_hat", "rotated target output", "rotated(S)", hh, pid, refs,
                note=f"cos(h_hat, T_h(h)) = {cos(hh, ra.t_h(h, R1, gamma).float()):.6f}")
        add_row("T_h_inv(h_hat)", "runtime unrotation (A)", "original",
                rec, pid, refs, note="should match h")

        # B-config levels (external = h_hat, folded fc, no conversion)
        ins_B, outs_B = ra.level_capture(draft_B, hh.to(DEV), full_ids, head_o)
        add_row("draft_external_input_B", "h_hat fed to folded fc",
                "rotated(S)", ins_B[0], pid, refs)
        add_row("draft_level1_output_B", "folded fc + layer (single forward)",
                "original", outs_B[0], pid, refs, f_ref=outs_ref[0],
                head_probe=outs_B[0], note="B is single-forward exact")
        for lvl in (1, 2):
            if lvl < len(ins_B) and ins_B[lvl].shape == ins_ref[lvl].shape:
                add_row(f"draft_level{lvl+1}_recycled_input_B",
                        "recycled raw draft output", "original (this is B's bug)",
                        ins_B[lvl], pid, refs, f_ref=ins_ref[lvl],
                        note="CONFOUNDED by token-path divergence" if lvl >= 2 else
                             "clean comparison (same token path at level 2 entry)")
            if lvl < len(outs_B) and outs_B[lvl].shape == outs_ref[lvl].shape:
                add_row(f"draft_level{lvl+1}_output_B", "folded fc on WRONG-basis input",
                        "corrupted", outs_B[lvl], pid, refs, f_ref=outs_ref[lvl],
                        head_probe=None)

    logging_utils.write_csv(os.path.join(args.out_dir, "basis_ledger.csv"), rows)

    # summary verdicts
    import statistics as st
    def col(name, key):
        v = [r[key] for r in rows if r["name"] == name and isinstance(r[key], float)]
        return st.mean(v) if v else float("nan")
    verdict = {
        "rotated_head_consistency_max_abs": head_consistency["max_abs_diff"],
        "h_hat_is_T_h_of_h": col("h_hat", "cosine_to_h_hat"),
        "recovered_matches_h_cos": col("T_h_inv(h_hat)", "cosine_to_h"),
        "level1_output_matches_ref_cos": col("draft_level1_output_B", "cosine_to_f_ref"),
        "level2_recycled_input_original_basis_cos": col(
            "draft_level2_recycled_input_B", "cosine_to_f_ref"),
        "level2_recycled_input_rotated_basis_cos": col(
            "draft_level2_recycled_input_B", "cosine_to_T_h_f_ref"),
        "level2_output_corrupted_cos": col("draft_level2_output_B", "cosine_to_f_ref"),
    }
    md = ["# Basis ledger summary", "",
          f"n = {args.num_prompts} prompts, model {cfg['model']['target']}", "",
          "| check | value | reading |", "|---|---|---|",
          f"| stash-built rotated head vs actual fused lm_head (max abs) | {verdict['rotated_head_consistency_max_abs']:.3e} | must be ~fp16 roundoff |",
          f"| cos(h_hat, T_h(h)) | {verdict['h_hat_is_T_h_of_h']:.6f} | h_hat IS T_h(h) |",
          f"| cos(T_h_inv(h_hat), h) | {verdict['recovered_matches_h_cos']:.6f} | A's inverse is exact |",
          f"| B level-1 output vs reference (cos) | {verdict['level1_output_matches_ref_cos']:.6f} | single-forward exact |",
          f"| B level-2 RECYCLED INPUT vs original-basis ref (cos) | {verdict['level2_recycled_input_original_basis_cos']:.6f} | recycled features are ORIGINAL basis |",
          f"| B level-2 RECYCLED INPUT vs rotated-basis ref (cos) | {verdict['level2_recycled_input_rotated_basis_cos']:.6f} | ...and NOT rotated basis |",
          f"| B level-2 OUTPUT vs reference (cos) | {verdict['level2_output_corrupted_cos']:.6f} | wrong-basis input corrupts the fold |",
          "", "Full per-tensor table: basis_ledger.csv"]
    with open(os.path.join(args.out_dir, "basis_ledger_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out_dir, "basis_ledger_verdicts.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"[ledger] -> {args.out_dir}/basis_ledger.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
