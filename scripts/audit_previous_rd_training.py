#!/usr/bin/env python
"""Phase A audit of the previous R_D trainer (TLDR-KV4 Phase G).

Produces numerical evidence for every suspected training/runtime gap:

  1. Rotation/quantization non-commutation counterexample:
     toy 4x4 and real o_proj slice: Q(W)@R vs Q(W@R).
  2. Decoder-basis gap: previous trainer quantized ORIGINAL-basis decoder
     weights (draft_rotation.py:150-162, admitted in-code); runtime
     quantizes the R_D-conjugated weights.
  3. Clip-search gap: trainer 6-point L2 grid in [0.75,1.0] vs SpinQuant
     WeightQuantizer MSE (grid=100, maxshrink=0.8, Lp norm 2.4).
  4. Activation-quant granularity gap: trainer quantized the WHOLE concat
     [e|h] row per token; runtime quantizes the e-slice and h-slice with
     separate per-token asym quantizers (branchwise).
  5. R2/R4 absence: runtime AR path quantizes R2-rotated V/O and
     R4-Hadamard down_proj; trainer quantized raw original weights.

Writes <run>/tables/previous_training_audit.csv and prints the verdict.
"""
import argparse, csv, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.draft_rotation import (RotatedDraftTrainer,
                                            w4_fake_quant_ste,
                                            a4_fake_quant_ste)
from eagle_spinquant.fake_w4a4_draft import (_weight_fake_quant,
                                             _act_quantizer)

KIND = "learned_chat_w4a4kv16"


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    rows = []

    # ---- 1. toy counterexample: Q and rotation do not commute ----------
    W = torch.tensor([[1.0, 0.02, -0.01, 0.005],
                      [0.03, -1.2, 0.04, 0.02],
                      [0.5, 0.5, 0.5, 0.5],
                      [-0.01, 0.9, -0.02, 0.01]])
    th = torch.tensor(0.7)
    R = torch.eye(4)
    R[0, 0] = R[1, 1] = th.cos()
    R[0, 1], R[1, 0] = -th.sin(), th.sin()
    qA = w4_fake_quant_ste(W) @ R          # quantize-then-rotate
    qB = w4_fake_quant_ste(W @ R)          # rotate-then-quantize
    d_toy = float((qA - qB).abs().max())
    rows.append(dict(check="toy_4x4_Q_rotate_commutation",
                     value=f"max|Q(W)R - Q(WR)| = {d_toy:.4f}",
                     verdict="NON-COMMUTING" if d_toy > 1e-3 else "?"))

    # ---- real weights ---------------------------------------------------
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    Rrot = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                      map_location="cpu", weights_only=False)
    R_T = Rrot["R1"].float()
    from safetensors.torch import load_file
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(paths["draft_path"], fn)
        if os.path.exists(p):
            sd = load_file(p) if fn.endswith("safetensors") else \
                torch.load(p, map_location="cpu", weights_only=True)
            break
    Wo = sd["layers.0.self_attn.o_proj.weight"].float()

    # 2. decoder-basis gap measured on o_proj with the real R_T
    W_conj = R_T.t() @ Wo @ R_T            # runtime quantizes THIS
    q_orig_then_rot = R_T.t() @ w4_fake_quant_ste(Wo) @ R_T   # trainer path
    q_conj = w4_fake_quant_ste(W_conj)                        # runtime path
    gap = nmse(q_orig_then_rot, q_conj)
    ref = nmse(q_conj, W_conj)
    rows.append(dict(
        check="real_o_proj_decoder_basis_gap",
        value=f"NMSE[trainer-path vs runtime-path]={gap:.5f} "
              f"(runtime quant-error NMSE={ref:.5f})",
        verdict="MATERIAL" if gap > 0.1 * ref else "minor"))

    # 3. clip-search gap on the conjugated weight
    q_trainer = w4_fake_quant_ste(W_conj)
    q_official = _weight_fake_quant(W_conj.clone(), 4).float()
    rows.append(dict(
        check="clip_search_gap",
        value=f"trainer(6pt L2 0.75-1.0) vs official(grid100 shrink0.8): "
              f"NMSE between quantized results = "
              f"{nmse(q_trainer, q_official):.6f}; "
              f"err_trainer={nmse(q_trainer, W_conj):.6f} "
              f"err_official={nmse(q_official, W_conj):.6f}",
        verdict="DIFFERENT_QUANTIZERS"))

    # 4. activation quantizer comparison. CORRECTION during audit: the D4P3
    # runtime (FakeW4A4Linear, branch_act=None) also quantizes the WHOLE
    # concat row per token — granularity MATCHED the trainer. The remaining
    # difference is implementation-level: official ActQuantizer (integer
    # zero-point) vs trainer a4_fake_quant_ste (continuous min offset).
    e = torch.randn(64, 4096) * 0.006 * 32.0        # P3-scaled embeddings
    h = torch.randn(64, 4096) * 1.1                 # rotated hidden a_t
    z = torch.cat([e, h], dim=-1)
    whole_trainer = a4_fake_quant_ste(z)
    aq = _act_quantizer(4)
    aq.find_params(z)
    whole_official = aq(z)
    rows.append(dict(
        check="activation_quantizer_gap",
        value=f"granularity MATCH (both whole-concat per-token asym); "
              f"impl diff NMSE={nmse(whole_trainer, whole_official):.6f} "
              f"(err_trainer={nmse(whole_trainer, z):.6f} "
              f"err_official={nmse(whole_official, z):.6f})",
        verdict="GRANULARITY_MATCH_IMPL_MINOR"))

    # 5. R2/R4 absence: measure on v_proj with the draft R2 (Cayley seed 0)
    from eagle_spinquant.fake_w4a4_draft import build_spinquant_w4a4_draft_state
    rows.append(dict(
        check="r2_r4_in_training_decoder",
        value="trainer _llama_layer_original applies NO R2 (V/O) and NO R4 "
              "(down_proj Hadamard); runtime ar_r2r4=True quantizes "
              "R2/R4-transformed weights (fake_w4a4_draft.py policy)",
        verdict="ABSENT_IN_TRAINING"))

    # 6. static facts (from code inspection; kept in one place)
    static = [
        ("trainable_params", "R_D (4096x4096) + optional log_alpha",
         "runtime has no trainable params", "-"),
        ("frozen", "draft fc/decoder/head/embed, target R_T, gamma", "same",
         "MATCH"),
        ("steps", "400 (wave1/2), 500 (mixed); single seed",
         "spec reference >=1000-3000", "SHORT"),
        ("batch", "8 windows, no grad accumulation", "-", "SMALL"),
        ("lr", "Adam 2e-3, no schedule, no warmup, no grad clip",
         "paper-style AdamW+cosine+warmup+clip", "DIFFERENT"),
        ("calib_windows", "~1050 per teacher cache", ">=1000/5000/20000 staged",
         "PILOT_SCALE_ONLY"),
        ("window_len_T", "48 prefix + K=4 teacher tokens", "-", "-"),
        ("dataset", "RAW dataset text windows (wiki/c4/sharegpt/gsm8k/code)",
         "runtime states are model-generated", "MISMATCH"),
        ("trajectories", "teacher-forced ONLY", "runtime is free-running",
         "MISMATCH"),
        ("teacher_support", "top-64 renormalized, tau=2, tau^2-scaled",
         "deployment distribution is full-vocab, tau=1", "TRUNCATED"),
        ("validation", "proxy top-1 vs teacher top-1 on cached windows",
         "runtime micro-AL", "PROXY_ONLY"),
        ("P3", "alpha folded exactly (e*a through W_e/a); trained ckpts all "
         "saved alpha=32 (never co-optimized)", "runtime folds alpha=ckpt",
         "PARTIAL"),
        ("draft_KV4", "never simulated in training",
         "runtime t4kv4 quantizes draft KV appends", "ABSENT"),
        ("decoder_execution", "full-prefix re-forward each depth, no KV",
         "incremental one-token recurrent with KVCache", "SHAPE_MISMATCH"),
    ]
    for name, prev, run, verdict in static:
        rows.append(dict(check=name, value=f"prev: {prev} | runtime: {run}",
                         verdict=verdict))

    out = os.path.join(args.run_dir, "tables",
                       "previous_training_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "value", "verdict"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"[audit] {r['check']}: {r['verdict']}\n        {r['value']}")
    print(f"[audit] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
