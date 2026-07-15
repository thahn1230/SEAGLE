"""Required tests (spec §15) for the SpinQuant PPL reproduction study.

CPU-only; validates saved artifacts + loud-failure behaviors. Run:
  python -m pytest tests/test_ppl_reproduction.py -q
"""
import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ART = os.path.join(ROOT, "artifacts", "spinquant_ppl_reproduction_fix")


# ---------- evaluator ----------

def test_official_ppl_window_math():
    """Non-overlapping 2048 windows, per-window mean over 2047 shifted
    positions, PPL = exp(mean of window means) — on a synthetic model."""
    from eagle_spinquant.official_ppl import official_ppl
    V, T = 11, 2048 * 3 + 100          # tail must be truncated
    ids = torch.arange(T) % V
    const = torch.log(torch.ones(V) / V)      # uniform logits -> CE = ln V

    def fwd(w):
        return const.expand(w.shape[0], w.shape[1], V).clone()

    ppl, ce, rows = official_ppl(fwd, ids[None], "cpu")
    assert len(rows) == 3                      # tail truncated
    assert all(r["n_pred"] == 2047 for r in rows)
    assert abs(ce - math.log(V)) < 1e-5
    assert abs(ppl - V) < 1e-3


def test_evaluator_parity_summary_gate_b():
    p = os.path.join(ART, "evaluator_parity", "evaluator_parity_summary.json")
    d = json.load(open(p))
    for m in ("base", "chat"):
        # like-for-like: SAME 32k tokens, project vs official aggregation
        # (the summary's ce_diff_shared field compares against the FULL-set
        # official CE and therefore includes the prefix-subsample effect)
        same_tokens_diff = abs(d[m]["project_agg_shared_tokens"]["ce"]
                               - d[m]["official_agg_32k_prefix"]["ce"])
        assert same_tokens_diff <= 1.0e-4 + 1e-9, \
            f"{m}: same-token aggregation diff {same_tokens_diff} exceeds " \
            "GATE B bound"


def test_official_token_count_and_windows():
    p = os.path.join(ART, "evaluator_parity", "shared_tokens_chat.pt")
    toks = torch.load(p, weights_only=True)
    assert toks.shape[0] == 1
    n = toks.numel()
    assert n // 2048 == 166, f"expected 166 full windows, got {n // 2048}"
    # official contract: NO BOS prepended
    assert toks[0, 0].item() != 1, "BOS found at position 0 (add_bos leaked)"


def test_legacy_prefix_metric_reproduction_record():
    d = json.load(open(os.path.join(
        ART, "audit", "current_ppl_execution_contract.json")))
    rep, orig = d["reproduction"], d["original_values"]
    for k in ("fp16", "rh0_w4a4"):
        assert abs(rep[k]["ce"] - orig[k]["ce"]) < 1e-5
        assert abs(rep[k]["ppl"] - orig[k]["ppl"]) < 1e-3


# ---------- rotations ----------

def _rbin():
    from eagle_spinquant import experiment, study
    p = experiment.resolve_paths(experiment.load_config(None))
    rr = experiment.load_config(None).get("paths", {}).get("rotations_root")
    return study.r_bin_path("random_hadamard", 0, p["target_path"], rr)


def test_r1_r2_shapes_and_layer_count():
    R = torch.load(_rbin(), map_location="cpu", weights_only=False)
    assert R["R1"].shape == (4096, 4096)
    r2 = [k for k in R if k.endswith("self_attn.R2")]
    assert len(r2) == 32
    for k in r2:
        assert R[k].shape == (128, 128)
    # orthogonality (rotation validity)
    I = torch.eye(4096, dtype=torch.float64)
    assert (R["R1"].to(torch.float64) @ R["R1"].to(torch.float64).T - I) \
        .abs().max() < 1e-6


def test_learned_rotation_missing_fails_loudly():
    """No silent fallback to random when a learned rotation is requested."""
    from eagle_spinquant import experiment, study
    p = experiment.resolve_paths(experiment.load_config(None))
    with pytest.raises(FileNotFoundError):
        study.r_bin_path("learned", 0, p["target_path"],
                         "/nonexistent/rotations_root")


def test_official_artifact_format_matches_project():
    off = "/data/thahn1230/spinquant_official_rotations/LLaMA-2-7B/" \
          "7B_W4A4KV16_lr_1.5_seed_0/R.bin"
    if not os.path.exists(off):
        pytest.skip("official artifact not downloaded")
    Ro = torch.load(off, map_location="cpu", weights_only=False)
    Rp = torch.load(_rbin(), map_location="cpu", weights_only=False)
    assert set(Ro.keys()) == set(Rp.keys())


# ---------- pipeline parity (GATE D artifacts) ----------

def test_transformed_weight_parity():
    import csv
    p = os.path.join(ART, "pipeline_parity", "transformed_weight_diff.csv")
    rows = list(csv.DictReader(open(p)))
    assert len(rows) >= 400
    bad = [r for r in rows if r["identical"] != "True"]
    assert not bad, f"{len(bad)} transformed weights differ: {bad[:3]}"


def test_quantizer_config_parity():
    p = os.path.join(ART, "pipeline_parity", "quantizer_config_diff.csv")
    content = open(p).read().strip()
    data_lines = [ln for ln in content.splitlines()[1:] if ln.strip()]
    assert not data_lines, f"quantizer config diffs: {data_lines[:3]}"


def test_gate_d_ppl_within_tolerance():
    d = json.load(open(os.path.join(ART, "pipeline_parity",
                                    "inprocess_official_ppl.json")))
    official = 10.6272
    assert abs(math.log(d["ppl"]) - math.log(official)) < 0.02, \
        "in-process vs official PPL beyond kernel-noise tolerance"


# ---------- labeling ----------

def test_rotation_labeling_terms_in_report():
    txt = open(os.path.join(ROOT, "docs",
                            "SPINQUANT_PPL_REPRODUCTION_AND_FIX_REPORT.md")).read()
    assert "random-Hadamard control" in txt
    assert "learned SpinQuant" in txt
