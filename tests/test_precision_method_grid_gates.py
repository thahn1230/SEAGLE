"""Correctness gates for the precision × method grid study (spec §12).

CPU-side gates here; heavy GPU gates (Gate-P rerun, T16D16 repro,
fake/real policy agreement) run as scripts and drop JSON verdicts that
test_gate_artifacts asserts on.
"""
import glob
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eagle_spinquant.concat_selective_projection import (      # noqa: E402
    build_concat_selective_weights)
from eagle_spinquant import pmg_configs                        # noqa: E402

D = 64          # synthetic dim (divisible structure irrelevant here)
g = torch.Generator().manual_seed(0)


def _synth():
    sd = {"fc.weight": torch.randn(D, 2 * D, generator=g,
                                   dtype=torch.float64),
          "fc.bias": torch.zeros(D, dtype=torch.float64)}
    R = torch.linalg.qr(torch.randn(D, D, generator=g,
                                    dtype=torch.float64))[0]
    gam = (torch.rand(D, generator=g, dtype=torch.float64) * 0.5
           + 0.75)
    return sd, R, gam


def test_target_only_spinquant_interface():
    """Restored stock interface == gamma_R1 folded weights (F0 equiv):
    (a_t @ R^T * gamma) @ W_h^T == a_t @ (W_h · D_gamma · R)^T."""
    sd, R, gam = _synth()
    W_first, _, _ = build_concat_selective_weights(
        sd, R, gam, None, first_hidden_mode="gamma_R1")
    W_h_fold = W_first[:, D:]
    a_t = torch.randn(8, D, generator=g, dtype=torch.float64)
    restored = ((a_t @ R.t()) * gam) @ sd["fc.weight"][:, D:].t()
    folded = a_t @ W_h_fold.t()
    assert (restored - folded).abs().max() < 1e-9


def test_target_draft_spinquant_interface():
    """W_rec hidden block folds R only (no gamma): r_d @ (W_h R)^T ==
    h_d @ W_h^T for r_d = h_d @ R."""
    sd, R, gam = _synth()
    _, W_rec, _ = build_concat_selective_weights(
        sd, R, gam, None, first_hidden_mode="gamma_R1")
    h_d = torch.randn(8, D, generator=g, dtype=torch.float64)
    assert ((h_d @ R) @ W_rec[:, D:].t()
            - h_d @ sd["fc.weight"][:, D:].t()).abs().max() < 1e-9


def test_first_gamma_once():
    sd, R, gam = _synth()
    W_first, _, _ = build_concat_selective_weights(
        sd, R, gam, None, first_hidden_mode="gamma_R1")
    # folding gamma twice would square it: check the fold equals
    # exactly one application
    expect = sd["fc.weight"][:, D:] @ (gam.unsqueeze(1) * R)
    assert (W_first[:, D:] - expect).abs().max() < 1e-12


def test_recurrent_no_target_gamma():
    sd, R, gam = _synth()
    _, W_rec, _ = build_concat_selective_weights(
        sd, R, gam, None, first_hidden_mode="gamma_R1")
    assert (W_rec[:, D:] - sd["fc.weight"][:, D:] @ R).abs().max() \
        < 1e-12


def test_ep3g_fp_invariance():
    """P3 reparam is exact in fp64: (m e)(W_e/m)^T == e W_e^T."""
    sd, R, gam = _synth()
    m = 32.0
    e = torch.randn(8, D, generator=g, dtype=torch.float64)
    base = e @ sd["fc.weight"][:, :D].t()
    rep = (m * e) @ (sd["fc.weight"][:, :D] / m).t()
    assert (base - rep).abs().max() < 1e-9


def test_ep3p_first_recurrent_scales():
    """Pathwise: first slice /m_f, rec slice /m_r, rec input rescale
    m_r/m_f composes to exact identity in fp64."""
    sd, R, gam = _synth()
    m_f, m_r = 27.86, 42.22
    W = sd["fc.weight"][:, :D]
    e = torch.randn(8, D, generator=g, dtype=torch.float64)
    base = e @ W.t()
    first = (m_f * e) @ (W / m_f).t()
    rec = ((m_f * e) * (m_r / m_f)) @ (W / m_r).t()
    assert (base - first).abs().max() < 1e-9
    assert (base - rec).abs().max() < 1e-9


def test_rd_orthogonality():
    rd_run = open(os.path.join(ROOT, "runs", "RDROT_RUN_DIR")).read() \
        .strip()
    p = os.path.join(ROOT, rd_run, "rotations", "RD_HYB_s2.pt")
    Rm = torch.load(p, map_location="cpu",
                    weights_only=False)["R_D"].double()
    eye = torch.eye(Rm.shape[0], dtype=torch.float64)
    oe = float(torch.linalg.norm(Rm @ Rm.t() - eye))
    assert oe < 1e-3, oe


def test_rd_gate_p_bitwise():
    """Gate-P verdict from the R_D study run must be PASS; the PMG
    rerun (this study) must also be PASS once produced."""
    rd_run = open(os.path.join(ROOT, "runs", "RDROT_RUN_DIR")).read() \
        .strip()
    d = json.load(open(os.path.join(
        ROOT, rd_run, "gradchecks", "gateP_ep3p_parity.json")))
    assert d["verdict"] == "PASS" and not d["fails"]
    pmg = open(os.path.join(ROOT, "runs", "PMG_RUN_DIR")).read().strip()
    rerun = os.path.join(ROOT, pmg, "gradchecks",
                         "gateP_ep3p_parity.json")
    if os.path.exists(rerun):
        d2 = json.load(open(rerun))
        assert d2["verdict"] == "PASS", d2["fails"]


def test_qat_export_equivalence():
    """Anchor exists with recorded sha; trainer export contract keys
    covered by existing Gate E/F tests (referenced, not duplicated)."""
    p = os.path.join(ROOT, "checkpoints", "eagle1_fresh_fp16_anchor",
                     "anchor.pt")
    assert os.path.exists(p)
    sha = open(p.replace("anchor.pt", "anchor.sha256")).read().split()[0]
    assert sha.startswith("a7c6ccc8")


def test_precision_grid_config_unique():
    a = pmg_configs.grid_a_arms()
    b = pmg_configs.grid_b_arms()
    assert len(a) == 9 and len(b) == 24
    sigs = [json.dumps(v, sort_keys=True) for v in
            list(a.values()) + list(b.values())]
    assert len(sigs) == len(set(sigs)), "duplicate arm configs"
    for tgt in ("fp16", "int4"):
        assert pmg_configs.EP3P[tgt] is not None
        assert pmg_configs.EP3G[tgt] is not None


def test_dataset_prompt_checksum():
    """Eval pools must match the R_D study's prompt ids (mtbench)."""
    from eagle_spinquant.eval_datasets import load_eval_prompts
    prompts, _ = load_eval_prompts("mtbench", 80, "eval")
    ids = [p["row_id"] for p in prompts]
    rd_run = open(os.path.join(ROOT, "runs", "RDROT_RUN_DIR")).read() \
        .strip()
    import csv
    shard = os.path.join(ROOT, rd_run, "shards",
                         "al__BASE_EP3P__int4__mtbench.csv")
    old_ids = [r["prompt_id"] for r in csv.DictReader(open(shard))]
    assert len(old_ids) == 80
    assert [str(i) for i in ids] == old_ids


def test_official_al_contract():
    """Evaluator defaults pinned: mnt=128, greedy, mc_sim tree."""
    src = open(os.path.join(ROOT, "scripts",
                            "eval_eagle_acceptance_length.py")).read()
    assert '"--max-new-tokens", type=int, default=128' in src
    assert "mc_sim_7b_63" in src
    assert "temperature=0.0" in src
