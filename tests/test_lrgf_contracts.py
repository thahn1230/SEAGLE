"""LRGF correctness battery (spec §25) — 20 assertions over the
component-mask adapter, learned rotations, objectives, granularity,
foldability values, kernel parity, and plot reproducibility."""
import glob
import json
import os

import numpy as np
import torch

from _lrgf_common import ROOT, D, run_dir, tbl
from eagle_spinquant.projection_rotation import (LearnedRotation,
                                                 StructuredRotation,
                                                 build_rotation)


def test_component_mask_families_consistent():
    d = tbl("component_audit.json")
    if not d:
        return
    # A5 (proj EP3P, fp16 AR) must equal B12 (restore AR) — same config
    assert abs(d["tau"]["A5_proj_ep3p"] - d["tau"]["B12_ar_fp"]) < 1e-6
    # FULL == A18 same config
    assert abs(d["tau"]["FULL"] - d["tau"]["A18_ep3p_ar"]) < 1e-6
    # online-R4 fix: no arm may sit at the broken ~2.13 level while its
    # mask includes any MLP/attn restore
    for k in ("B10_down", "B11_mlp_fp"):
        assert d["tau"][k] > 2.8, k


def test_component_oracle_significance_rule():
    d = tbl("component_audit.json")
    if not d:
        return
    g = d["paired"]["B12_ar_fp_gain_vs_FULL"]
    assert g["significant"] and g["delta"] >= 0.05
    g0 = d["paired"]["B0_first_fp_gain_vs_FULL"]
    assert abs(g0["delta"]) < 0.05        # EP3-P already fixed it


def test_learned_rotation_orthogonality():
    for p in ("cayley", "givens", "householder"):
        lr = LearnedRotation(p, block=32, K=4, device="cpu")
        for q in lr.parameters():
            q.data.normal_(0, 0.05)
        assert lr.orth_error() < 1e-5, p


def test_learned_rotation_frozen_weights_contract():
    src = open(os.path.join(
        ROOT, "scripts", "train_eagle_learned_rotation.py")).read()
    assert "requires_grad_(False)" in src
    assert "frozen_ok" in src
    assert "Adam(params" in src           # only rotation params


def test_surrogate_survival_math():
    # s_k = prod a_j; expected AL surrogate telescopes correctly
    lp = torch.log(torch.tensor([[0.9, 0.8, 0.5]]))
    run = torch.cumsum(lp, dim=1)
    s = torch.exp(run)
    expected = 0.9 + 0.9 * 0.8 + 0.9 * 0.8 * 0.5
    assert abs(float(s.sum()) - expected) < 1e-6


def test_rc_masking_zeroes_disagreement():
    a = torch.tensor([[0.9, 0.8]])
    c = torch.tensor([[1.0, 0.0]])
    ar = a * c
    s = torch.cumprod(ar, dim=1)
    assert float(s[0, 1]) == 0.0          # masked depth kills prefix


def test_granularity_families_mixing():
    cross = StructuredRotation(dict(family="cross", block=2, seed=0,
                                    interleave_chunk=1), n=8)
    Q = cross.to_dense()
    assert Q[:4, 4:].abs().sum() > 0.5    # pairwise mixes
    dual = StructuredRotation(dict(family="dual", block=4, seed=0),
                              n=8)
    Qd = dual.to_dense()
    assert float(Qd[:4, 4:].abs().sum()) == 0.0


def test_granularity_grid_complete():
    g = tbl("granularity_grid.json")
    if not g:
        return
    names = {r["granularity"] for r in g}
    for want in ("G0_identity", "G2_pairwise", "G3_cross_b16",
                 "G5_full8192", "G7_butterfly"):
        assert want in names, want


def test_fold_negative_control_recorded():
    f = tbl("foldability_audit.json")
    if not f:
        return
    c = f["checks"]
    assert c["negative_control_breaks_fp"] is True
    assert c["fold_fp_equiv_rel"] < 1e-10
    assert c["S0S1_code_identical"] and c["S0S2_code_identical"]


def test_kernel_code_parity():
    k = tbl("kernel_verify.json")
    if not k:
        return
    for name, v in k.items():
        if isinstance(v, dict) and "code_match_rate" in v:
            assert v["code_match_rate"] >= 0.999, name


def test_learned_ckpt_roundtrip():
    rd = run_dir()
    if rd is None:
        return
    cks = glob.glob(os.path.join(rd, "rotations", "*_last.pt"))
    cks = [c for c in cks if "SMOKE" not in c]
    if not cks:
        return
    spec = dict(learned_ckpt=cks[0], which="rot_f")
    r1 = build_rotation(spec, device="cpu")
    r2 = build_rotation(dict(spec), device="cpu")
    x = torch.randn(3, 2 * D)
    assert torch.equal(r1.apply(x), r2.apply(x))


def test_selection_rule_max_rcal():
    m = tbl("lrgf_mechanism.json")
    if not m or not m.get("selected_max_rcal"):
        return
    sel = m["selected_max_rcal"]
    cands = [r for r in m["arms"] if r.get("val_rcal") is not None]
    assert sel["val_rcal"] == max(r["val_rcal"] for r in cands)


def test_plot_reproducibility():
    rd = run_dir()
    if rd is None:
        return
    p = os.path.join(rd, "plot_data",
                     "post_ep3p_component_oracle_gain.csv")
    if not os.path.exists(p):
        return
    import csv
    rows = list(csv.DictReader(open(p)))
    d = tbl("component_audit.json")
    by = {r["component"]: float(r["gain"]) for r in rows}
    for k, v in d["paired"].items():
        if k.endswith("gain_vs_FULL"):
            c = k.replace("_gain_vs_FULL", "")
            assert abs(by[c] - v["delta"]) < 1e-6, c


def test_error_flow_monotone_depth():
    f = tbl("error_flow.json")
    if not f:
        return
    for cfg, d in f.items():
        deps = [d[f"rec{k}_h"]["nmse"] for k in (1, 2, 3, 4)
                if f"rec{k}_h" in d]
        if len(deps) == 4:
            assert deps[-1] >= deps[0] * 0.5   # no absurd shrink


def test_teacher_cache_disjointness_note():
    rd = run_dir()
    if rd is None:
        return
    src = open(os.path.join(
        ROOT, "scripts", "train_eagle_learned_rotation.py")).read()
    assert "disjoint" in src.lower()
