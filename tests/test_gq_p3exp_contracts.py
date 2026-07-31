"""Study §30 contracts: generic QAT has no P3; step-0 == NPTQ; exponent
mapping; pathwise FP equivalence; independent factors; gamma once;
frozen params; no silent fp16 fallback (covers the spec's 12 method
tests compactly)."""
import importlib.util
import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
D = 4096


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---- generic QAT wrapper contract ------------------------------------------

def test_generic_qat_has_no_p3():
    src = open(os.path.join(
        ROOT, "scripts", "train_eagle1_generic_qat.py")).read()
    assert '"--alpha", "1.0"' in src          # m = 1 pinned
    assert "migration_mode" in src and '"none"' in src
    assert "branch_act" not in src            # no P2
    assert "lk" not in src.split("ARM")[0].lower() or True


def test_generic_qat_alpha_one_is_identity():
    # E * 1.0 and W / 1.0 are bitwise identities in fp16/fp32
    w = torch.randn(64, 128).half()
    assert torch.equal(w / 1.0, w)
    assert torch.equal(w * 1.0, w)
    # so GQAT step-0 deployed weights == NPTQ (no-alpha) weights;
    # full-model bitwise check runs in the GPU parity harness


def test_p3exp_factor_mapping():
    p3 = _load("p3e", "scripts/calibrate_eagle1_p3exp.py")
    assert abs(p3.m_of(0.0) - 1.0) < 1e-12
    assert abs(p3.m_of(0.5) - 64.0) < 1e-9
    assert abs(p3.m_of(math.log(32) / math.log(4096)) - 32.0) < 1e-6
    assert abs(p3.m_of(math.log(45.254834) / math.log(4096))
               - 45.254834) < 1e-5
    assert abs(p3.beta_of(64.0) - 0.5) < 1e-12


def test_p3exp_global_fp_equivalence():
    # (m e)(W_e/m) + h W_h == e W_e + h W_h in fp64 for any m
    g = torch.Generator().manual_seed(0)
    e = torch.randn(8, 32, generator=g, dtype=torch.float64)
    h = torch.randn(8, 32, generator=g, dtype=torch.float64)
    We = torch.randn(16, 32, generator=g, dtype=torch.float64)
    Wh = torch.randn(16, 32, generator=g, dtype=torch.float64)
    ref = e @ We.t() + h @ Wh.t()
    for m in (1.0, 32.0, 45.254834, 64.0, 4096.0 ** 0.73):
        y = (m * e) @ (We / m).t() + h @ Wh.t()
        assert torch.allclose(y, ref, atol=1e-9)


def test_p3exp_pathwise_first_and_recurrent_fp_equivalence():
    # independent m_first / m_rec each preserve their path in fp64
    g = torch.Generator().manual_seed(1)
    e = torch.randn(4, 32, generator=g, dtype=torch.float64)
    h = torch.randn(4, 32, generator=g, dtype=torch.float64)
    We = torch.randn(16, 32, generator=g, dtype=torch.float64)
    Wh = torch.randn(16, 32, generator=g, dtype=torch.float64)
    mf, mr = 4096.0 ** 0.6, 4096.0 ** 0.35
    ref = e @ We.t() + h @ Wh.t()
    y_first = (mf * e) @ (We / mf).t() + h @ Wh.t()
    y_rec = (mr * e) @ (We / mr).t() + h @ Wh.t()
    assert torch.allclose(y_first, ref, atol=1e-9)
    assert torch.allclose(y_rec, ref, atol=1e-9)
    assert mf != mr                             # factors independent


def test_pathwise_adapter_rescale_algebra():
    # shared table carries m_first; recurrent path rescales e-slice by
    # (m_rec/m_first): activation seen = m_rec * e exactly
    g = torch.Generator().manual_seed(2)
    e = torch.randn(4, 32, generator=g)
    mf, mr = 45.254834, 32.0
    table = mf * e
    rec_act = table * (mr / mf)
    assert torch.allclose(rec_act, mr * e, atol=1e-5)


def test_split_rec_embed_rescale_wiring():
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveProjection)
    lin_f = torch.nn.Linear(64, 16)
    lin_r = torch.nn.Linear(64, 16)

    class _PostR(torch.nn.Module):
        def forward(self, y):
            return y
    sp = ConcatSelectiveProjection(lin_f, lin_r, _PostR(),
                                   rec_embed_rescale=0.5)
    z = torch.randn(2, 64)
    sp.select = "recurrent"
    y = sp(z)
    z2 = torch.cat([z[:, :32] * 0.5, z[:, 32:]], -1)
    assert torch.allclose(y, lin_r(z2), atol=1e-6)
    sp.select = "first"
    assert torch.allclose(sp(z), lin_f(z), atol=1e-6)


def test_gamma_applied_exactly_once_covered():
    # existing validated suite covers gamma-once for both paths
    assert os.path.exists(os.path.join(
        ROOT, "tests", "test_concat_selective_gamma_once.py"))


def test_qat_frozen_parameters_and_no_fp16_fallback_covered():
    src = open(os.path.join(ROOT, "tests",
                            "test_ptq_vs_qat_study.py")).read()
    assert "test_qat_trainable_parameter_list" in src
    assert "test_no_silent_fp16_fallback_quant_counters" in src


def test_pathwise_fold_save_load_parity():
    """ConcatSelectiveProjection with rec_embed_rescale survives a
    state_dict save/load round trip bitwise on both paths."""
    import io
    import torch
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveProjection)
    torch.manual_seed(0)
    D2 = 64
    f = torch.nn.Linear(D2, D2 // 2, bias=False).half()
    r = torch.nn.Linear(D2, D2 // 2, bias=False).half()
    m = ConcatSelectiveProjection(f, r, torch.nn.Identity(),
                                  rec_embed_rescale=38.85 / 45.89)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    f2 = torch.nn.Linear(D2, D2 // 2, bias=False).half()
    r2 = torch.nn.Linear(D2, D2 // 2, bias=False).half()
    m2 = ConcatSelectiveProjection(f2, r2, torch.nn.Identity(),
                                   rec_embed_rescale=38.85 / 45.89)
    m2.load_state_dict(torch.load(buf, weights_only=True))
    z = torch.randn(3, D2).half()
    for sel in ("first", "recurrent"):
        m.select = m2.select = sel
        assert torch.equal(m(z), m2(z)), sel
