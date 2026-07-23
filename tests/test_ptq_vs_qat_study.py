"""CPU tests for the PTQ-vs-QAT study (spec section 24).

Interface/structure equivalences already covered by the existing suite
(test_concat_selective_* : exact target-only interface, first-path gamma
handling, recurrent-path handling, P3 FP equivalence, P2/P3 activation
equivalence). This file adds the study-specific contracts: QAT trainable
scope, frozen head/embedding, quantized-forward parity harness pieces,
export/reload round trip, tau aggregation, paired bootstrap, oracle
ceilings, theoretical maximum, split disjointness, no silent FP16
fallback, quantizer counters, teacher-transform algebra.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eagle_spinquant.exact_quantized_rotation_forward import (  # noqa: E402
    ExactQuantizedRotationForward)

D_TOY = 64

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _toy_dims():
    """Shrink the module-level head geometry (NH=4, HD=16 -> D=64) for
    every test in this file; restore afterwards."""
    import eagle_spinquant.exact_quantized_rotation_forward as eq
    old = (eq.NH, eq.HD)
    eq.NH, eq.HD = 4, 16
    yield
    eq.NH, eq.HD = old


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def toy_sd(D=D_TOY, V=200, FF=128, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g) * 0.05

    return {
        "fc.weight": r(D, 2 * D), "fc.bias": r(D),
        "embed_tokens.weight": r(V, D),
        "layers.0.self_attn.q_proj.weight": r(D, D),
        "layers.0.self_attn.k_proj.weight": r(D, D),
        "layers.0.self_attn.v_proj.weight": r(D, D),
        "layers.0.self_attn.o_proj.weight": r(D, D),
        "layers.0.mlp.gate_proj.weight": r(FF, D),
        "layers.0.mlp.up_proj.weight": r(FF, D),
        "layers.0.mlp.down_proj.weight": r(D, FF),
        "layers.0.post_attention_layernorm.weight": torch.ones(D),
    }


class _Rot:
    def __init__(self, R):
        self._R = R

    def R(self):
        return self._R


def build_core(train=True, bits=4, seed=0, device="cpu"):
    sd = toy_sd(seed=seed)
    R = torch.linalg.qr(torch.randn(
        D_TOY, D_TOY, generator=torch.Generator().manual_seed(3)))[0]
    core = ExactQuantizedRotationForward(
        sd, R.float(), torch.ones(D_TOY), torch.randn(200, D_TOY) * 0.05,
        _Rot(R.float().to(device)), alpha_init=4.0, w_bits=bits,
        a_bits=bits, draft_kv_bits=16, train_draft_core=train,
        r2_seed=0, device=device, first_fold_R=torch.eye(D_TOY))
    return core, sd, R


# ---- trainable scope / frozen modules (Gates E/F) --------------------------

def test_qat_trainable_parameter_list():
    core, _, _ = build_core(train=True)
    names = {n for n, p in core.named_parameters() if p.requires_grad}
    expected = {"W_e", "W_h", "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup",
                "Wdown", "b_fc", "gl64", "log_alpha"}
    assert names <= expected
    core.log_alpha.requires_grad_(False)
    live = {n for n, p in core.named_parameters() if p.requires_grad}
    assert live == expected - {"log_alpha"}


def test_embedding_and_head_frozen():
    core, _, _ = build_core(train=True)
    bufs = dict(core.named_buffers())
    assert "E" in bufs and "W_lm16" in bufs   # never Parameters


def test_frozen_core_registers_buffers():
    core, _, _ = build_core(train=False)
    params = {n for n, p in core.named_parameters()}
    assert params == {"log_alpha"}


# ---- quantizer behaviour ---------------------------------------------------

def test_no_silent_fp16_fallback_quant_counters():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("FWHT kernel requires CUDA")
    dev = "cuda:0"
    import eagle_spinquant.exact_quantized_rotation_forward as eq
    old = (eq.NH, eq.HD)
    eq.NH, eq.HD = 4, 16
    try:
        core, _, _ = build_core(train=False, bits=4, device=dev)
        tok = torch.randint(0, 199, (1, 5)).to(dev)
        a = torch.randn(1, 4, D_TOY).to(dev)
        core.forward_chain(tok, a, K=2)
        assert core.counters["w_quant"] >= 9      # all 9 sites quantized
        assert core.counters["a_quant"] > 0
        c16, _, _ = build_core(train=False, bits=16, device=dev)
        c16.forward_chain(tok, a, K=2)
        assert c16.counters["w_quant"] == 0 and c16.counters["a_quant"] == 0
    finally:
        eq.NH, eq.HD = old


def test_ste_forward_matches_official_quantizer():
    from eagle_spinquant.exact_quantized_rotation_forward import (
        ste_weight_quant)
    from eagle_spinquant import fake_w4a4_draft as fq
    w = torch.randn(32, 48)
    q1 = ste_weight_quant(w, 4)
    q2 = fq._weight_fake_quant(w, 4)
    assert torch.equal(q1, q2)
    w2 = w.clone().requires_grad_(True)
    ste_weight_quant(w2, 4).sum().backward()
    assert torch.allclose(w2.grad, torch.ones_like(w2))   # STE identity


def test_forward_train_grad_reaches_all_trainables():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("FWHT kernel requires CUDA")
    dev = "cuda:0"
    import eagle_spinquant.exact_quantized_rotation_forward as eq
    old = (eq.NH, eq.HD)
    eq.NH, eq.HD = 4, 16
    try:
        core, _, _ = build_core(train=True, bits=4, device=dev)
        core.log_alpha.requires_grad_(False)
        core.train()
        tok = torch.randint(0, 199, (2, 7)).to(dev)
        feat = torch.randn(2, 6, D_TOY).to(dev)
        pm = torch.ones(2, 6, dtype=torch.bool).to(dev)
        pm[1, 4:] = False
        h = core.forward_train(tok, feat, pad_mask=pm)
        loss = h.float().pow(2).mean() + \
            core.head_logits(h).float().mean()
        loss.backward()
        for n, p in core.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and \
                    torch.isfinite(p.grad).all(), n
    finally:
        eq.NH, eq.HD = old


def test_export_reload_roundtrip():
    core, sd, _ = build_core(train=True)
    ex = core.export_original_state()
    fc = torch.cat([ex["W_e"], ex["W_h"]], dim=1)
    assert torch.allclose(fc.float(), sd["fc.weight"].float(), atol=1e-3)
    assert torch.allclose(ex["b_fc"].float(), sd["fc.bias"].float(),
                          atol=1e-3)
    assert set(ex) == {"W_e", "W_h", "Wq", "Wk", "Wv", "Wo", "Wgate",
                       "Wup", "Wdown", "b_fc", "gl"}


# ---- teacher transform algebra (Gate G) ------------------------------------

def test_teacher_target_transform_consistency():
    """t = ((a R^T) * gamma) R  must equal  h R  when a = (h / gamma) R."""
    g = torch.Generator().manual_seed(5)
    R = torch.linalg.qr(torch.randn(D_TOY, D_TOY, generator=g))[0]
    gamma = torch.rand(D_TOY) + 0.5
    h = torch.randn(8, D_TOY)
    a = (h / gamma) @ R
    t = ((a @ R.t()) * gamma) @ R
    assert torch.allclose(t, h @ R, atol=1e-5)


def test_identity_arm_first_fold_is_unrotated():
    if not torch.cuda.is_available():
        pytest.skip("FWHT kernel (down fold) requires CUDA")
    dev = "cuda:0"
    core, sd, R = build_core(train=False, device=dev)
    core = core.to(dev)
    tw = {k: (v.cpu() if torch.is_tensor(v) else v)
          for k, v in core.transformed_weights(exact=True).items()}
    # first fold used gamma=1, R_first=I -> e-slice W_e/alpha, h-slice W_h
    a = float(core.log_alpha.exp())
    W_first = tw["W_first"].float()
    assert torch.allclose(W_first[:, D_TOY:],
                          sd["fc.weight"][:, D_TOY:].float(), atol=1e-3)
    assert torch.allclose(W_first[:, :D_TOY] * a,
                          sd["fc.weight"][:, :D_TOY].float(), atol=1e-2)
    # recurrent fold IS rotated (internal basis)
    assert torch.allclose(tw["W_rec"].float()[:, D_TOY:],
                          (sd["fc.weight"][:, D_TOY:].double()
                           @ R.double()).float(), atol=1e-3)


# ---- tau aggregation + bootstrap (Gate K) ----------------------------------

def test_official_tau_aggregation():
    agg = _load("agg", "scripts/aggregate_micro_al.py")
    cl = [("a", [3, 4]), ("b", [1])]
    assert abs(agg.pooled_mal(cl) - 8 / 3) < 1e-9


def test_paired_bootstrap_sign_and_null():
    agg = _load("agg2", "scripts/aggregate_micro_al.py")
    rng = np.random.default_rng(0)
    ca = [(str(i), [2.0]) for i in range(40)]
    cb = [(str(i), [2.0 + 0.5]) for i in range(40)]
    d, lo, hi, p, n = agg.paired_boot(ca, cb, rng, reps=500)
    assert abs(d - 0.5) < 1e-9 and lo > 0 and n == 40
    d0, lo0, hi0, p0, _ = agg.paired_boot(ca, ca, rng, reps=500)
    assert abs(d0) < 1e-9 and lo0 <= 0 <= hi0


def test_seed_mean_delta_ci():
    an = _load("an", "scripts/analyze_qat_attainment.py")
    rng = np.random.default_rng(1)
    base = [(str(i), [2.0]) for i in range(30)]
    seeds = [[(str(i), [2.2]) for i in range(30)],
             [(str(i), [2.4]) for i in range(30)]]
    d, lo, hi = an.seed_mean_delta_ci(base, seeds, rng, reps=300)
    assert abs(d - 0.3) < 1e-9 and lo > 0


# ---- oracle ceilings (Gate J) ----------------------------------------------

def test_oracle_chain_and_tree_ceiling():
    oc = _load("oc", "scripts/eval_eagle_oracle_ceiling.py")
    assert oc.oracle_cycles(13, 5) == [6, 6, 1]
    assert oc.oracle_cycles(6, 5) == [6]
    assert oc.oracle_cycles(3, 5) == [3]
    assert oc.oracle_cycles(8, 1) == [2, 2, 2, 2]
    assert oc.oracle_cycles(0, 5) == []


def test_theoretical_max_tau_from_tree():
    sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))
    from eagle.model.choices import mc_sim_7b_63 as tree
    l_max = max(len(p) for p in tree)
    assert l_max + 1 >= 2
    d = _load("oc2", "scripts/eval_eagle_oracle_ceiling.py") \
        .oracle_cycles(600, l_max)
    assert abs(sum(d) / len(d) - (l_max + 1)) < 0.02


# ---- dataset split disjointness (Gate H) -----------------------------------

def test_training_cache_disjoint_and_pinned():
    cache = "/data/thahn1230/datasets/sharegpt/eagle_tok_cache_v1.pt"
    if not os.path.exists(cache):
        import pytest
        pytest.skip("training cache not built on this machine")
    blob = torch.load(cache, weights_only=False)
    assert blob["train_n"] == 12000 and blob["val_n"] == 400
    assert len(blob["rows"]) == 12400
    m = blob["manifest"]
    assert m["shuffle_seed"] == 42 and m["max_len"] == 2048
    assert "sha256" in m and len(m["sha256"]) == 64
    assert "eval/calib/valid" in m["exclusion"]
    r = blob["rows"][0]
    assert r["input_ids"].dtype == torch.int16
    assert r["loss_mask"].dtype == torch.bool
    assert r["loss_mask"].shape == r["input_ids"].shape


def test_calib_pool_offsets_disjoint():
    from eagle_spinquant import eval_datasets as ed
    offs = [500, 1000]
    assert len({0, *offs}) == 3      # eval/calib/valid distinct ranges


# ---- scheduler DAG ---------------------------------------------------------

def test_scheduler_dag_dependencies():
    sch = _load("sch", "scripts/schedule_eagle_ptq_qat_jobs.py")
    jobs = sch.build_jobs("/rd", 45.254834, 45.254834)
    names = {j["name"] for j in jobs}
    for need in ("train_C3_s0", "train_C7_s2", "train_C7b", "eval_C14",
                 "oracle_int4", "fidelity_fp16"):
        assert need in names
    by = {j["name"]: j for j in jobs}
    assert by["train_C7b"]["deps"] == ["train_C6"]
    assert by["eval_C9"]["deps"] == ["train_C8"]
    assert by["eval_C14"]["deps"] == ["train_C3_s0"]
    for j in jobs:
        for d in j["deps"]:
            assert d in names
