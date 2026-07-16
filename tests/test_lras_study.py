"""Required tests for the learned-rotation AL sensitivity study (spec §16).

CPU-only. Artifact-based tests read the study run directory named in
runs/LRAS_RUN_DIR (written by the launcher); they skip if artifacts are not
yet produced, and the final gate re-runs the full suite when all runs are
complete.
"""
import glob
import hashlib
import json
import os
import sys

import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _run_dir():
    p = os.path.join(ROOT, "runs", "LRAS_RUN_DIR")
    if not os.path.exists(p):
        pytest.skip("LRAS run dir pointer not written yet")
    return open(p).read().strip()


# ---------- 1. exact nine-cell enumeration ----------

def test_precision_matrix_config_complete():
    spec = yaml.safe_load(open(os.path.join(
        ROOT, "configs", "eagle_learned_rotation_precision_matrix.yaml")))
    cells = spec["cells"]
    want = {f"T{t}_D{d}" for t in (16, 8, 4) for d in (16, 8, 4)}
    assert set(cells) == want, set(cells) ^ want
    for cid, c in cells.items():
        assert c["target_bits"] in (16, 8, 4)
        assert c["draft_bits"] in (16, 8, 4)
    assert spec["rotation_kind"] == "learned_chat_w4a4kv16"
    assert spec["target_quant"]["gptq"] is False
    assert spec["target_quant"]["w_clip"] is True
    assert spec["target_quant"]["kv"] == "fp16"
    assert spec["target_quant"]["act"] == "per_token_asymmetric"


# ---------- 2-5. provenance / no fallback / no gptq / clipping ----------

def test_learned_rotation_checkpoint_provenance():
    from eagle_spinquant import experiment, study
    paths = experiment.resolve_paths(experiment.load_config(None))
    rr = experiment.load_config(None).get("paths", {}).get("rotations_root")
    p = study.r_bin_path("learned_chat_w4a4kv16", 0, paths["target_path"], rr)
    assert os.path.isfile(p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man_files = glob.glob(os.path.join(
        ROOT, "runs", "eagle_learned_rotation_al_sensitivity_*",
        "checkpoint_manifest.json"))
    assert man_files, "no checkpoint manifest"
    man = json.load(open(sorted(man_files)[-1]))
    assert man["learned_rotation_sha256"] == h, "rotation artifact changed"
    R = torch.load(p, map_location="cpu", weights_only=False)
    assert R["R1"].shape == (4096, 4096)
    assert len([k for k in R if k.endswith("self_attn.R2")]) == 32


def test_no_random_hadamard_fallback():
    """Requesting a named learned rotation must raise loudly if absent —
    never silently fall back to a random rotation."""
    from eagle_spinquant import experiment, study
    paths = experiment.resolve_paths(experiment.load_config(None))
    with pytest.raises(FileNotFoundError):
        study.r_bin_path("learned_chat_w4a4kv16", 0, paths["target_path"],
                         "/nonexistent/root")


def test_no_gptq_and_kv_fp16_policy():
    from eagle_spinquant import study
    for q in ("w4a4", "w8a8"):
        c = study.QUANT_CFGS[q]
        assert c["k_bits"] == 16 and c["v_bits"] == 16, "KV must stay FP16"
        assert "gptq" not in {k.lower() for k in c}, "GPTQ path forbidden"


def test_weight_clipping_enabled_in_fakequant():
    """The draft-side weight fake-quant must use the SpinQuant clip search
    (RTN+clip), mirroring the official w_clip policy."""
    import inspect
    from eagle_spinquant import fake_w4a4_draft as fq
    src = inspect.getsource(fq._weight_fake_quant)
    assert ("find_params" in src or "clip" in src.lower()), \
        "weight fake-quant lost its clip search"


# ---------- 7. first/recurrent projection dispatch ----------

def test_projection_dispatch_counters():
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveProjection, PostProjectionR1)
    import torch.nn as nn
    D = 8
    lin_f, lin_r = nn.Linear(2 * D, D), nn.Linear(2 * D, D)
    proj = ConcatSelectiveProjection(lin_f, lin_r,
                                     PostProjectionR1(torch.eye(D)))
    z = torch.randn(3, 2 * D)
    proj.select = "first"; proj(z)
    proj.select = "recurrent"; proj(z); proj(z)
    assert proj.calls_first == 1 and proj.calls_recurrent == 2
    proj.select = None
    with pytest.raises(RuntimeError):
        proj(z)


# ---------- 8. embedding scaling FP equivalence (Gate E) ----------

def test_embedding_scale_fp_equivalence():
    torch.manual_seed(0)
    D = 64
    for _ in range(2):  # first / recurrent style weights
        W = torch.randn(D, 2 * D, dtype=torch.float32)
        b = torch.randn(D, dtype=torch.float32)
        e = torch.randn(16, D) * 0.01          # realistic small embedding
        h = torch.randn(16, D) * 3.0
        z = torch.cat([e, h], -1)
        y_ref = z @ W.t() + b
        for alpha in (0.5, 2.0, 37.27, 512.0):
            Wp = W.clone()
            Wp[:, :D] = Wp[:, :D] / alpha
            zp = torch.cat([e * alpha, h], -1)
            y = zp @ Wp.t() + b
            assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4), alpha
        # fp16 path tolerance
        y16 = (torch.cat([e * 37.27, h], -1).half()
               @ (torch.cat([W[:, :D] / 37.27, W[:, D:]], 1)).half().t()
               + b.half())
        assert (y16.float() - y_ref).abs().max() < 0.35  # fp16 GEMM tol


# ---------- 9. separate-scale fake-quant correctness ----------

def test_projection_split_scale_correctness():
    """BranchwiseActLinear must equal the two-term reference
    Q(W)·[deq(Q_e(e)) | deq(Q_h(h))] — NOT integer-concat under one scale."""
    from eagle_spinquant.concat_selective_projection import (
        BranchwiseActLinear)
    from eagle_spinquant import fake_w4a4_draft as fq
    torch.manual_seed(0)
    D = 32
    W = torch.randn(D, 2 * D).half()
    b = torch.randn(D).half()
    lin = torch.nn.Linear(2 * D, D)
    lin.weight.data, lin.bias.data = W, b
    m = BranchwiseActLinear(lin.weight, lin.bias, "t", w_bits=4,
                            e_bits=4, h_bits=4)
    e = (torch.randn(8, D) * 0.01).half()
    h = (torch.randn(8, D) * 3.0).half()
    z = torch.cat([e, h], -1)
    y = m(z)
    # reference: the SAME official per-token asym quantizer applied to each
    # slice INDEPENDENTLY (this verifies slice independence, not the
    # quantizer formula)
    def q(x, bits=4):
        qq = fq._act_quantizer(bits)
        qq.find_params(x)
        return qq(x).float()
    Wq = m.w_fake.float()
    qe, qh = q(e), q(h)
    y_ref = (torch.cat([qe, qh], -1) @ Wq.t() + b.float())
    assert (y.float() - y_ref).abs().max() < 0.05, \
        float((y.float() - y_ref).abs().max())
    # two-term computation is identical by linearity
    y_two = qe @ Wq[:, :D].t() + qh @ Wq[:, D:].t() + b.float()
    assert torch.allclose(y_ref, y_two, atol=1e-4)
    # and it must DIFFER from the shared-scale result on scale-mismatched data
    y_shared = (q(z) @ Wq.t() + b.float())
    assert (y_ref - y_shared).abs().max() > 0.01


# ---------- 10. acceptance metric definition ----------

def test_acceptance_metric_definition():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_eagle_component_grid import run_gen

    class FakeGen:
        def __init__(self):
            # ilen=4; cycle commits: +3, +1, +2  (tau values)
            self.seqs = [torch.zeros(1, 7), torch.zeros(1, 8),
                         torch.zeros(1, 10)]
        def __iter__(self):
            return iter(self.seqs)

    toks, deltas = run_gen(FakeGen(), 4, 6)
    assert deltas == [3, 1, 2], deltas          # tau per verification cycle
    assert sum(deltas) == 6                      # committed tokens
    # AL = mean(tau); accepted draft tokens per cycle = tau - 1
    assert abs(sum(deltas) / len(deltas) - 2.0) < 1e-9


# ---------- 6. isolation artifacts (Gate D records) ----------

def test_draft_target_module_isolation_records():
    rd = _run_dir()
    metas = glob.glob(os.path.join(rd, "shards", "grid__*.json"))
    if not metas:
        pytest.skip("no grid runs yet")
    for m in metas:
        d = json.load(open(m))
        assert d["isolation"]["ok"] is True, m


# ---------- 11. deterministic repeated baseline ----------

def test_deterministic_t16_d16_repeat():
    rd = _run_dir()
    a = os.path.join(rd, "matrix", "shards", "cell__T16_D16.csv")
    b = os.path.join(rd, "matrix_repeat", "shards", "cell__T16_D16.csv")
    if not (os.path.exists(a) and os.path.exists(b)):
        pytest.skip("determinism repeat not run yet")
    ra = open(a).read().splitlines()
    rb = open(b).read().splitlines()
    assert ra == rb, "T16_D16 repeat differs (see report for explanation)"
