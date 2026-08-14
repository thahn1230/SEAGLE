"""Unit gates for the quantized-anchor machinery (study section 22)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "third_party", "EAGLE"))
from eagle_spinquant import anchor_quant as aq  # noqa: E402


def _toy_tw(seed=0, shape=(8, 16)):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(*shape, generator=g) * 0.05
            for k in aq.QSITES}


def test_capture_matches_official_and_is_frozen():
    tw = _toy_tw()
    a = aq.capture_anchor(tw)
    for k in aq.QSITES:
        # scales positive, per-out-channel
        assert (a[k]["scale"] > 0).all()
        assert a[k]["scale"].shape[0] == tw[k].shape[0]
        # c0 in the official clamp range
        assert int(a[k]["c0"].min()) >= -(aq.MAXQ + 1)
        assert int(a[k]["c0"].max()) <= aq.MAXQ
        # frozen re-quant of the anchor reproduces c0 exactly
        assert torch.equal(aq.codes(tw[k], a[k]["scale"]), a[k]["c0"])
    # capture is deterministic (scale freeze across "training")
    b = aq.capture_anchor(tw)
    for k in aq.QSITES:
        assert torch.equal(a[k]["scale"], b[k]["scale"])
        assert torch.equal(a[k]["c0"], b[k]["c0"])


def test_frozen_fake_quant_matches_official_same_scale():
    tw = _toy_tw(1)
    a = aq.capture_anchor(tw)
    for k in ("down", "q"):
        w = tw[k] + 0.01 * torch.randn_like(tw[k])
        s = a[k]["scale"]
        got = aq.frozen_fake_quant(w, s)
        want = s * torch.clamp(torch.round(w / s), -(aq.MAXQ + 1), aq.MAXQ)
        assert torch.allclose(got, want.to(got.dtype), atol=0, rtol=0)


def test_frozen_fake_quant_ste_identity_backward():
    tw = _toy_tw(2)
    a = aq.capture_anchor(tw)
    w = tw["down"].clone().requires_grad_(True)
    y = aq.frozen_fake_quant(w, a["down"]["scale"])
    y.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_cell_stability_under_training_like_updates():
    """anchor code c0 maps to a fixed cell throughout training: moving a
    weight WITHIN |u-c0|<0.5 never changes its code under frozen scales."""
    tw = _toy_tw(3)
    a = aq.capture_anchor(tw)
    k = "gate"
    s = a[k]["scale"]
    u0 = tw[k] / s
    inner = tw[k] + (0.49 - (u0 - torch.round(u0)).abs()).clamp(min=0) \
        * 0.9 * s * torch.sign(torch.randn_like(tw[k]))
    # only touch elements whose c0 is strictly inside the clamp range
    safe = (a[k]["c0"] > -(aq.MAXQ + 1)) & (a[k]["c0"] < aq.MAXQ)
    moved = torch.where(safe, inner, tw[k])
    assert torch.equal(aq.codes(moved, s)[safe], a[k]["c0"][safe])


def test_cell_loss_zero_inside_rho_grad_outside():
    tw = {k: torch.zeros(4, 4) for k in aq.QSITES}
    a = {k: dict(scale=torch.ones(4, 1), c0=torch.zeros(4, 4,
                                                        dtype=torch.int8))
         for k in aq.QSITES}
    rho = 0.45
    # inside the safe region: loss == 0 and grad == 0
    w_in = {k: (0.3 * torch.ones(4, 4)).requires_grad_(True)
            for k in aq.QSITES}
    loss_in = aq.cell_loss(w_in, a, rho=rho)
    assert float(loss_in) == 0.0
    loss_in.backward()
    for k in aq.QSITES:
        assert torch.equal(w_in[k].grad, torch.zeros(4, 4))
    # outside rho (u=0.48): positive loss, grad pushes back toward c0
    w_out = {k: (0.48 * torch.ones(4, 4)).requires_grad_(True)
             for k in aq.QSITES}
    loss_out = aq.cell_loss(w_out, a, rho=rho)
    assert float(loss_out) > 0
    loss_out.backward()
    for k in aq.QSITES:
        assert (w_out[k].grad > 0).all()   # d/dw of relu(u-rho)^2, u>0
    # same magnitude just past the boundary on the negative side
    w_neg = {k: (-0.48 * torch.ones(4, 4)).requires_grad_(True)
             for k in aq.QSITES}
    aq.cell_loss(w_neg, a, rho=rho).backward()
    for k in aq.QSITES:
        assert (w_neg[k].grad < 0).all()


def test_save_load_roundtrip(tmp_path):
    tw = _toy_tw(4)
    a = aq.capture_anchor(tw)
    p = str(tmp_path / "anchor.pt")
    aq.save_anchor(a, p)
    b = aq.load_anchor(p)
    for k in aq.QSITES:
        assert torch.equal(a[k]["scale"], b[k]["scale"])
        assert torch.equal(a[k]["c0"], b[k]["c0"])


def test_drift_metrics_basic():
    tw = _toy_tw(5)
    a = aq.capture_anchor(tw)
    m0 = aq.drift_metrics(tw, a, tw0=tw)
    assert m0["H_Q"] == 0.0 and m0["D_Q"] == 0.0 and m0["D_FP"] == 0.0
    # flip exactly one code far away
    tw2 = {k: v.clone() for k, v in tw.items()}
    s = a["down"]["scale"]
    tw2["down"][0, 0] = (a["down"]["c0"][0, 0].float() + 3.0) * s[0, 0]
    m1 = aq.drift_metrics(tw2, a, tw0=tw)
    assert m1["n_flipped"] == 1
    assert m1["D_Q"] > 0 and m1["D_FP"] > 0
