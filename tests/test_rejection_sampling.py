"""Gate I: stochastic rejection sampling passes analytical toy tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from eagle_spinquant.rejection_sampling import (accept_prob, residual_dist,
                                                spec_step)
from eagle_spinquant.lk_losses import overlap_alpha


def _toy(seed=0, V=6):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(V, generator=g) * 1.5, -1)
    q = torch.softmax(torch.randn(V, generator=g) * 1.5, -1)
    return p, q


def test_acceptance_rate_equals_alpha():
    p, q = _toy()
    alpha = float(torch.minimum(p, q).sum())
    g = torch.Generator().manual_seed(1)
    n, acc = 40000, 0
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=g))
        ok, _c = spec_step(p, q, x, gen=g)
        acc += ok
    assert abs(acc / n - alpha) < 0.01


def test_output_distribution_is_p():
    p, q = _toy(seed=2)
    g = torch.Generator().manual_seed(3)
    n = 60000
    counts = torch.zeros(p.shape[0])
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=g))
        ok, corr = spec_step(p, q, x, gen=g)
        counts[x if ok else corr] += 1
    emp = counts / n
    assert float((emp - p).abs().max()) < 0.012, (emp, p)


def test_identical_distributions_always_accept():
    p, _ = _toy()
    g = torch.Generator().manual_seed(4)
    for _ in range(200):
        x = int(torch.multinomial(p, 1, generator=g))
        ok, _c = spec_step(p, p, x, gen=g)
        assert ok


def test_residual_dist_normalized():
    p, q = _toy(seed=5)
    r = residual_dist(p, q)
    assert abs(float(r.sum()) - 1.0) < 1e-6
    assert (r >= 0).all()


def test_accept_prob_clamped():
    p, q = _toy(seed=6)
    for x in range(p.shape[0]):
        a = accept_prob(p, q, x)
        assert 0.0 <= float(a) <= 1.0
