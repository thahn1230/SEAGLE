"""Gate I tests: KV4 append/readback, branching, rollback, compaction,
tails, context extension, counters, no-silent-fallback (spec §10 B3/§30)."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle.model.kv_cache import KVCache  # noqa: E402
from eagle_spinquant.kv4_cache import (fake_quant_kv, install_kv4_on_past,
                                       KV4Stats, HFIncrementalKV4)  # noqa: E402


def mk_cache(B=1, H=4, L=64, D=128):
    data = torch.zeros(B, H, L, D, dtype=torch.float16)
    return KVCache(data, torch.zeros((), dtype=torch.long))


def test_fake_quant_kv_error_nonzero_and_bounded():
    x = torch.randn(1, 4, 8, 128, dtype=torch.float16) * 3
    q = fake_quant_kv(x, 4)
    err = (q - x).abs()
    assert float(err.max()) > 0, "KV4 must actually change values (Gate C)"
    step = ((x.amax(-1, True) - x.amin(-1, True)) / 15)
    assert bool((err <= step * 0.51 + 1e-3).all()), "error exceeds half-step"
    # 16 levels max per (head, token)
    for h in range(4):
        for t in range(8):
            assert torch.unique(q[0, h, t]).numel() <= 16


def test_append_and_readback():
    k, v = mk_cache(), mk_cache()
    ks, vs = install_kv4_on_past([[k, v]])
    x = torch.randn(1, 4, 5, 128, dtype=torch.float16)
    out = k.cat(x.clone())
    assert out.shape[2] == 5
    assert ks.n_append_calls == 1 and ks.n_tokens_quantized == 5
    # readback returns the QUANTIZED values, not the originals
    assert not torch.equal(out, x)
    assert torch.allclose(out.float(), fake_quant_kv(x, 4).float(),
                          atol=1e-3)
    # second append (multi-token verification block)
    y = torch.randn(1, 4, 3, 128, dtype=torch.float16)
    out2 = k.cat(y.clone())
    assert out2.shape[2] == 8 and ks.n_tokens_quantized == 8
    assert torch.allclose(out2[:, :, :5].float(), out.float(), atol=1e-3)


def test_branch_rejection_rollback():
    k, v = mk_cache(), mk_cache()
    ks, _ = install_kv4_on_past([[k, v]])
    k.cat(torch.randn(1, 4, 6, 128, dtype=torch.float16))
    stored6 = k.data[:, :, :6].clone()
    # speculative tree block appended then fully rejected
    k.cat(torch.randn(1, 4, 26, 128, dtype=torch.float16))
    k.current_length.fill_(6)                       # rollback pointer
    assert torch.equal(k.data[:, :, :6], stored6)   # prefix untouched
    z = torch.randn(1, 4, 2, 128, dtype=torch.float16)
    out = k.cat(z)
    assert out.shape[2] == 8                        # context extends cleanly


def test_accepted_prefix_compaction_no_requant():
    k, v = mk_cache(), mk_cache()
    ks, _ = install_kv4_on_past([[k, v]])
    k.cat(torch.randn(1, 4, 4, 128, dtype=torch.float16))   # prefix
    k.cat(torch.randn(1, 4, 26, 128, dtype=torch.float16))  # tree block
    # accept tree positions 4+idx {2,7}: EAGLE copy(select) semantics
    sel = torch.tensor([6, 11])
    vals = k.data[:, :, sel].clone()
    before_tokens = ks.n_tokens_quantized
    k.copy(sel, prev_length=4)
    assert int(k.current_length) == 6
    assert torch.equal(k.data[:, :, 4:6], vals), "compaction must MOVE " \
        "already-quantized values"
    assert ks.n_tokens_quantized == before_tokens, "no requantization"


def test_non_multiple_group_tail_single_token():
    k, v = mk_cache(), mk_cache()
    ks, _ = install_kv4_on_past([[k, v]])
    k.cat(torch.randn(1, 4, 1, 128, dtype=torch.float16))   # 1-token append
    assert ks.n_tokens_quantized == 1


def test_k_only_and_v_only_switches():
    k, v = mk_cache(), mk_cache()
    ks, vs = install_kv4_on_past([[k, v]], quantize_k=True, quantize_v=False)
    x = torch.randn(1, 4, 3, 128, dtype=torch.float16)
    k.cat(x.clone()); out_v = v.cat(x.clone())
    assert ks.n_tokens_quantized == 3 and vs.n_tokens_quantized == 0
    assert torch.equal(out_v, x), "V must pass through untouched"


def test_hf_incremental_kv4_counters():
    q = HFIncrementalKV4(bits=4)
    past = tuple((torch.randn(1, 4, 7, 128, dtype=torch.float16),
                  torch.randn(1, 4, 7, 128, dtype=torch.float16))
                 for _ in range(2))
    out = q.quantize_new(past, prev_len=4)
    assert q.k_stats.n_tokens_quantized == 6      # 3 new x 2 layers
    assert q.k_stats.nmse > 0 and q.v_stats.nmse > 0
    for (k0, _), (k1, _) in zip(past, out):
        assert torch.equal(k0[:, :, :4], k1[:, :, :4])
        assert not torch.equal(k0[:, :, 4:], k1[:, :, 4:])


def test_no_silent_fallback_counter_gate():
    """A KV4-labeled run must show positive counters; this mirrors the
    runtime assertion used by the evaluation scripts."""
    k, v = mk_cache(), mk_cache()
    ks, vs = install_kv4_on_past([[k, v]])
    with pytest.raises(AssertionError):
        assert ks.n_tokens_quantized > 0, "KV4 requested but never invoked"
