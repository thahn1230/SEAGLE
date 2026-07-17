"""Fake-quant KV4 for the EAGLE runtime KV cache (and HF incremental eval).

Policy (documented for the B3 audit):
  - K and V quantized independently at APPEND time (quantize->dequantize,
    storage stays fp16 = fake quant; no real packed kernels, no latency
    claims).
  - Granularity: per head, per token, group size 128 over head_dim
    (head_dim=128 -> exactly one group per head/token, matching the
    official SpinQuant k/v_groupsize=128 asymmetric policy).
  - Asymmetric 4-bit (qmax=15).
  - K is quantized post-RoPE (the cache stores post-RoPE keys). No R3
    online Hadamard is applied: the validated learned chat rotation is a
    KV16-class artifact (R1+R2 only), so this KV4 is "KV4 without R3" and
    is labeled as such everywhere.
  - Accepted-prefix compaction (KVCache.copy) moves already-quantized
    values WITHOUT requantization — the same values a real packed cache
    would move. Branch rejection / rollback = length pointer only.
  - Non-multiple-of-group tails: with group == head_dim there are no
    partial groups along the quantized axis; per-token quantization makes
    sequence-length tails trivially exact.

Counters (Gate C / no-silent-fallback):
  n_append_calls, n_tokens_quantized, sum_sqerr/sum_sqref (NMSE) per cache.
"""
from __future__ import annotations

import torch


QMAX = 15  # 4-bit asymmetric


@torch.no_grad()
def fake_quant_kv(x: torch.Tensor, bits: int = 4):
    """x: (B, n_heads, T, head_dim). Per (head, token) asymmetric fake quant
    over head_dim (group 128 == head_dim)."""
    qmax = 2 ** bits - 1
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn).clamp(min=1e-6) / qmax).to(x.dtype)
    q = ((x - mn) / scale).round_().clamp_(0, qmax)
    return q * scale + mn


class KV4Stats:
    __slots__ = ("n_append_calls", "n_tokens_quantized", "sum_sqerr",
                 "sum_sqref")

    def __init__(self):
        self.n_append_calls = 0
        self.n_tokens_quantized = 0
        self.sum_sqerr = 0.0
        self.sum_sqref = 0.0

    @property
    def nmse(self):
        return self.sum_sqerr / max(self.sum_sqref, 1e-12)


def install_kv4_on_past(past_key_values, bits: int = 4,
                        quantize_k: bool = True, quantize_v: bool = True):
    """Wrap every EAGLE KVCache in `past_key_values` (list of [K, V] pairs)
    so that .cat() fake-quantizes incoming blocks before storage.
    Returns (k_stats, v_stats)."""
    k_stats, v_stats = KV4Stats(), KV4Stats()

    def make_cat(cache, stats, enabled):
        orig_cat = cache.cat

        def cat(tensor, dim=2):
            if enabled and tensor.shape[dim] > 0:
                q = fake_quant_kv(tensor, bits)
                stats.n_append_calls += 1
                stats.n_tokens_quantized += int(tensor.shape[dim])
                stats.sum_sqerr += float(((q - tensor).float() ** 2).sum())
                stats.sum_sqref += float((tensor.float() ** 2).sum())
                tensor = q
            return orig_cat(tensor, dim=dim)

        return cat

    for pair in past_key_values:
        k_cache, v_cache = pair[0], pair[1]
        k_cache.cat = make_cat(k_cache, k_stats, quantize_k)
        v_cache.cat = make_cat(v_cache, v_stats, quantize_v)
    return k_stats, v_stats


class HFIncrementalKV4:
    """past_key_values fake-quant for stock HF incremental evaluation:
    quantizes the NEWLY APPENDED slice of each layer's K/V after every
    forward step (legacy tuple cache format)."""

    def __init__(self, bits=4, quantize_k=True, quantize_v=True):
        self.bits = bits
        self.qk, self.qv = quantize_k, quantize_v
        self.k_stats, self.v_stats = KV4Stats(), KV4Stats()

    @torch.no_grad()
    def quantize_new(self, past, prev_len: int):
        out = []
        for (k, v) in past:
            if self.qk and k.shape[2] > prev_len:
                new = k[:, :, prev_len:, :]
                q = fake_quant_kv(new, self.bits)
                self.k_stats.n_append_calls += 1
                self.k_stats.n_tokens_quantized += int(new.shape[2])
                self.k_stats.sum_sqerr += float(((q - new).float() ** 2).sum())
                self.k_stats.sum_sqref += float((new.float() ** 2).sum())
                k = torch.cat([k[:, :, :prev_len, :], q], dim=2)
            if self.qv and v.shape[2] > prev_len:
                new = v[:, :, prev_len:, :]
                q = fake_quant_kv(new, self.bits)
                self.v_stats.n_append_calls += 1
                self.v_stats.n_tokens_quantized += int(new.shape[2])
                self.v_stats.sum_sqerr += float(((q - new).float() ** 2).sum())
                self.v_stats.sum_sqref += float((new.float() ** 2).sum())
                v = torch.cat([v[:, :, :prev_len, :], q], dim=2)
            out.append((k, v))
        return tuple(out)


class DraftKV4Patch:
    """Draft-side KV4: wraps ea_layer.forward (per-instance) so every
    use_cache call returns past_key_values whose NEWLY APPENDED slices are
    fake-quantized in place (append-time semantics; stable_kv then carries
    quantized values across verification cycles)."""

    def __init__(self, ea_layer, bits=4):
        self.ea_layer = ea_layer
        self.bits = bits
        self.k_stats, self.v_stats = KV4Stats(), KV4Stats()
        self._orig = None

    def install(self):
        ea = self.ea_layer
        self._orig = ea.forward
        patch = self

        def fwd(*a, **k):
            past_in = k.get("past_key_values")
            prev = past_in[0][0].shape[2] if past_in else 0
            out = patch._orig(*a, **k)
            if k.get("use_cache") and isinstance(out, tuple) \
                    and len(out) == 2:
                h, past = out
                past = patch._quant_new(past, prev)
                return h, past
            return out

        ea.forward = fwd
        return self

    def _quant_new(self, past, prev):
        out = []
        for layer in past:
            k, v = layer[0], layer[1]
            if k.shape[2] > prev:
                nk = k[:, :, prev:, :]
                q = fake_quant_kv(nk, self.bits)
                self.k_stats.n_append_calls += 1
                self.k_stats.n_tokens_quantized += int(nk.shape[2])
                self.k_stats.sum_sqerr += float(((q - nk).float() ** 2).sum())
                self.k_stats.sum_sqref += float((nk.float() ** 2).sum())
                k[:, :, prev:, :] = q
            if v.shape[2] > prev:
                nv = v[:, :, prev:, :]
                q = fake_quant_kv(nv, self.bits)
                self.v_stats.n_append_calls += 1
                self.v_stats.n_tokens_quantized += int(nv.shape[2])
                self.v_stats.sum_sqerr += float(((q - nv).float() ** 2).sum())
                self.v_stats.sum_sqref += float((nv.float() ** 2).sum())
                v[:, :, prev:, :] = q
            out.append((k, v))
        return tuple(out)

    def uninstall(self):
        if self._orig is not None:
            self.ea_layer.forward = self._orig
            self._orig = None
