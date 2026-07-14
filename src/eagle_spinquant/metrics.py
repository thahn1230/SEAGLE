"""Measurement primitives: timing, GPU memory, throughput, acceptance, PPL.

IMPORTANT (see docs/00 audit S4): SpinQuant's quantization path is fake quant
(quantize-dequantize with FP16 matmuls; no INT4 kernels). Absolute tokens/sec is
therefore NOT a deployment number. Every timing row must carry a `runtime_mode`
tag so the report never conflates fake-quant timing with real low-bit speedup.
"""

from __future__ import annotations

import contextlib
import time
from typing import Iterator

import torch

# runtime_mode vocabulary used across all scripts
RUNTIME_REAL_KERNEL = "real_low_bit_cuda_kernel"
RUNTIME_FAKE_QUANT = "fake_quant_pytorch"          # QDQ modules, FP16 matmul
RUNTIME_DEQUANT_FP16 = "dequantized_fp16"          # weights dequantized, run FP16
RUNTIME_EXECUTORCH = "executorch_export_only"      # off-CUDA export path
RUNTIME_FP16 = "fp16_bf16_baseline"
RUNTIME_UNKNOWN = "unknown"


@contextlib.contextmanager
def cuda_timer(device: int | None = None) -> Iterator[dict]:
    """Context manager returning {'ms': float} using CUDA events (falls back to
    wall clock if CUDA is unavailable). Synchronizes at both ends."""
    result: dict = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield result
        finally:
            end.record()
            torch.cuda.synchronize(device)
            result["ms"] = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        try:
            yield result
        finally:
            result["ms"] = (time.perf_counter() - t0) * 1e3


def reset_peak_memory(device: int | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gib(device: int | None = None) -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 2**30
    return None


def throughput_tokens_per_s(num_tokens: int, ms: float) -> float | None:
    if ms and ms > 0:
        return num_tokens / (ms / 1e3)
    return None


def ms_per_token(num_tokens: int, ms: float) -> float | None:
    if num_tokens and num_tokens > 0:
        return ms / num_tokens
    return None


# ---------------------------------------------------------------------------
# Speculative-decoding acceptance
# ---------------------------------------------------------------------------

def acceptance_stats(new_tokens_total: int, forward_rounds: int) -> dict:
    """EAGLE bookkeeping: each decoding round accepts `accept_length + 1` tokens
    in one target forward. Average acceptance length per round = total generated
    tokens / number of target forward rounds.

    A vanilla autoregressive decoder accepts exactly 1 token per forward, so
    avg_accept_len == 1.0 there and the ratio to it is the token-level speedup
    factor (independent of kernel speed)."""
    avg = new_tokens_total / forward_rounds if forward_rounds else None
    return {
        "new_tokens_total": new_tokens_total,
        "forward_rounds": forward_rounds,
        "avg_accept_length": avg,
    }


class DepthAcceptanceTracker:
    """Accumulates accepted-length histogram to derive per-depth acceptance rate.

    Record `accept_length` (number of *bonus* tokens accepted, i.e. tree depth
    reached) for each decoding round. alpha[d] = P(at least d bonus tokens
    accepted) = fraction of rounds whose accept_length >= d."""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.round_count = 0
        self.ge_counts = [0] * (max_depth + 1)  # ge_counts[d] = #rounds accept_length>=d

    def record(self, accept_length: int) -> None:
        self.round_count += 1
        for d in range(self.max_depth + 1):
            if accept_length >= d:
                self.ge_counts[d] += 1

    def alpha_by_depth(self) -> list[float | None]:
        if self.round_count == 0:
            return [None] * (self.max_depth + 1)
        # alpha[d] for d>=1: conditional accept rate at depth d given depth d-1 reached
        alpha = [None]
        for d in range(1, self.max_depth + 1):
            denom = self.ge_counts[d - 1]
            alpha.append(self.ge_counts[d] / denom if denom else None)
        return alpha

    def summary(self) -> dict:
        return {
            "rounds": self.round_count,
            "ge_counts": self.ge_counts,
            "alpha_by_depth": self.alpha_by_depth(),
        }


# ---------------------------------------------------------------------------
# Perplexity (wikitext-2), self-contained so it works on any HF-style model
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl_wikitext2(model, tokenizer, seqlen: int = 2048, device: str = "cuda",
                       limit_tokens: int | None = None) -> dict:
    """Standard sliding-window (non-overlapping) wikitext-2 test PPL.

    `model` must be callable as model(input_ids).logits or return logits as the
    first element. `limit_tokens` truncates the corpus for smoke tests."""
    from datasets import load_dataset

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = enc.input_ids
    if limit_tokens is not None:
        input_ids = input_ids[:, :limit_tokens]
    n_tokens = input_ids.numel()
    n_chunks = n_tokens // seqlen
    if n_chunks == 0:
        raise ValueError(f"corpus too short ({n_tokens} toks) for seqlen {seqlen}")

    nlls = []
    total = 0
    for i in range(n_chunks):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        out = model(batch)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        total += shift_labels.numel()
    ppl = float(torch.exp(torch.tensor(sum(nlls) / total)))
    return {"ppl": ppl, "n_chunks": n_chunks, "seqlen": seqlen, "eval_tokens": total}
