"""Bridge to EAGLE-1: load EaModel, run speculative / baseline generation, and
measure latency + acceptance without editing third_party.

Acceptance is recovered by driving EAGLE's generator APIs (ea_generate /
naive_generate), which yield input_ids once per target forward round. The number
of tokens added in a round equals accept_length+1 (EAGLE) or exactly 1 (vanilla),
so per-round acceptance falls out of the length deltas — no third_party edits.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator

import torch

from . import metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EAGLE_DIR = os.path.join(PROJECT_ROOT, "third_party", "EAGLE")


def add_eagle_to_syspath() -> None:
    if EAGLE_DIR not in sys.path:
        sys.path.insert(0, EAGLE_DIR)


def load_eagle_model(base_model_path: str, ea_model_path: str,
                     dtype=torch.float16, device_map="cuda", low_cpu_mem=True):
    """EaModel.from_pretrained wrapper. base_model_path/ea_model_path may be HF
    repo ids or local dirs (the cache resolves either)."""
    add_eagle_to_syspath()
    from eagle.model.ea_model import EaModel

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem,
        device_map=device_map,
    )
    model.eval()
    return model


def get_tokenizer(model):
    return model.get_tokenizer()


def build_llama2_chat_prompt(tokenizer, user_message: str) -> torch.Tensor:
    """Llama-2-chat prompt formatting (matches EAGLE's llama2chat eval)."""
    sys_prompt = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  "answers to the user's questions.")
    text = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    return tokenizer(text, return_tensors="pt").input_ids


def build_vicuna_prompt(tokenizer, user_message: str) -> torch.Tensor:
    """Vicuna v1.1 conversation template. Matches fastchat's
    get_conversation_template('vicuna') as used by EAGLE's
    eagle/evaluation/gen_ea_answer_vicuna.py (sep_style ADD_COLON_TWO,
    sep=' '): system + ' USER: ' + msg + ' ASSISTANT:'."""
    sys_prompt = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  "answers to the user's questions.")
    text = f"{sys_prompt} USER: {user_message} ASSISTANT:"
    return tokenizer(text, return_tensors="pt").input_ids


PROMPT_BUILDERS = {
    "llama2": build_llama2_chat_prompt,
    "vicuna": build_vicuna_prompt,
}


@torch.no_grad()
def generate_and_measure(model, input_ids: torch.Tensor, mode: str = "eagle",
                         max_new_tokens: int = 256, temperature: float = 0.0,
                         tree_choices=None, max_depth: int = 8,
                         warmup: bool = False) -> dict:
    """Drive EAGLE (mode='eagle') or vanilla (mode='baseline') generation, one
    round at a time, timing with CUDA events and tracking acceptance.

    Returns latency/throughput/acceptance metrics + the generated text length.
    runtime_mode is set by the caller (depends on target quantization)."""
    add_eagle_to_syspath()
    from eagle.model.choices import mc_sim_7b_63
    if tree_choices is None:
        tree_choices = mc_sim_7b_63

    device = model.base_model.lm_head.weight.device
    input_ids = input_ids.to(device)
    input_len = input_ids.shape[1]

    metrics.reset_peak_memory(device)
    tracker = metrics.DepthAcceptanceTracker(max_depth=max_depth)
    rounds = 0
    prev_len = input_len

    if mode == "eagle":
        gen = model.ea_generate(input_ids, temperature=temperature,
                                max_steps=max_new_tokens, tree_choices=tree_choices)
    elif mode == "baseline":
        gen = model.naive_generate(input_ids, temperature=temperature,
                                   max_steps=max_new_tokens)
    else:
        raise ValueError(mode)

    final_ids = input_ids
    with metrics.cuda_timer(device) as t:
        for out_ids in gen:
            cur_len = out_ids.shape[1]
            delta = cur_len - prev_len
            if delta > 0:
                accept_length = delta - 1  # bonus tokens beyond the guaranteed one
                tracker.record(max(accept_length, 0))
                rounds += 1
                prev_len = cur_len
            final_ids = out_ids
            if cur_len - input_len >= max_new_tokens:
                break

    new_tokens = final_ids.shape[1] - input_len
    total_ms = t["ms"]
    acc = metrics.acceptance_stats(new_tokens, rounds)
    result = {
        "mode": mode,
        "new_tokens": new_tokens,
        "forward_rounds": rounds,
        "total_ms": total_ms,
        "ms_per_token": metrics.ms_per_token(new_tokens, total_ms),
        "tokens_per_s": metrics.throughput_tokens_per_s(new_tokens, total_ms),
        "avg_accept_length": acc["avg_accept_length"],
        "peak_mem_gib": metrics.peak_memory_gib(device),
        **tracker.summary(),
    }
    if warmup:
        result["warmup"] = True
    return result


def decode_completion(model, final_ids: torch.Tensor, input_len: int) -> str:
    tok = model.get_tokenizer()
    return tok.decode(final_ids[0, input_len:], skip_special_tokens=True)
