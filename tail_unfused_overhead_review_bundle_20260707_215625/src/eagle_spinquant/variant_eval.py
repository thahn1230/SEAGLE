"""Unified evaluation of the integration variants on the rotated/quantized target.

Method names (match docs/05 / the task spec):
  fp16_target_fp16_eagle            : stock EAGLE on the unrotated FP16 target
  spinquant_target_no_eagle         : vanilla decoding on the rotated(+quant) target
  spinquant_target_original_eagle   : NAIVE — rotated target, stock draft fed h_hat
  spinquant_target_unrotate_interface : Variant A (h = unrotate(h_hat))
  spinquant_target_conjugated_draft : Variant B (fc h-block folded)
  spinquant_target_retrained_rotated_draft : Variant C (draft trained on h_hat)

For each built model we run EAGLE speculative + vanilla baseline on the same
prompts and record latency / throughput / acceptance / per-depth alpha / memory,
tagged with the correct runtime_mode (fake-quant vs fp16; never a real-kernel
claim — docs/00 S4).
"""

from __future__ import annotations

import os

import torch

from . import eagle_bridge, metrics, rotated_target

VARIANT_TO_ADAPTER = {
    "naive": "naive",
    "unrotate": "unrotate",
    "conjugate": "conjugate",
}

# method name -> (stage, adapter variant)
METHODS = {
    "spinquant_target_original_eagle": "naive",
    "spinquant_target_unrotate_interface": "unrotate",
    "spinquant_target_conjugated_draft": "conjugate",
}


def runtime_mode_for_stage(stage: str) -> str:
    return metrics.RUNTIME_FP16 if stage == "rotate_only" else metrics.RUNTIME_FAKE_QUANT


def target_quant_tag(stage: str, quant_config: dict | None) -> str:
    if stage == "rotate_only":
        return "rotated_fp16"
    if quant_config is None:
        return "rotated_fp16"
    return f"W{quant_config.get('w_bits')}A{quant_config.get('a_bits')}" \
           f"KV{quant_config.get('k_bits')}"


def build_variant_model(paths: dict, input_model_id: str, r_bin: str,
                        variant: str, stage: str, quant_config: dict | None,
                        dtype=torch.float16, device="cuda"):
    """Build a rotated EAGLE model with the chosen adapter variant."""
    adapter_variant = VARIANT_TO_ADAPTER[variant]
    model, adapter, stash = rotated_target.build_rotated_eagle_model(
        target_path=paths["target_path"], draft_path=paths["draft_path"],
        input_model_id=input_model_id, r_bin_path=r_bin,
        quant_config=quant_config, stage=stage, variant=adapter_variant,
        dtype=dtype, device=device)
    return model, adapter, stash


@torch.no_grad()
def run_prompts(model, tokenizer, prompts, max_new_tokens, temperature,
                method: str, stage: str, quant_config: dict | None,
                include_baseline: bool = True, warmup: bool = True) -> list[dict]:
    rmode = runtime_mode_for_stage(stage)
    qtag = target_quant_tag(stage, quant_config)
    rows = []
    if warmup and prompts:
        wi = eagle_bridge.build_llama2_chat_prompt(tokenizer, prompts[0]["text"])
        eagle_bridge.generate_and_measure(model, wi, mode="eagle",
                                          max_new_tokens=8, warmup=True)
    modes = ["eagle"] + (["baseline"] if include_baseline else [])
    for p in prompts:
        input_ids = eagle_bridge.build_llama2_chat_prompt(tokenizer, p["text"])
        ctx = input_ids.shape[1]
        for mode in modes:
            r = eagle_bridge.generate_and_measure(
                model, input_ids, mode=mode, max_new_tokens=max_new_tokens,
                temperature=temperature)
            method_name = method if mode == "eagle" else "spinquant_target_no_eagle"
            if stage == "rotate_only" and mode == "baseline":
                method_name = "rotated_target_no_eagle"
            r.update({
                "method": method_name, "eagle_mode": mode,
                "question_id": p["question_id"], "category": p["category"],
                "context_len": ctx, "batch_size": 1,
                "runtime_mode": rmode, "target_quant": qtag, "stage": stage,
            })
            rows.append(r)
    return rows


@torch.no_grad()
def measure_unrotation_overhead(D=4096, seq=26, R1=None, gamma_f=None,
                                iters=200, device="cuda") -> dict:
    """Isolate the cost of the unrotation GEMM (Variant A per-draft-step op)."""
    from .rotation_interface import unrotate_hidden
    if R1 is None:
        R1, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    if gamma_f is None:
        gamma_f = torch.rand(D, device=device) + 0.5
    x = torch.randn(1, seq, D, device=device, dtype=torch.float16)
    R1 = R1.to(device); gamma_f = gamma_f.to(device)
    for _ in range(10):
        unrotate_hidden(x.float(), R1.float(), gamma_f.float())
    with metrics.cuda_timer(device) as t:
        for _ in range(iters):
            unrotate_hidden(x.float(), R1.float(), gamma_f.float())
    return {"seq": seq, "D": D, "iters": iters,
            "us_per_call": t["ms"] * 1e3 / iters}
