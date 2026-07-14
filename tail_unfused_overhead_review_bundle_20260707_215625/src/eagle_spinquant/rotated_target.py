"""Build an EAGLE EaModel whose TARGET is SpinQuant-rotated (+optionally
quantized), with the draft interface adapted (Variant A/B/naive). Task I-1.

Pipeline (docs/01 S4):
  1. load EaModel (base on GPU, draft loaded + on GPU)
  2. stash gamma_f + original lm_head/embed, fuse norms, rotate (load R.bin),
     optionally add ActQuantWrapper + weight/act/KV quant  (spinquant_bridge)
  3. move everything back to CUDA (rotate_model leaves weights on CPU)
  4. install the draft interface adapter (unrotate/conjugate/naive) with an
     ORIGINAL-basis head for the draft

The target's rotated lm_head yields correct verification logits from h_hat
directly; only the draft needs the original basis (see docs/01 S2).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from . import eagle_bridge, spinquant_bridge as sb
from .draft_conjugation import conjugate_draft_fc
from .rotation_interface import DraftInterfaceAdapter


def build_rotated_eagle_model(
    target_path: str, draft_path: str, input_model_id: str,
    r_bin_path: str, quant_config: dict | None = None,
    stage: str = "rotate_only", variant: str = "unrotate",
    dtype=torch.float16, device="cuda",
):
    """Returns (ea_model, adapter, stash).

    stage: 'rotate_only' (FP16 rotated, R1/R2/R4) or 'full' (adds W/A/KV quant).
    variant: 'naive' | 'unrotate' (A) | 'conjugate' (B).
    r_bin_path: path to R.bin (learned or random) — required so we know R1.
    """
    model = eagle_bridge.load_eagle_model(
        base_model_path=target_path, ea_model_path=draft_path,
        dtype=dtype, device_map=device)

    if quant_config is None:
        spec = sb.default_ptq_args(rotate=True, optimized_rotation_path=r_bin_path)
    else:
        spec = sb.quant_config_to_args(
            quant_config, rotate=True, optimized_rotation_path=r_bin_path)

    stash = sb.apply_spinquant_pipeline(
        model.base_model, spec, input_model_id=input_model_id, stage=stage)

    # rotate_model left rotated weights on CPU; bring the whole model back to GPU.
    model.base_model.to(device)
    model.ea_layer.to(device)

    R1 = stash.get("R1")
    if R1 is None:
        raise RuntimeError("R1 not available; pass an R.bin via r_bin_path")
    gamma_f = stash["gamma_f"]
    orig_head_w = stash["lm_head_weight"]

    if variant == "conjugate":
        conjugate_draft_fc(model.ea_layer, R1.to(device), gamma_f.to(device))

    adapter = DraftInterfaceAdapter(
        model, R1=R1, gamma_f=gamma_f, original_head_weight=orig_head_w,
        variant=variant).install()

    return model, adapter, stash


@torch.no_grad()
def build_rotated_base_only(target_path: str, input_model_id: str, r_bin_path: str,
                           quant_config: dict | None = None,
                           stage: str = "rotate_only", dtype=torch.float16,
                           device="cuda"):
    """Rotated target WITHOUT the EAGLE draft — for PPL eval of our ported model
    (task X-1). Returns (hf_model, tokenizer, stash)."""
    eagle_bridge.add_eagle_to_syspath()
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    from transformers import AutoTokenizer

    model = KVLlama.from_pretrained(target_path, torch_dtype=dtype,
                                    low_cpu_mem_usage=True).to(device)
    tok = AutoTokenizer.from_pretrained(target_path)
    if quant_config is None:
        spec = sb.default_ptq_args(rotate=True, optimized_rotation_path=r_bin_path)
    else:
        spec = sb.quant_config_to_args(quant_config, rotate=True,
                                       optimized_rotation_path=r_bin_path)
    stash = sb.apply_spinquant_pipeline(model, spec, input_model_id=input_model_id,
                                        stage=stage)
    model.to(device)
    return model, tok, stash
