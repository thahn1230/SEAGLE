"""Rotated + fake-quantized Llama-3.1 target for DFlash (SEAGLE policy port).

Quantization policy (mirrors SEAGLE fake_w4a4_draft.py:48-50):
  weights     per-channel symmetric RTN + MSE clip (SpinQuant WeightQuantizer)
  activations per-token asymmetric (SpinQuant ActQuantizer), groupsize -1
  KV cache    bf16 (untouched)
  embed / lm_head / norms / RoPE / softmax / residual: bf16

Rotation: R1 global residual rotation + R2 per-head V/O + R4 online Hadamard
on down_proj input (weight side folded by rotate_mlp_output, so the online
activation Hadamard is MANDATORY in every rotated arm, including "FP16").
"""
from . import SPINQUANT_ROOT  # noqa: F401  (sys.path side effect)

import torch
from transformers import AutoModelForCausalLM

from utils import quant_utils, fuse_norm_utils, hadamard_utils
from utils.hadamard_utils import random_hadamard_matrix
from eval_utils import rotation_utils

WEIGHT_SCHEME = "per_channel_symmetric_RTN_MSEclip"
ACT_SCHEME = "per_token_asymmetric"


def mint_hadamard_rbin(hidden_size: int, head_dim: int, num_layers: int,
                       out_path: str, seed: int = 0) -> dict:
    """Random-Hadamard R.bin in the SEAGLE schema: {R1, layers.i.self_attn.R2}."""
    torch.manual_seed(seed)
    rb = {"R1": random_hadamard_matrix(hidden_size, "cuda").float().cpu()}
    for i in range(num_layers):
        rb[f"model.layers.{i}.self_attn.R2"] = (
            random_hadamard_matrix(head_dim, "cuda").float().cpu())
    torch.save(rb, out_path)
    return rb


def load_rbin(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_target(model_id: str, dtype=torch.bfloat16, device="cuda",
                attn="sdpa"):
    m = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn, dtype=dtype)
    return m.to(device).eval()


@torch.inference_mode()
def rotate_target(model, rbin: dict) -> None:
    """Fuse RMSNorm scales then fold R1/R2/R4-weight-side. In-place, on CPU
    weights (component fns stage through cuda fp64)."""
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    model.cpu()
    fuse_norm_utils.fuse_layer_norms(model)
    R1 = rbin["R1"].cuda().to(torch.float64)
    rotation_utils.rotate_embeddings(model, R1)
    rotation_utils.rotate_head(model, R1)
    for idx, layer in enumerate(model.model.layers):
        R2 = rbin[f"model.layers.{idx}.self_attn.R2"].cuda().to(torch.float64)
        rotation_utils.rotate_attention_inputs(layer, R1)
        rotation_utils.rotate_attention_output(layer, R1)
        rotation_utils.rotate_mlp_input(layer, R1)
        rotation_utils.rotate_mlp_output(layer, R1)   # includes R4 weight fold
        rotation_utils.rotate_ov_proj(layer, cfg.num_attention_heads, head_dim,
                                      R2=R2)


@torch.inference_mode()
def quantize_target_weights(model, w_bits: int) -> int:
    """Per-channel sym RTN + MSE clip on every decoder-layer Linear.
    Embed/head untouched. Returns number of quantized linears."""
    n = 0
    for layer in model.model.layers:
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                q = quant_utils.WeightQuantizer()
                q.configure(w_bits, perchannel=True, sym=True, mse=True)
                w = mod.weight.data.cuda()
                q.find_params(w)
                mod.weight.data = q.quantize(w).to(mod.weight.dtype).to(
                    mod.weight.device)
                n += 1
    return n


def install_act_quant(model, a_bits: int, fp32_had: bool = False,
                      online_had: bool = True) -> None:
    """Wrap decoder-layer linears in ActQuantWrapper; enable online R4 Hadamard
    on down_proj input (only when the weight side was folded by rotate_target);
    configure per-token asym act quant (16 = passthrough)."""
    quant_utils.add_actquant(model.model.layers)
    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    for layer in model.model.layers:
        for name, mod in layer.named_modules():
            if isinstance(mod, quant_utils.ActQuantWrapper):
                if online_had and name.endswith("mlp.down_proj"):
                    mod.online_full_had = True
                    mod.had_K = had_K
                    mod.K = K
                    mod.fp32_had = fp32_had
                if a_bits < 16:
                    mod.quantizer.configure(
                        bits=a_bits, groupsize=-1, sym=False, clip_ratio=1.0)


def build_target(model_id: str, mode: str, rbin_path: str | None = None,
                 dtype=torch.bfloat16, device="cuda", attn="sdpa"):
    """mode: fp16 | rot_fp16 | w8a8 | w4a4 (rotated, need rbin_path)
           | w4a4_norot (pure RTN, no rotation, no online Hadamard)."""
    model = load_target(model_id, dtype=dtype, device=device, attn=attn)
    if mode == "fp16":
        return model
    if mode == "w4a4_norot":
        model.cuda()
        quantize_target_weights(model, 4)
        install_act_quant(model, 4, online_had=False)
        return model.to(device).eval()
    assert rbin_path, "rotated modes need an R.bin"
    rbin = load_rbin(rbin_path)
    rotate_target(model, rbin)
    bits = {"rot_fp16": (16, 16), "w8a8": (8, 8), "w4a4": (4, 4)}[mode]
    if bits[0] < 16:
        model.cuda()
        quantize_target_weights(model, bits[0])
    install_act_quant(model, bits[1])
    return model.to(device).eval()
