"""Bridge to SpinQuant: subprocess wrappers for its own scripts, plus a
weight-level pipeline that applies SpinQuant's rotation/quantization to an
arbitrary LLaMA-class model (used to rotate EAGLE's vendored target).

Two usage modes:
  1. run_optimize_rotation() / run_ptq_eval(): shell out to SpinQuant's
     optimize_rotation.py / ptq.py under torchrun (scripts 10/11). Faithful to
     the upstream pipeline; produces R.bin and wikitext PPL.
  2. apply_spinquant_pipeline(model, spec): replicate ptq_model()'s transforms
     on a model object in-process, while stashing the tensors the EAGLE
     integration needs (R1, gamma_f, original lm_head/embedding). Used by
     src/rotated_target and the variant scripts.

All of this is fake quantization (QDQ, FP16 matmuls) — see docs/00 S4.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPINQUANT_DIR = os.path.join(PROJECT_ROOT, "third_party", "SpinQuant")


def add_spinquant_to_syspath() -> None:
    if SPINQUANT_DIR not in sys.path:
        sys.path.insert(0, SPINQUANT_DIR)


# ---------------------------------------------------------------------------
# args namespace matching utils/process_args.py defaults
# ---------------------------------------------------------------------------

def default_ptq_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        seed=0,
        # rotation
        rotate=False, rotate_mode="hadamard", rotation_seed=-1, fp32_had=False,
        optimized_rotation_path=None,
        # activation quant
        a_bits=16, a_groupsize=-1, a_asym=False, a_clip_ratio=1.0,
        # weight quant
        w_bits=16, w_groupsize=-1, w_asym=False, w_rtn=False, w_clip=False,
        nsamples=128, percdamp=0.01, act_order=False,
        # general
        int8_down_proj=False,
        # v cache
        v_bits=16, v_groupsize=-1, v_asym=False, v_clip_ratio=1.0,
        # k cache
        k_bits=16, k_groupsize=-1, k_asym=False, k_pre_rope=False, k_clip_ratio=1.0,
        # save/load/export
        load_qmodel_path=None, save_qmodel_path=None, export_to_et=False,
        capture_layer_io=False, layer_idx=10,
        bsz=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def quant_config_to_args(qc: dict, rotate: bool = True,
                         optimized_rotation_path: str | None = None,
                         **extra: Any) -> SimpleNamespace:
    """Translate configs/default_experiment.yaml:quantization into an args ns
    mirroring scripts/2_eval_ptq.sh semantics."""
    return default_ptq_args(
        rotate=rotate,
        optimized_rotation_path=optimized_rotation_path,
        w_bits=qc.get("w_bits", 16), a_bits=qc.get("a_bits", 16),
        k_bits=qc.get("k_bits", 16), v_bits=qc.get("v_bits", 16),
        w_rtn=(qc.get("w_method", "gptq") == "rtn"),
        w_clip=qc.get("w_clip", True),
        a_asym=qc.get("a_asym", True), k_asym=qc.get("k_asym", True),
        v_asym=qc.get("v_asym", True),
        k_groupsize=qc.get("k_groupsize", 128), v_groupsize=qc.get("v_groupsize", 128),
        nsamples=qc.get("gptq_calib_samples", 128),
        **extra,
    )


# ---------------------------------------------------------------------------
# subprocess wrappers for SpinQuant's own scripts
# ---------------------------------------------------------------------------

def _spinquant_env() -> dict:
    env = dict(os.environ)
    env["CUDA_HOME"] = env.get("CUDA_HOME_OVERRIDE", "/usr/local/cuda-12.8")
    env["PATH"] = env["CUDA_HOME"] + "/bin:" + env.get("PATH", "")
    # SpinQuant scripts run from repo root with implicit `import utils`
    env["PYTHONPATH"] = SPINQUANT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # RTX 4090 (Ada, no NVLink) lacks P2P/IB; NCCL must have these disabled or
    # accelerate/torchrun raises NotImplementedError on device setup.
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    # HF Trainer auto-enables wandb; disable it (headless, no api key).
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def build_optimize_rotation_cmd(input_model: str, output_rotation_path: str,
                                w_bits: int, a_bits: int, kv_bits: int,
                                nproc_per_node: int = 1, max_steps: int = 100,
                                learning_rate: float = 1.5, seqlen: int = 2048,
                                per_device_batch_size: int = 1,
                                access_token: str | None = None,
                                master_port: int = 29500) -> list[str]:
    """Mirror scripts/10_optimize_rotation.sh with configurable nproc/steps."""
    cmd = [
        "torchrun", "--nnodes=1", f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "optimize_rotation.py",
        "--input_model", input_model,
        "--output_rotation_path", output_rotation_path,
        "--output_dir", os.path.join(output_rotation_path, "hf_trainer"),
        "--logging_dir", os.path.join(output_rotation_path, "logs"),
        "--model_max_length", str(seqlen),
        "--fp16", "False", "--bf16", "True",
        "--log_on_each_node", "False",
        "--per_device_train_batch_size", str(per_device_batch_size),
        "--logging_steps", "1", "--learning_rate", str(learning_rate),
        "--weight_decay", "0.", "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "True",
        "--save_safetensors", "False",
        "--report_to", "none",
        "--max_steps", str(max_steps),
        "--w_bits", str(w_bits), "--a_bits", str(a_bits),
        "--k_bits", str(kv_bits), "--v_bits", str(kv_bits),
        "--w_clip", "--a_asym", "--k_asym", "--v_asym",
        "--k_groupsize", "128", "--v_groupsize", "128",
    ]
    if access_token:
        cmd += ["--access_token", access_token]
    return cmd


def build_ptq_eval_cmd(input_model: str, w_bits: int, a_bits: int, kv_bits: int,
                       optimized_rotation_path: str | None,
                       w_rtn: bool = False, rotate: bool = True,
                       eval_batch_size: int = 4, seqlen: int = 2048,
                       access_token: str | None = None,
                       save_qmodel_path: str | None = None,
                       master_port: int = 29501) -> list[str]:
    """Mirror scripts/2_eval_ptq.sh; kv_bits<16 enables R3+K/V cache quant."""
    cmd = [
        "torchrun", "--nnodes=1", "--nproc_per_node=1",
        f"--master_port={master_port}",
        "ptq.py",
        "--input_model", input_model,
        "--do_train", "False", "--do_eval", "True",
        "--per_device_eval_batch_size", str(eval_batch_size),
        "--model_max_length", str(seqlen),
        "--fp16", "False", "--bf16", "True",
        "--save_safetensors", "False",
        "--w_bits", str(w_bits), "--a_bits", str(a_bits),
        "--k_bits", str(kv_bits), "--v_bits", str(kv_bits),
        "--w_clip", "--a_asym", "--k_asym", "--v_asym",
        "--k_groupsize", "128", "--v_groupsize", "128",
    ]
    if rotate:
        cmd += ["--rotate"]
    if w_rtn:
        cmd += ["--w_rtn"]
    if optimized_rotation_path:
        cmd += ["--optimized_rotation_path", optimized_rotation_path]
    if save_qmodel_path:
        cmd += ["--save_qmodel_path", save_qmodel_path]
    if access_token:
        cmd += ["--access_token", access_token]
    return cmd


def run_spinquant_cmd(cmd: list[str], log_path: str, timeout: int | None = None) -> dict:
    """Run a SpinQuant torchrun command from the repo root, teeing to log_path."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=SPINQUANT_DIR, env=_spinquant_env(),
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
        )
    return {"returncode": proc.returncode, "cmd": cmd, "log": log_path}


# ---------------------------------------------------------------------------
# in-process weight-level pipeline (for the EAGLE-integrated target)
# ---------------------------------------------------------------------------

def stash_original_tensors(model) -> dict:
    """Snapshot the tensors the EAGLE draft interface needs, BEFORE any fuse/
    rotate mutates them: the final-norm scale gamma_f, and clean copies of the
    original lm_head and embedding (used for the draft's original-basis head and
    embedding — see docs/01 S2)."""
    gamma_f = model.model.norm.weight.data.detach().clone().float().cpu()
    lm_head_w = model.lm_head.weight.data.detach().clone().cpu()
    embed_w = model.model.embed_tokens.weight.data.detach().clone().cpu()
    return {"gamma_f": gamma_f, "lm_head_weight": lm_head_w, "embed_weight": embed_w}


def load_R1(optimized_rotation_path: str, device="cuda") -> torch.Tensor:
    obj = torch.load(optimized_rotation_path, map_location="cpu", weights_only=False)
    return obj["R1"].to(device).to(torch.float64)


@torch.inference_mode()
def make_random_rotation_bin(config, path: str, mode: str = "hadamard",
                             seed: int = 0) -> str:
    """Mint an R.bin (R1 + per-layer R2) with random/Hadamard orthogonal matrices,
    in the exact format rotate_model expects. Used to validate the rotation PORT in
    FP (any orthogonal R preserves logits) without waiting for learned rotations."""
    add_spinquant_to_syspath()
    from eval_utils import rotation_utils
    import torch as _t

    _t.manual_seed(seed)
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    n_layers = config.num_hidden_layers
    R = {"R1": rotation_utils.get_orthogonal_matrix(hidden, mode).to("cpu")}
    for i in range(n_layers):
        R[f"model.layers.{i}.self_attn.R2"] = \
            rotation_utils.get_orthogonal_matrix(head_dim, mode).to("cpu")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(R, path)
    return path


@torch.inference_mode()
def apply_spinquant_pipeline(model, spec: SimpleNamespace, input_model_id: str,
                             stage: str = "full") -> dict:
    """Apply SpinQuant's transforms in-process to `model` (a LLaMA-class HF or
    EAGLE-vendored model with .model.embed_tokens/.model.layers/.model.norm/
    .lm_head/.config). Returns stashed tensors (R1, gamma_f, original head/embed).

    stage:
      'rotate_only' : fuse norms + rotate (R1/R2/R4-fused). FP16 execution, no
                      activation/weight quant. Isolates the basis change.
      'full'        : rotate + ActQuantWrapper + weight quant (RTN/GPTQ) + online
                      R4 + K/V cache quant (+R3 if k_bits<16). Fake W4A4(KV4).

    Mirrors eval_utils/main.py:ptq_model but keeps the stashed tensors and works
    on EAGLE's model object too (module paths line up)."""
    add_spinquant_to_syspath()
    from eval_utils import gptq_utils, rotation_utils
    from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
    import transformers

    transformers.set_seed(spec.seed)
    model.eval()

    stashed = stash_original_tensors(model)
    if spec.optimized_rotation_path:
        stashed["R1"] = load_R1(spec.optimized_rotation_path).cpu()

    if not spec.rotate:
        quant_utils.add_actquant(model)
        return stashed

    # --- rotate (fuse norms first so gamma_f is already stashed above) ---
    fuse_norm_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, spec)
    utils.cleanup_memory(verbos=False)

    if spec.optimized_rotation_path is None:
        # random/hadamard R1 was generated inside rotate_model; recover it is not
        # exposed, so callers who need R1 must pass an optimized_rotation_path.
        stashed["R1"] = None

    if stage == "rotate_only":
        # Still add actquant wrappers (with 16-bit passthrough) so the online R4
        # Hadamard on down_proj is applied — rotation is only correct WITH R4.
        quant_utils.add_actquant(model)
        _enable_online_r4(model, hadamard_utils, quant_utils, spec)
        return stashed

    # --- full quant path (faithful to ptq_model) ---
    quant_utils.add_actquant(model)
    _enable_online_r4(model, hadamard_utils, quant_utils, spec)

    if spec.w_bits < 16:
        if spec.w_rtn:
            gptq_utils.rtn_fwrd(model, "cuda", spec)
        else:
            trainloader = data_utils.get_wikitext2(
                nsamples=spec.nsamples, seed=spec.seed,
                model=input_model_id, seqlen=2048, eval_mode=False,
            )
            gptq_utils.gptq_fwrd(model, trainloader, "cuda", spec)

    _configure_activation_and_kv_quant(model, spec, quant_utils, utils, rotation_utils)
    return stashed


def _enable_online_r4(model, hadamard_utils, quant_utils, spec) -> None:
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = spec.fp32_had


def _configure_activation_and_kv_quant(model, spec, quant_utils, utils, rotation_utils) -> None:
    if spec.a_bits < 16 or spec.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if spec.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, spec.a_groupsize)
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        for name in qlayers:
            layer_input_bits = spec.a_bits
            layer_groupsize = spec.a_groupsize
            layer_a_sym = not spec.a_asym
            layer_a_clip = spec.a_clip_ratio
            if "v_proj" in name and spec.v_bits < 16:
                qlayers[name].out_quantizer.configure(
                    bits=spec.v_bits, groupsize=head_dim,
                    sym=not spec.v_asym, clip_ratio=spec.v_clip_ratio)
            if "o_proj" in name:
                layer_groupsize = head_dim
            if "lm_head" in name:
                layer_input_bits = 16
            if "down_proj" in name:
                if spec.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize
            qlayers[name].quantizer.configure(
                bits=layer_input_bits, groupsize=layer_groupsize,
                sym=layer_a_sym, clip_ratio=layer_a_clip)

    if spec.k_bits < 16:
        assert not spec.k_pre_rope, "pre-RoPE K quant unsupported"
        k_quant_config = {
            "k_bits": spec.k_bits, "k_groupsize": spec.k_groupsize,
            "k_sym": not spec.k_asym, "k_clip_ratio": spec.k_clip_ratio,
        }
        for layer in model.model.layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, "apply_rotary_pos_emb",
                config=model.config, **k_quant_config)
