"""Rotation-study core: variant adapters (naive/A/A_nogamma/B/B2/C), component-
controllable rotation+quant application, depth-truncated trees, and phase-split
timing. Everything installs by instance-attribute patching — third_party is
never edited.

Terminology (see docs/eagle_interface_audit.md):
  h      original-basis post-final-norm target hidden (what the draft expects)
  h_hat  rotated target hidden: h_hat = (h / gamma_f) @ R1
  A      runtime unrotation  h = (h_hat @ R1^T) * gamma_f
  B      first-layer-only fold: fc.weight[:, D:2D] <- W_h @ diag(gamma_f) @ R1
  B2     two-path fold: folded weights ONLY for the external call inside each
         topK_genrate; original weights for recycled draft-feature calls.
GPU policy: processes are expected to be launched with CUDA_VISIBLE_DEVICES
restricted to a subset of physical GPUs {4,5,6,7}; logical cuda:0 is used.
"""

from __future__ import annotations

import copy
import os
import time
from types import SimpleNamespace

import torch

from . import eagle_bridge, metrics, spinquant_bridge as sb
from .rotation_interface import build_original_head

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D_HIDDEN = 4096


# ---------------------------------------------------------------------------
# GPU policy guard
# ---------------------------------------------------------------------------

ALLOWED_PHYSICAL = {"4", "5", "6", "7"}


def assert_gpu_policy() -> dict:
    """Refuse to run unless CUDA_VISIBLE_DEVICES is a non-empty subset of the
    allowed physical GPUs {4,5,6,7}. Returns an info dict for logging."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is unset; refusing to run "
                           "(policy: physical GPUs 4-7 only)")
    ids = [x.strip() for x in cvd.split(",") if x.strip()]
    bad = [x for x in ids if x not in ALLOWED_PHYSICAL]
    if bad:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={cvd} includes forbidden "
                           f"physical GPUs {bad}; policy allows only 4-7")
    assert torch.cuda.device_count() <= 4, "more than 4 visible CUDA devices"
    return {"cuda_visible_devices": cvd,
            "visible_count": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i)
                      for i in range(torch.cuda.device_count())]}


# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------

def truncate_tree(depth: int) -> list:
    """mc_sim_7b_63 truncated to paths of length <= depth. Prefix-closure is
    asserted. depth=5 returns the full tree. depth=1 is rejected (EAGLE's own
    tree-buffer builder crashes on trees without level-2 nodes)."""
    eagle_bridge.add_eagle_to_syspath()
    from eagle.model.choices import mc_sim_7b_63
    if depth < 2:
        raise ValueError("depth must be >= 2 (see audit doc S5)")
    td = [list(p) for p in mc_sim_7b_63 if len(p) <= depth]
    s = set(map(tuple, td))
    assert all(tuple(p[:k]) in s for p in td for k in range(1, len(p))), \
        "truncated tree not prefix-closed"
    return td


def set_draft_tree(ea_model, tree_choices, device) -> None:
    """The draft keeps its OWN tree buffer (built at init from the full tree);
    a depth-truncated run must update it too, or draft/target trees disagree."""
    import eagle.model.cnets as cn
    ea_model.ea_layer.tree = tree_choices
    ea_model.ea_layer.tree_buffer = cn.generate_tree_buffers(tree_choices, device)


# ---------------------------------------------------------------------------
# CUDA-event accumulator (no per-call sync; summed once at finalize)
# ---------------------------------------------------------------------------

class EventAccum:
    def __init__(self):
        self.pairs = []

    def start(self):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        self.pairs.append((s, e))
        return e

    def total_ms(self) -> float:
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.pairs)

    @property
    def calls(self):
        return len(self.pairs)

    def reset(self):
        self.pairs = []


class PhaseTimers:
    """target / draft / verify accumulators + install/uninstall of wrappers."""

    def __init__(self, ea_model):
        self.ea_model = ea_model
        self.target = EventAccum()
        self.draft = EventAccum()
        self.verify = EventAccum()
        self._orig = {}

    def install(self):
        m = self.ea_model
        # target core forward (covers prefill, tree decoding, vanilla decode)
        tgt = m.base_model.model
        self._orig["tgt_fwd"] = tgt.forward
        acc_t = self.target

        def tgt_fwd(*a, **k):
            end = acc_t.start()
            out = self._orig["tgt_fwd"](*a, **k)
            end.record()
            return out
        tgt.forward = tgt_fwd

        # draft (outermost around whatever adapter wrapped topK_genrate)
        self._orig["topk"] = m.ea_layer.topK_genrate
        acc_d = self.draft

        def topk(*a, **k):
            end = acc_d.start()
            out = self._orig["topk"](*a, **k)
            end.record()
            return out
        m.ea_layer.topK_genrate = topk

        # verify: patch the name in ea_model's module globals (star-imported)
        import eagle.model.ea_model as eam
        self._orig["ep"] = eam.evaluate_posterior
        acc_v = self.verify

        def ep(*a, **k):
            end = acc_v.start()
            out = self._orig["ep"](*a, **k)
            end.record()
            return out
        eam.evaluate_posterior = ep
        return self

    def uninstall(self):
        m = self.ea_model
        if "tgt_fwd" in self._orig:
            m.base_model.model.forward = self._orig["tgt_fwd"]
        if "topk" in self._orig:
            m.ea_layer.topK_genrate = self._orig["topk"]
        if "ep" in self._orig:
            import eagle.model.ea_model as eam
            eam.evaluate_posterior = self._orig["ep"]
        self._orig = {}

    def reset(self):
        self.target.reset(); self.draft.reset(); self.verify.reset()


# ---------------------------------------------------------------------------
# variant adapters
# ---------------------------------------------------------------------------

class VariantAdapter:
    """Base: wraps ea_layer.topK_genrate, substitutes the ORIGINAL-basis head,
    optionally transforms the external hidden. Subclasses override."""
    name = "base"

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16):
        self.ea_model = ea_model
        self.ea_layer = ea_model.ea_layer
        self.device = device
        R1 = stash.get("R1")
        self.R1 = R1.to(device).to(torch.float32) if R1 is not None else None
        self.gamma = stash["gamma_f"].to(device).to(torch.float32)
        self.head = build_original_head(stash["lm_head_weight"], device, dtype)
        self.unrot = EventAccum()
        self._orig_topk = None

    # -- hooks ---------------------------------------------------------------
    def transform_hidden(self, hs: torch.Tensor) -> torch.Tensor:
        return hs

    def arm(self):
        """Called once per topK_genrate invocation, before the original fn."""

    # -- install -------------------------------------------------------------
    def install(self):
        assert self._orig_topk is None, "adapter already installed"
        self._orig_topk = self.ea_layer.topK_genrate
        adapter = self

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            hs = adapter.transform_hidden(hidden_states)
            adapter.arm()
            return adapter._orig_topk(hs, input_ids, adapter.head,
                                      logits_processor, *a, **k)
        self.ea_layer.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None

    def unrotation_stats(self):
        return {"unrotation_time_ms_total": self.unrot.total_ms() if self.unrot.pairs else 0.0,
                "unrotation_calls": self.unrot.calls}


class NaiveAdapter(VariantAdapter):
    name = "naive"


class UnrotateAdapter(VariantAdapter):
    """Variant A (with_gamma=True) / gamma ablation (with_gamma=False)."""
    def __init__(self, *a, with_gamma=True, **k):
        super().__init__(*a, **k)
        self.with_gamma = with_gamma
        self.name = "A" if with_gamma else "A_nogamma"

    def transform_hidden(self, hs):
        end = self.unrot.start()
        x = hs.to(torch.float32) @ self.R1.t()
        if self.with_gamma:
            x = x * self.gamma
        out = x.to(hs.dtype)
        end.record()
        return out


def fold_matrix(R1: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """M = diag(gamma_f) @ R1 in float64 (right-multiplied onto W_h)."""
    return gamma.double().unsqueeze(1) * R1.double()


class FoldAdapter(VariantAdapter):
    """Variant B: fold into fc's hidden block in place (backed up for restore)."""
    name = "B"

    def install(self):
        fc = self.ea_layer.fc
        D = fc.weight.shape[0]
        assert fc.weight.shape[1] == 2 * D
        self._w_backup = fc.weight.data.clone()
        M = fold_matrix(self.R1.to(fc.weight.device), self.gamma.to(fc.weight.device))
        W_h = fc.weight.data[:, D:2 * D].double()
        fc.weight.data[:, D:2 * D] = (W_h @ M).to(fc.weight.dtype)
        return super().install()

    def uninstall(self):
        super().uninstall()
        if hasattr(self, "_w_backup"):
            self.ea_layer.fc.weight.data.copy_(self._w_backup)
            del self._w_backup


class TwoPathAdapter(VariantAdapter):
    """Variant B2: two fc weight tensors; the folded one is used ONLY for the
    first draft forward after each topK_genrate entry (external h_hat), the
    original one for recycled-feature calls. Switch = .data pointer swap."""
    name = "B2"

    def install(self):
        fc = self.ea_layer.fc
        D = fc.weight.shape[0]
        self.W_orig = fc.weight.data
        Wf = fc.weight.data.clone()
        M = fold_matrix(self.R1.to(Wf.device), self.gamma.to(Wf.device))
        Wf[:, D:2 * D] = (self.W_orig[:, D:2 * D].double() @ M).to(Wf.dtype)
        self.W_folded = Wf
        self._pending_external = False

        adapter = self
        self._orig_forward = self.ea_layer.forward

        def patched_forward(hidden_states, *a, **k):
            if adapter._pending_external:
                fc.weight.data = adapter.W_folded
                adapter._pending_external = False
            else:
                fc.weight.data = adapter.W_orig
            return adapter._orig_forward(hidden_states, *a, **k)
        self.ea_layer.forward = patched_forward
        return super().install()

    def arm(self):
        self._pending_external = True

    def uninstall(self):
        super().uninstall()
        if hasattr(self, "_orig_forward"):
            self.ea_layer.forward = self._orig_forward
            del self._orig_forward
        self.ea_layer.fc.weight.data = self.W_orig


ADAPTERS = {
    "naive": lambda m, s, dev, dt: NaiveAdapter(m, s, dev, dt),
    "A": lambda m, s, dev, dt: UnrotateAdapter(m, s, dev, dt, with_gamma=True),
    "A_nogamma": lambda m, s, dev, dt: UnrotateAdapter(m, s, dev, dt, with_gamma=False),
    "B": lambda m, s, dev, dt: FoldAdapter(m, s, dev, dt),
    "B2": lambda m, s, dev, dt: TwoPathAdapter(m, s, dev, dt),
    # 'C' = load retrained draft, then use the A adapter (handled by runner)
}


# ---------------------------------------------------------------------------
# target construction: rotation components x quantization
# ---------------------------------------------------------------------------

QUANT_CFGS = {
    "none": None,
    "w4a16": {"w_bits": 4, "a_bits": 16, "k_bits": 16, "v_bits": 16},
    "w4a4": {"w_bits": 4, "a_bits": 4, "k_bits": 16, "v_bits": 16},
    "w4a4kv4": {"w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4},
    # bitwidth-AL component-causality study (weight quant auto-skipped when
    # w_bits==16 via `spec.w_bits < 16`; ActQuantizer no-ops at bits==16)
    "w8a8": {"w_bits": 8, "a_bits": 8, "k_bits": 16, "v_bits": 16},
    "w8a16": {"w_bits": 8, "a_bits": 16, "k_bits": 16, "v_bits": 16},
    "w16a8": {"w_bits": 16, "a_bits": 8, "k_bits": 16, "v_bits": 16},
    "w16a4": {"w_bits": 16, "a_bits": 4, "k_bits": 16, "v_bits": 16},
}

ROTATION_COMPONENTS = {
    "none": set(),
    "r1": {"r1"},
    "r1r2": {"r1", "r2"},
    "r1r2r3r4": {"r1", "r2", "r3", "r4"},
    "full": {"r1", "r2", "r4"},   # SpinQuant standard; r3 auto-added iff k_bits<16
}


def r_bin_path(rotation_type: str, seed: int, target_path: str,
               rotations_root: str | None = None) -> str | None:
    """rotations_root defaults to the original (Llama-2) location; a second
    model pair MUST pass its own root so R.bin files are never shared across
    targets in the artifact record (mathematically a random Hadamard R1 is
    model-independent for equal dims, but runs must stay self-documenting)."""
    root = rotations_root or os.path.join(PROJECT_ROOT, "outputs", "rotations")
    if rotation_type == "learned":
        p = os.path.join(root, "learned_w16a4kv4", "R.bin")
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} (no learned rotation trained for this target; "
                "train one or use rotation-type random_hadamard)")
        return p
    # a named rotation directory (e.g. 'eagle_aware'): outputs/rotations/<name>/R.bin
    if rotation_type not in ("random_hadamard",):
        p = os.path.join(root, rotation_type, "R.bin")
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"{p} (unknown rotation-type '{rotation_type}')")
    if seed == 0:
        p = os.path.join(root, "random_hadamard", "R.bin")
    else:
        p = os.path.join(root, f"random_hadamard_seed{seed}", "R.bin")
    if not os.path.isfile(p):
        from transformers import AutoConfig
        conf = AutoConfig.from_pretrained(target_path)
        sb.make_random_rotation_bin(conf, p, mode="hadamard", seed=seed)
    return p


@torch.inference_mode()
def apply_rotation_quant(base_model, rotation: str, r_bin: str | None,
                         quant: str, input_model_id: str, device: str) -> dict:
    """Apply the requested rotation components + quantization to base_model.
    Returns the stash (gamma_f, original lm_head/embed weights, R1).

    rotation='full' reproduces the exact pipeline used in earlier published
    runs (sb.apply_spinquant_pipeline). Component configs re-implement the
    fusions selectively; notably rotate_mlp_output is split so 'r1'-only does
    NOT fold the R4 Hadamard into down_proj (that is only valid together with
    the online Hadamard)."""
    sb.add_spinquant_to_syspath()
    quant_cfg = QUANT_CFGS[quant]

    if rotation == "full":
        if quant == "none":
            spec = sb.default_ptq_args(rotate=True, optimized_rotation_path=r_bin)
            stash = sb.apply_spinquant_pipeline(base_model, spec,
                                                input_model_id=input_model_id,
                                                stage="rotate_only")
        else:
            qc = dict(quant_cfg); qc["w_method"] = "rtn"
            qc.update({"w_clip": True, "a_asym": True, "k_asym": True,
                       "v_asym": True, "k_groupsize": 128, "v_groupsize": 128})
            spec = sb.quant_config_to_args(qc, rotate=True,
                                           optimized_rotation_path=r_bin)
            stash = sb.apply_spinquant_pipeline(base_model, spec,
                                                input_model_id=input_model_id,
                                                stage="full")
        base_model.to(device)
        return stash

    # ---- component-controlled path ----
    from eval_utils import rotation_utils, gptq_utils
    from utils import fuse_norm_utils, hadamard_utils, quant_utils, utils as squtils
    import transformers

    comps = ROTATION_COMPONENTS[rotation]
    # Guard (review finding): K-cache quantization installs the QK wrapper,
    # which ALWAYS applies the R3 Hadamard. A component config without 'r3'
    # combined with kv4 would silently run R3 while being labeled otherwise.
    if quant_cfg and quant_cfg.get("k_bits", 16) < 16 and "r3" not in comps:
        raise ValueError(f"rotation='{rotation}' with K-cache quantization would "
                         "silently activate R3 (QK wrapper); use a rotation set "
                         "containing r3 or rotation='full'")
    stash = sb.stash_original_tensors(base_model)

    if comps:
        assert r_bin is not None
        R = torch.load(r_bin, map_location="cpu", weights_only=False)
        R1 = R["R1"].cuda().to(torch.float64)
        stash["R1"] = R["R1"].clone()
        cfg = base_model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        fuse_norm_utils.fuse_layer_norms(base_model)
        if "r1" in comps:
            rotation_utils.rotate_embeddings(base_model, R1)
            rotation_utils.rotate_head(base_model, R1)
            for idx, layer in enumerate(base_model.model.layers):
                rotation_utils.rotate_attention_inputs(layer, R1)
                rotation_utils.rotate_attention_output(layer, R1)
                rotation_utils.rotate_mlp_input(layer, R1)
                # R1 part of rotate_mlp_output WITHOUT the R4 fold:
                W = layer.mlp.down_proj
                dt = W.weight.data.dtype
                W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
                W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dt)
        if "r2" in comps:
            for idx, layer in enumerate(base_model.model.layers):
                key = f"model.layers.{idx}.self_attn.R2"
                R2 = R[key].cuda().to(torch.float64)
                rotation_utils.rotate_ov_proj(layer, cfg.num_attention_heads,
                                              head_dim, R2=R2)
        if "r4" in comps:
            for layer in base_model.model.layers:
                hadamard_utils.apply_exact_had_to_linear(
                    layer.mlp.down_proj, had_dim=-1, output=False)
    else:
        stash["R1"] = None

    need_wrappers = ("r4" in comps) or (quant_cfg is not None)
    spec = sb.default_ptq_args(rotate=bool(comps))
    if quant_cfg:
        qc = dict(quant_cfg)
        spec = sb.default_ptq_args(
            rotate=bool(comps), w_bits=qc["w_bits"], a_bits=qc["a_bits"],
            k_bits=qc["k_bits"], v_bits=qc["v_bits"], w_rtn=True, w_clip=True,
            a_asym=True, k_asym=True, v_asym=True,
            k_groupsize=128, v_groupsize=128)

    if need_wrappers:
        quant_utils.add_actquant(base_model)
        if "r4" in comps:
            sb._enable_online_r4(base_model, hadamard_utils, quant_utils, spec)

    if quant_cfg and spec.w_bits < 16:
        transformers.set_seed(spec.seed)
        gptq_utils.rtn_fwrd(base_model, "cuda", spec)

    if quant_cfg:
        sb._configure_activation_and_kv_quant(base_model, spec, quant_utils,
                                              squtils, rotation_utils)
    if "r3" in comps and (quant_cfg is None or spec.k_bits >= 16):
        # pure online R3 Hadamard (no K quant): wrapper with k_bits=16
        k_cfg = {"k_bits": 16, "k_groupsize": 128, "k_sym": True,
                 "k_clip_ratio": 1.0}
        for layer in base_model.model.layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, "apply_rotary_pos_emb",
                config=base_model.config, **k_cfg)

    base_model.to(device)
    return stash


def build_study_target(target_path: str, draft_path: str, input_model_id: str,
                       rotation: str, rotation_type: str, quant: str,
                       seed: int, device: str = "cuda:0",
                       dtype=torch.float16, rotations_root: str | None = None):
    """Load EaModel and apply the requested (rotation, quant) to its target.
    Returns (ea_model, stash, r_bin)."""
    torch.manual_seed(seed)
    model = eagle_bridge.load_eagle_model(
        base_model_path=target_path, ea_model_path=draft_path,
        dtype=dtype, device_map=device)
    r_bin = None
    if rotation != "none":
        r_bin = r_bin_path(rotation_type, seed, target_path, rotations_root)
    stash = apply_rotation_quant(model.base_model, rotation, r_bin, quant,
                                 input_model_id, device)
    model.base_model.to(device)
    model.ea_layer.to(device)
    return model, stash, r_bin


# ---------------------------------------------------------------------------
# generation + measurement (one prompt)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_one_prompt(ea_model, timers: PhaseTimers, input_ids: torch.Tensor,
                   mode: str, max_new_tokens: int, tree_choices,
                   max_depth: int = 8) -> dict:
    device = ea_model.base_model.lm_head.weight.device
    input_ids = input_ids.to(device)
    input_len = input_ids.shape[1]
    timers.reset()
    metrics.reset_peak_memory(device)
    tracker = metrics.DepthAcceptanceTracker(max_depth=max_depth)
    rounds, prev_len = 0, input_len

    if mode == "eagle":
        gen = ea_model.ea_generate(input_ids, temperature=0.0,
                                   max_steps=max_new_tokens,
                                   tree_choices=tree_choices)
    else:
        gen = ea_model.naive_generate(input_ids, temperature=0.0,
                                      max_steps=max_new_tokens)

    final_ids = input_ids
    with metrics.cuda_timer(device) as t:
        for out_ids in gen:
            cur = out_ids.shape[1]
            if cur > prev_len:
                tracker.record(max(cur - prev_len - 1, 0))
                rounds += 1
                prev_len = cur
            final_ids = out_ids
            if cur - input_len >= max_new_tokens:
                break

    new_tokens = final_ids.shape[1] - input_len
    total_ms = t["ms"]
    return {
        "context_len": input_len,
        "generated_tokens": new_tokens,
        "target_forwards": rounds,
        "accepted_tokens_total": new_tokens,
        "acceptance_length": new_tokens / rounds if rounds else None,
        "total_ms": total_ms,
        "tokens_per_second": metrics.throughput_tokens_per_s(new_tokens, total_ms),
        "target_time_ms_total": timers.target.total_ms(),
        "draft_time_ms_total": timers.draft.total_ms(),
        "verify_time_ms_total": timers.verify.total_ms(),
        "peak_mem_gib": metrics.peak_memory_gib(device),
        "alpha_by_depth": tracker.alpha_by_depth(),
    }
