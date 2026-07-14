"""Explicit-unfused-tail study (docs/tail_unfused_overhead_summary.md).

SpinQuant fuses R1 + the final RMSNorm scale gamma_f + lm_head into ONE fused
lm_head, so the target's `model.norm` output that EAGLE captures is h_hat
(rotated basis) and the frozen draft needs an A/B2 correction. This module
implements an alternative: run the final tail EXPLICITLY so `model.norm` emits
the true original hidden h, and measure the overhead.

Bases (row-vector convention, orthogonal R1, a := RMSNorm0(x) scale-free):
  x_R = x @ R1            (residual entering the final norm in the fused target)
  RMSNorm0(x_R) = a @ R1  (orthogonal R1 preserves the RMS -> commutes into a)
  original h = a * gamma_f              (EAGLE-compatible hidden)
  fused    h_hat = RMSNorm0(x_R) = a @ R1

Tail variants (what `model.norm` emits + which lm_head is used):
  T0_original_fp16                 x  -> a*gamma_f            -> W_lm      [unrotated ref]
  T1_spinquant_fused               x_R-> a@R1 (=h_hat)        -> W_fused
  T2_explicit_unfused_correct      x_R-> (x_R@R1.T)=x; a*g    -> W_lm
  T3_explicit_unfused_postnorm     x_R-> (a@R1)@R1.T=a; a*g   -> W_lm
  T4_wrong_order_gamma_after_rot   x_R-> RMSNorm0(x_R)*g      -> W_lm      [DIAGNOSTIC, wrong]

T2 and T3 emit the SAME h (differ only in whether R1.T runs before or after the
norm); T4 keeps gamma in the rotated basis (a@R1)*g != a*g and is intentionally
wrong unless gamma_f is uniform.
"""

from __future__ import annotations

import torch

from .study import EventAccum

D_HIDDEN = 4096
TAIL_MODES = ["original_fp16", "spinquant_fused", "explicit_unfused_correct",
              "explicit_unfused_postnorm_equiv", "wrong_order_gamma_after_rotation",
              "eagle_friendly_hR"]
MODE_TAG = {"original_fp16": "T0", "spinquant_fused": "T1",
            "explicit_unfused_correct": "T2",
            "explicit_unfused_postnorm_equiv": "T3",
            "wrong_order_gamma_after_rotation": "T4",
            "eagle_friendly_hR": "T5"}


# ---------------------------------------------------------------------------
# pure tail functions (microbench + correctness); no timing, no model state
# ---------------------------------------------------------------------------

def rmsnorm0(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale-free RMSNorm, matching modeling_llama_kv.LlamaRMSNorm exactly:
    variance in fp32, normalized value cast back to input dtype BEFORE any
    weight multiply."""
    dt = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(dt)


def _rot(x, Rt):
    return (x @ Rt).to(x.dtype)


def tail_T0(x, gamma, W_lm, eps):
    h = rmsnorm0(x, eps) * gamma
    return h, h @ W_lm.t()


def tail_T1(x_R, W_fused, eps):
    h_hat = rmsnorm0(x_R, eps)
    return h_hat, h_hat @ W_fused.t()


def tail_T2(x_R, R1t, gamma, W_lm, eps):
    x = _rot(x_R, R1t)
    h = rmsnorm0(x, eps) * gamma
    return h, h @ W_lm.t()


def tail_T3(x_R, R1t, gamma, W_lm, eps):
    a_R = rmsnorm0(x_R, eps)
    a = _rot(a_R, R1t)
    h = a * gamma
    return h, h @ W_lm.t()


def tail_T4(x_R, gamma, W_lm, eps):
    a_R = rmsnorm0(x_R, eps)
    h = a_R * gamma                      # gamma applied in ROTATED coords: wrong
    return h, h @ W_lm.t()


def run_tail(mode, *, x=None, x_R=None, R1t=None, gamma=None, W_lm=None,
             W_fused=None, eps=1e-5):
    """Dispatch by mode; returns (exposed_hidden, logits)."""
    if mode == "original_fp16":
        return tail_T0(x, gamma, W_lm, eps)
    if mode == "spinquant_fused":
        return tail_T1(x_R, W_fused, eps)
    if mode == "explicit_unfused_correct":
        return tail_T2(x_R, R1t, gamma, W_lm, eps)
    if mode == "explicit_unfused_postnorm_equiv":
        return tail_T3(x_R, R1t, gamma, W_lm, eps)
    if mode == "wrong_order_gamma_after_rotation":
        return tail_T4(x_R, gamma, W_lm, eps)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# runtime adapter: patch model.norm.forward + swap lm_head to realize a tail
# mode on a LIVE (SpinQuant-fused) target
# ---------------------------------------------------------------------------

class TailAdapter:
    """Realize a tail mode on a fused-SpinQuant EAGLE target by patching
    `base_model.model.norm.forward` (its INPUT is x_R, the rotated residual)
    and, for the unfused modes, restoring `base_model.lm_head` to the ORIGINAL
    W_lm so logits stay correct while the emitted hidden becomes original h.

    Requires a fused/rotated target (rotation='full', quant off). Do NOT use on
    mode 'original_fp16' — that is an unrotated reference target, handled by the
    caller with no adapter.

    Timing: per-forward CUDA-event pairs for the R1.T GEMM and the RMSNorm are
    accumulated (summed lazily). lm_head is timed by a separate wrapper.
    """

    def __init__(self, ea_model, stash, mode: str, rot_dtype=torch.float16):
        assert mode in TAIL_MODES and mode != "original_fp16"
        self.ea_model = ea_model
        self.mode = mode
        self.norm = ea_model.base_model.model.norm
        self.lm_head = ea_model.base_model.lm_head
        self.eps = float(self.norm.variance_epsilon)
        dev = self.lm_head.weight.device
        self.rot_dtype = rot_dtype
        R1 = stash["R1"].to(dev)
        self.R1t = R1.t().contiguous().to(rot_dtype)          # R1^T
        self.gamma = stash["gamma_f"].to(dev).to(torch.float16)
        self.W_lm_orig = stash["lm_head_weight"].to(dev).to(torch.float16)
        # T5 (eagle_friendly_hR): emit h_R = h @ R1 = h_hat @ G where
        # G = R1^T diag(gamma_f) R1, and score with W_lm @ R1.
        R1_64 = R1.to(torch.float64)
        self.G = (R1_64.t() @ torch.diag(stash["gamma_f"].to(dev).double())
                  @ R1_64).to(rot_dtype)
        self.W_lm_R = (self.W_lm_orig.double() @ R1_64).to(torch.float16)
        self.r1t = EventAccum()
        self.rms = EventAccum()
        self.head = EventAccum()
        self._installed = False

    # -- patched norm --------------------------------------------------------
    def _norm_forward(self, x_R):
        mode = self.mode
        if mode == "spinquant_fused":
            e = self.rms.start(); h = rmsnorm0(x_R, self.eps); e.record()
            return h
        if mode == "explicit_unfused_correct":                 # T2
            e = self.r1t.start()
            x = (x_R.to(self.rot_dtype) @ self.R1t).to(x_R.dtype)
            e.record()
            e2 = self.rms.start(); h = rmsnorm0(x, self.eps) * self.gamma; e2.record()
            return h
        if mode == "explicit_unfused_postnorm_equiv":          # T3
            e2 = self.rms.start(); a_R = rmsnorm0(x_R, self.eps); e2.record()
            e = self.r1t.start()
            a = (a_R.to(self.rot_dtype) @ self.R1t).to(x_R.dtype)
            h = a * self.gamma                # count gamma inside r1t bracket,
            e.record()                        # matching T2 (gamma in a timed op)
            return h
        if mode == "wrong_order_gamma_after_rotation":         # T4 (wrong)
            e2 = self.rms.start()
            h = rmsnorm0(x_R, self.eps) * self.gamma
            e2.record()
            return h
        if mode == "eagle_friendly_hR":                        # T5
            # h_hat = RMSNorm0(x_R); h_R = h_hat @ G  (one dense GEMM)
            e2 = self.rms.start(); h_hat = rmsnorm0(x_R, self.eps); e2.record()
            e = self.r1t.start()
            h_R = (h_hat.to(self.rot_dtype) @ self.G).to(x_R.dtype)
            e.record()
            return h_R
        raise ValueError(mode)

    def install(self):
        assert not self._installed
        self._orig_norm_forward = self.norm.forward
        self._orig_lm_weight = self.lm_head.weight.data
        self.norm.forward = self._norm_forward
        if self.mode == "eagle_friendly_hR":
            # emits h_R -> logits must use W_lm @ R1
            self.lm_head.weight.data = self.W_lm_R
        elif self.mode != "spinquant_fused":
            # T2/T3/T4 emit original-basis h -> logits use the ORIGINAL head
            self.lm_head.weight.data = self.W_lm_orig
        # time lm_head
        self._orig_head_forward = self.lm_head.forward
        acc = self.head

        def head_forward(x, *a, **k):
            e = acc.start()
            out = self._orig_head_forward(x, *a, **k)
            e.record()
            return out
        self.lm_head.forward = head_forward
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        self.norm.forward = self._orig_norm_forward
        self.lm_head.weight.data = self._orig_lm_weight
        self.lm_head.forward = self._orig_head_forward
        self._installed = False

    def reset_timing(self):
        self.r1t.reset(); self.rms.reset(); self.head.reset()

    def timing(self):
        return {
            "r1t_ms_total": self.r1t.total_ms() if self.r1t.pairs else 0.0,
            "r1t_calls": self.r1t.calls,
            "rmsnorm_ms_total": self.rms.total_ms() if self.rms.pairs else 0.0,
            "rmsnorm_calls": self.rms.calls,
            "lm_head_ms_total": self.head.total_ms() if self.head.pairs else 0.0,
            "lm_head_calls": self.head.calls,
        }

    @property
    def hidden_basis_exposed(self):
        return {"spinquant_fused": "rotated_h_hat",
                "explicit_unfused_correct": "original_h",
                "explicit_unfused_postnorm_equiv": "original_h",
                "wrong_order_gamma_after_rotation": "wrong_rotated_gamma",
                "eagle_friendly_hR": "rotated_h_R"}[self.mode]
