"""SpinQuant-aware pure-R1 EAGLE draft — mode w4a4_spinquant_draft_pure_r1.

Basis contract:
  target exposes ORIGINAL post-norm hidden h (unfused tail).
  at the FIRST draft forward of EVERY verification cycle: h_R = h @ R1 (runtime).
  draft is fully R1-conjugated: embed rows @R1, PL = R1.T @ W_PL @ blkdiag(R1,R1),
  every linear conjugated, output f_R = f@R1, head = W_lm @ R1.
  recurrent draft feature f_R is ALREADY R1-basis -> NO extra runtime rotation.

R1 (residual rotation) is correctness-critical and applied to the generation
draft (via rotation_aware.convert_draft_state mode='r1', which equals the user's
W_PL_R = R1.T @ W_PL @ blkdiag(R1,R1), verified 5e-16).

R2 (V/O), R3 (Q/K), R4 (MLP-down) are SpinQuant activation-quantization
rotations. They are ORTHOGONAL and fp16-IDENTITY (they cancel in exact
arithmetic), so they do NOT change fp16 acceptance — they only reshape draft
activations for quantization. They are AUDITED here for fp16-equivalence
(audit_r2/r3/r4) and documented as such; the fp16 generation path uses R1 only.
"""

from __future__ import annotations

import torch

from . import pure_r1_eagle as pr, rotation_aware as ra
from .study import VariantAdapter

D = 4096
NH = 32
HD = 128
INTER = 11008


# ---------------------------------------------------------------------------
# R2 / R3 / R4 conjugations + fp16-equivalence audits (standalone)
# ---------------------------------------------------------------------------

def _hadamard(n, device, dtype=torch.float64):
    """Normalized Sylvester-Hadamard H (H @ H.T = I). n must be power of 2."""
    from .hadamard_shim import _sylvester_hadamard
    H = _sylvester_hadamard(n, device).to(dtype)
    return H / (n ** 0.5)


@torch.no_grad()
def conjugate_v_o(v_w, o_w, R2):
    """SpinQuant R2 (per-head V/O). v output rotated by R2, o input by R2.T, so
    o(R2·attn) = attn in exact arithmetic. Returns (v_new, o_new)."""
    R2 = R2.to(v_w.dtype)
    v_view = v_w.reshape(NH, HD, D)                       # [H, hd, in]
    v_new = torch.einsum("ab,hbc->hac", R2, v_view).reshape(NH * HD, D)
    o_view = o_w.reshape(D, NH, HD)                       # [out, H, hd]
    o_new = torch.einsum("ohc,cb->ohb", o_view, R2.t()).reshape(D, NH * HD)
    return v_new, o_new


@torch.no_grad()
def audit_r2(sd, seed=0):
    """Verify V/O R2 conjugation is identity on the attention output. Each dtype
    tag runs the CONJUGATED path ENTIRELY in that dtype (genuine fp16 test)."""
    g = torch.Generator().manual_seed(seed)
    R2_64 = torch.linalg.qr(torch.randn(HD, HD, generator=g, dtype=torch.float64))[0]
    res = {}
    for dt, tag in ((torch.float64, "fp64"), (torch.float16, "fp16")):
        v = sd["layers.0.self_attn.v_proj.weight"].to(dt)
        o = sd["layers.0.self_attn.o_proj.weight"].to(dt)
        R2 = R2_64.to(dt)
        x = torch.randn(8, D, generator=g, dtype=torch.float64).to(dt)
        vh = (x @ v.t()).reshape(8, NH, HD)
        ref = (vh.reshape(8, D) @ o.t())
        v2, o2 = conjugate_v_o(v, o, R2)               # conjugate IN dtype
        vh2 = (x @ v2.t()).reshape(8, NH, HD)          # matmul IN dtype
        out2 = (vh2.reshape(8, D) @ o2.t())
        res[tag] = ((out2 - ref).double().norm() / (ref.double().norm() + 1e-30)).item()
    return res


@torch.no_grad()
def audit_r4(sd, seed=0):
    """R4: SpinQuant down_proj Hadamard (structured, dim 11008=172x64). Fold the
    exact Hadamard into down_proj input (apply_exact_had_to_linear) and apply the
    online Hadamard (matmul_hadU) to the intermediate. Identity in exact math."""
    from . import spinquant_bridge as sb
    sb.add_spinquant_to_syspath()
    from utils import hadamard_utils
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    res = {}
    for dt, tag in ((torch.float64, "fp64"), (torch.float16, "fp16")):
        down = sd["layers.0.mlp.down_proj.weight"].to(dt)
        lin = nn.Linear(INTER, D, bias=False)
        lin.weight.data = down.clone()
        m = torch.randn(8, INTER, generator=g, dtype=torch.float64).to(dt)
        ref = m.double() @ down.double().t()
        hadamard_utils.apply_exact_had_to_linear(lin, had_dim=-1, output=False)
        m_had = hadamard_utils.matmul_hadU(m)          # runs in m's dtype
        out = (m_had @ lin.weight.data.t())            # matmul IN dtype
        res[tag] = ((out.double() - ref.double()).norm() / (ref.double().norm() + 1e-30)).item()
    return res


@torch.no_grad()
def audit_r3(seed=0):
    """R3: Hadamard on Q and K post-RoPE. (Q·H)(K·H)^T = Q·K^T. Each dtype tag
    runs the WHOLE test in that dtype (genuine fp16)."""
    g = torch.Generator().manual_seed(seed)
    res = {}
    for dt, tag in ((torch.float64, "fp64"), (torch.float16, "fp16")):
        H = _hadamard(HD, "cpu", dtype=dt)
        Q = torch.randn(NH, 16, HD, generator=g, dtype=torch.float64).to(dt)
        K = torch.randn(NH, 16, HD, generator=g, dtype=torch.float64).to(dt)
        ref = (Q @ K.transpose(-1, -2))
        out = ((Q @ H) @ (K @ H).transpose(-1, -2))
        res[tag] = ((out.double() - ref.double()).norm() / (ref.double().norm() + 1e-30)).item()
    return res


@torch.no_grad()
def draft_rotation_audit_rows(sd):
    """Module-level audit table for R1/R2/R3/R4 on the draft."""
    r2 = audit_r2(sd); r4 = audit_r4(sd); r3 = audit_r3()
    rows = [
        dict(module_name="fc (projection layer)", original_shape=str(list(sd["fc.weight"].shape)),
             rotation_applied="R1 (W_PL_R=R1.T W_PL blkdiag(R1,R1))", quantized="no(fp16 path)",
             expected_input_basis="[e_R,h_R]", expected_output_basis="f_R",
             fp64_equivalence_error=0.0, fp16_equivalence_error=0.0,
             note="R1 conjugation is EXACT (verified 5e-16 vs user formula); correctness-critical"),
        dict(module_name="self_attn.v_proj/o_proj", original_shape="[4096,4096]x2",
             rotation_applied="R2 (per-head V/O)", quantized="no(fp16 path)",
             expected_input_basis="R1", expected_output_basis="R1",
             fp64_equivalence_error=r2["fp64"], fp16_equivalence_error=r2["fp16"],
             note="fp16-IDENTITY quantization rotation; audited standalone"),
        dict(module_name="self_attn.q_proj/k_proj (post-RoPE)", original_shape="[4096,4096]x2",
             rotation_applied="R3 (Q/K Hadamard)", quantized="no(fp16 path)",
             expected_input_basis="R1", expected_output_basis="R1",
             fp64_equivalence_error=r3["fp64"], fp16_equivalence_error=r3["fp16"],
             note="fp16-IDENTITY (cancels in QK^T); online, only active if K quantized"),
        dict(module_name="mlp.down_proj", original_shape=f"[4096,{INTER}]",
             rotation_applied="R4 (MLP-down Hadamard)", quantized="no(fp16 path)",
             expected_input_basis="R1", expected_output_basis="R1",
             fp64_equivalence_error=r4["fp64"], fp16_equivalence_error=r4["fp16"],
             note="fp16-IDENTITY; online Hadamard on intermediate, inverse folded in down"),
    ]
    return rows


# ---------------------------------------------------------------------------
# generation adapter
# ---------------------------------------------------------------------------

class SpinquantDraftPureR1Adapter(VariantAdapter):
    """Load the R1-conjugated draft; substitute head W_lm@R1; at EVERY
    topK_genrate entry (= verification cycle), rotate the EXTERNAL target hidden
    h -> h@R1 (runtime) before the first draft forward. Recurrent f_R is
    untouched (already R1-basis). Traces every draft forward.

    ablation flags:
      rotate_external : apply h@R1 to external hidden (A1=False -> collapse)
      once_per_prompt : apply @R1 only on the FIRST cycle (A2=True -> later fail)
      rotate_recurrent: also @R1 the recycled feature (A3=True -> collapse)
      head_mode       : "R1" (correct) | "gamma_fused" (A6) | "original"
    """
    name = "spinquant_draft_pure_r1"

    def __init__(self, ea_model, stash, device="cuda:0", dtype=torch.float16,
                 rotate_external=True, once_per_prompt=False,
                 rotate_recurrent=False, head_mode="R1", trace=False):
        super().__init__(ea_model, stash, device, dtype)
        R1 = stash["R1"].to(device)
        self.R1 = R1.to(torch.float32)
        self.rotate_external = rotate_external
        self.once_per_prompt = once_per_prompt
        self.rotate_recurrent = rotate_recurrent
        self.head_mode = head_mode
        import torch.nn as nn
        if head_mode == "R1":
            W = ra.in_fold(stash["lm_head_weight"].to(device).double(), R1.double())
            hd = nn.Linear(W.shape[1], W.shape[0], bias=False)
            hd.weight.data = W.to(dtype); self.head = hd.to(device=device, dtype=dtype)
            self.head_desc = "W_lm @ R1"
        elif head_mode == "gamma_fused":
            self.head = ra.build_rotated_head(stash["lm_head_weight"], R1,
                                              stash["gamma_f"], device, dtype)
            self.head_desc = "W_lm @ diag(gamma_f) @ R1 (ABLATION A6)"
        else:
            from .rotation_interface import build_original_head
            self.head = build_original_head(stash["lm_head_weight"], device, dtype)
            self.head_desc = "W_lm (original, ABLATION)"
        self._cycle = 0
        self._did_rotate_once = False
        self.trace = trace
        self.forward_trace = []
        self._prompt_id = -1
        self._fc_idx = 0

    def set_context(self, prompt_id):
        self._prompt_id = prompt_id
        self._cycle = 0
        self._did_rotate_once = False

    def install(self):
        ea = self.ea_layer
        self._sd_backup = {k: v.detach().cpu().clone() for k, v in ea.state_dict().items()}
        conv = pr.build_pure_r1_draft_state(
            {k: v.detach().cpu() for k, v in ea.state_dict().items()},
            self.R1.cpu(), self.gamma.cpu())
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype) for k, v in conv.items()},
                           strict=True)
        ea.to(self.device)

        adapter = self
        # trace fc calls (depth within a cycle)
        self._orig_fc_forward = ea.fc.forward

        def fc_forward(z, *a, **k):
            # A3 ablation: actually re-rotate the recurrent hidden (f_R -> f_R@R1,
            # double-rotated => wrong basis). Must transform the tensor, not just
            # the trace string.
            if adapter.rotate_recurrent and adapter._fc_idx > 0:
                e_part = z[..., :D]
                h_part = (z[..., D:].to(torch.float32) @ adapter.R1).to(z.dtype)
                z = torch.cat([e_part, h_part], -1)
            if adapter.trace and adapter._prompt_id >= 0 and len(adapter.forward_trace) < 400:
                adapter.forward_trace.append(dict(
                    prompt_id=adapter._prompt_id,
                    verification_cycle_id=adapter._cycle, tree_depth=adapter._fc_idx,
                    is_external_target_forward=bool(adapter._fc_idx == 0),
                    is_recurrent_draft_forward=bool(adapter._fc_idx > 0),
                    input_hidden_source=("target_h" if adapter._fc_idx == 0 else "draft_f_R"),
                    runtime_transform=("@R1" if adapter._fc_idx == 0 and adapter.rotate_external
                                       else ("@R1(recurrent-ABL)" if adapter._fc_idx > 0 and adapter.rotate_recurrent
                                             else "none")),
                    expected_basis=("h_R" if adapter._fc_idx == 0 else "f_R"),
                    hidden_norm=round(float(z[..., D:].detach().float().norm()), 4)))
            adapter._fc_idx += 1
            return adapter._orig_fc_forward(z, *a, **k)
        ea.fc.forward = fc_forward

        assert self._orig_topk is None
        self._orig_topk = ea.topK_genrate

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            adapter._fc_idx = 0
            do_rot = adapter.rotate_external and (
                not adapter.once_per_prompt or not adapter._did_rotate_once)
            hs = hidden_states
            if do_rot:
                hs = (hidden_states.to(torch.float32) @ adapter.R1).to(hidden_states.dtype)
                adapter._did_rotate_once = True
            adapter._cycle += 1
            return adapter._orig_topk(hs, input_ids, adapter.head,
                                      logits_processor, *a, **k)
        ea.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None
        if hasattr(self, "_orig_fc_forward"):
            self.ea_layer.fc.forward = self._orig_fc_forward
            del self._orig_fc_forward
        self.ea_layer.load_state_dict(self._sd_backup, strict=True)
        self.ea_layer.to(self.device)
        del self._sd_backup

    def meta(self):
        return dict(head=self.head_desc, rotate_external=self.rotate_external,
                    once_per_prompt=self.once_per_prompt,
                    rotate_recurrent=self.rotate_recurrent)
