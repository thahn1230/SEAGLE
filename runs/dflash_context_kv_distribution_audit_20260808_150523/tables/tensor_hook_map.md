# Tensor hook map (§1) — verified against source, 2026-08-08

Repo: /home/thahn1230/dflash_workspace/dflash @ a134433,
branch exp/dflash-context-kv-distribution-audit.
"L" = line numbers at this commit.

## Target selected hidden (H_1, H_8, H_15, H_22, H_29)

- Producer: HF `LlamaModel.forward(output_hidden_states=True)` on the
  (rotated or stock) target built by `seagle_port/spinquant_target.py::
  build_target`. `output.hidden_states[l + 1]` = residual-stream OUTPUT of
  decoder layer l (index offset +1 because hidden_states[0] is the
  embedding output) — convention verified in Gate B2 of the previous study
  (relerr ≤ 0.023 vs H·R1 for every l ≤ 31).
- BEFORE R_T: hidden_states of the STOCK (fp16) target (C0).
- AFTER R_T: hidden_states of the ROTATED target (C1-C4). The rotation is
  weight-folded (`rotate_target`), so "after R_T" is simply the rotated
  model's residual stream — there is no runtime rotation op to hook.
- Selection + concat: `dflash/model.py::extract_context_feature` L39-45:
  `torch.cat([hidden_states[l+1] for l in target_layer_ids], dim=-1)`.

## W_c input

- Exact tensor fed to W_c = the concat above, THEN the arm's
  `ctx_transform` (only for explicit/rotate/MP3 arms;
  `seagle_port/generate.py::dflash_generate_hooked` applies it to
  `target_hidden` immediately after extract, L~96-99 and L~137-140).
  For the C1-C4 arms used here (folded interface) ctx_transform is None —
  the rotated concat goes straight in. Boundaries: source i occupies
  columns [4096*i, 4096*(i+1)), order (1, 8, 15, 22, 29).
- A4 quantization of the W_c input happens ONLY in quantized-draft arms,
  inside `seagle_port/draft_quant.py::QLinear.forward` (fc replaced by
  QLinear): `_aq(x)` reshapes to (-1, 20480) and calls SpinQuant
  `ActQuantizer` per row. The tensor quantized is EXACTLY the tensor
  entering the linear (verified: no intermediate ops between the module
  boundary and `_aq`). With `--fc-p2`, `branch_dims=[4096]*5` slices the
  last dim and quantizes each source block independently (QLinear.forward
  branch loop).

## Fusion output / H_t

- Z_t (W_c output BEFORE RMSNorm): `draft.fc(target_hidden)` —
  `dflash/model.py::DFlashDraftModel.forward` L334 (plain path) and
  `seagle_port/rc.py::RCDraft.forward` (`b.fc(target_hidden)`).
- H_t (AFTER hidden_norm, BEFORE R_C): same lines,
  `hidden_norm(fc(...))`. hidden_norm = Qwen3RMSNorm(4096) with γ.
- H_t is computed ONCE per draft forward and shared by ALL 5 layers with
  NO per-layer norm on the ctx branch (model.py L334→L336-346; rc.py
  RCDraft.forward loop). Verified fact from previous study (§15).

## Context rotation R_C

- Hook: `seagle_port/rc.py::RCContextKV.forward` — per layer:
  `x = act_fake_ste(H_t.float() @ R_C, a_bits)`;
  `k_ctx = x @ quant(W_k @ R_C)^T`, same x for V. The SAME H_t and SAME
  R_C are used in every layer (module list ctx_kv, R from
  `RCDraft.rc_matrix()`).
- R_C OFF (C2) = plain QLinear path: `rc.py` not involved;
  `k_ctx = k_proj(target_hidden)` with k_proj = QLinear
  (dflash/model.py L226-229 pattern reproduced in
  `rc.py::_layer_forward_ctxkv` for the RC arms).
- NOTE (two quantizer implementations, both deployed):
  * C2 (QLinear): act = SpinQuant ActQuantizer (per-token asym, bits 4,
    groupsize -1, clip 1.0); weight = SpinQuant WeightQuantizer
    (per-channel sym RTN + MSE clip search).
  * C3/C4 (RCContextKV views): act = `rc.py::act_fake_ste` (per-token
    asym STE, same formula, no clip search); weight =
    `rc.py::rtn_sym_perchannel` (per-channel sym RTN, NO MSE clip).
  The audit must measure each branch with ITS OWN deployed quantizer and
  report both policies explicitly (a same-quantizer control column is also
  produced so the R_C effect is not confounded by the clip-search
  difference).

## Draft-layer K/V inputs

- Context branch (per layer i):
  * C2: `k_ctx = k_proj(target_hidden)`, `v_ctx = v_proj(target_hidden)`
    — dflash/model.py L226-229; k_proj/v_proj are QLinear in quantized
    arms; ctx and noise are SEPARATE calls, so per-token qparams are
    computed independently per call (QLinear._aq has no state).
  * C3/C4: `RCContextKV.forward(H_t, R)` (rc.py) — K and V consume the
    IDENTICAL rotated+quantized activation tensor x (single `_aq`-style
    call shared by both projections). Equality is by construction (same
    Python object); recorded as `kv_ctx_input_identical=true` for RC arms
    and verified at runtime by tensor `data_ptr` in the capture.
- Draft-side branch (per layer i):
  `hs = layer.input_layernorm(hidden_states)` then `k_noise = k_proj(hs)`,
  `v_noise = v_proj(hs)` — dflash/model.py L221-229 /
  rc.py::_layer_forward_ctxkv. K and V inputs are the same tensor `hs`
  (per-layer γ_i RMSNorm of the residual), quantized independently per
  QLinear call in C2 (two separate `_aq` invocations on the same values).

## Downstream (for §20)

- k_norm (per-head-dim RMSNorm) applied AFTER ctx/noise concat:
  model.py L230-232 / rc.py; RoPE applied to full K (ctx positions get
  their sequence positions): model.py L234-235 (apply_rotary_pos_emb,
  q uses the last q_len slice).
- Attention runs in bf16 (sdpa), V has no norm; ctx K/V persist in the
  draft DynamicCache across cycles (model.py L120/L139 crop pattern).
