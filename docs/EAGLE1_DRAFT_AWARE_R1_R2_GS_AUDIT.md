# EAGLE-1 draft-aware R1/R2 GS study — R2 contract audit

Date: 2026-08-10. Host: gpusystem (8x RTX 4090). Branch:
`exp/eagle1-draft-aware-r1-r2-gs` @ base commit `a6a71da`.
Executable audit: `scripts/audit_r2_contract.py` (CPU level PASS 10/10;
model level re-run against the rebuilt R1_T — see run dir
`runs/eagle1_draft_aware_r1_r2_gs_20260810_065036/audit/`).

Terminology: GS = Global Scaling (legacy EP3-G), LS = Local Scaling
(legacy EP3-P). R_{1,T} = target SpinQuant residual R1; R_{1,D} =
draft-aware residual R1; R_{2,B} = baseline draft R2; R_{2,D} = learned
draft-aware R2.

## 1. What object is currently used as draft R2 (R_{2,B})?

A **deterministic seed-0 Haar-random orthogonal matrix** — `QR` of a
`randn(128, 128, dtype=float64)` draw with `torch.Generator().manual_seed(0)`
— regenerated on the fly at every adapter install and every trainer
construction. Canonical constructor (single source of truth added in this
study): `src/eagle_spinquant/fake_w4a4_draft.py::baseline_r2()`.
SHA-256 of the fp64 bytes: `145e00cfe8417ed6…` (executable audit,
`provenance_seed0_qr`).

## 2. Provenance classification

- **Random QR (Haar orthogonal): YES.** This is the deployed baseline.
- **Learned SpinQuant R2: NO.** Never loaded from any R.bin.
- **Transferred from target: NO.** The target's SpinQuant optimization
  learns 32 per-layer `[128,128]` R2 matrices stored in
  `R.bin["model.layers.{i}.self_attn.R2"]` and folds them into the TARGET
  only (`study.py:449-454`, `rotation_utils.rotate_ov_proj`). No R2
  crosses the target->draft interface (R2 cancels inside each attention
  block); only R1 crosses it.
- **Regenerated separately: YES** — deterministically, at install time,
  from seed 0; no artifact file exists or is needed.
- Historical footnote: the code comment at the old
  `fake_w4a4_draft.py:123` called it "random Hadamard-128"; the executable
  audit (`not_hadamard`) proves it is a Haar-QR matrix (|elements| range
  0.0000-0.3283 vs the constant 0.0884 of a Hadamard). The TARGET's
  random-mode R2s ARE Hadamard (`spinquant_bridge.py:229-230`) — the two
  must not be conflated. Because R_{2,B} is not the target's R2, this
  study calls it "baseline R2", never "target R2".

## 3. Exact tensor shape

`[128, 128]`, fp64 at fold time (`head_dim = 128`).

## 4. Granularity

**One matrix shared across all 32 heads of the single draft decoder
layer** (`layers.0`; the EAGLE-1 Llama-2-7B-chat draft has
`num_hidden_layers = 1`). Applied block-diagonally per head:
`blkdiag(R2 x 32)` on the 4096-dim head-major layout. There is no
per-head and no per-layer multiplicity in the draft. R_{2,D} therefore
learns ONE `[128,128]` residual generator (granularity preserved).

## 5. Where R2 is folded into v_proj / o_proj

`src/eagle_spinquant/spinquant_draft.py:46-54` (`conjugate_v_o`), called
from `fake_w4a4_draft.build_spinquant_w4a4_draft_state` after the R1
conjugation of the decoder:

```python
v_view = v_w.reshape(32, 128, 4096)                 # [head, hd, in]
v_new  = einsum("ab,hbc->hac", R2, v_view)          # V_h' = R2 @ V_h
o_view = o_w.reshape(4096, 32, 128)                 # [out, head, hd]
o_new  = einsum("ohc,cb->ohb", o_view, R2.t())      # O_h' = O_h @ R2^T
```

## 6. Exact orientation / transposes

Per head h: **V_h' = R2 · V_h** (left-multiply on v_proj's 128 output
rows) and **O_h' = O_h · R2ᵀ** (right-multiply of o_proj's 128 input
columns by R2ᵀ), verified element-exactly by the executable audit
(`fold_equations`, `blockdiag_equiv`). In activation terms the attention
context arrives as `R2 · ctx_h` and o_proj undoes it:
`O_h'(R2 ctx) = O_h R2ᵀ R2 ctx = O_h ctx` — exact gauge cancellation
(`fp_gauge_block`: rel. deviation 1.3e-15 in fp64; also holds for any
orthogonal R2', `fp_gauge_any_orthogonal`).

NOTE: an earlier internal summary wrote "O_h' = O_h @ R2"; the code is
`O_h @ R2ᵀ`. The executable audit caught this and it is corrected here.

Trainer-side replica cast order
(`exact_quantized_rotation_forward.py:transformed_weights`):
`v = cast16(R2 ⊗_head cast16(V ·fp64 R_D))`,
`o = cast16((cast16(R_Dᵀ ·fp64 O)) ⊗_head R2ᵀ)` — R2 conjugation happens
AFTER the R1/R_D conjugation, on the fp16-cast intermediate, in the fold
dtype; bitwise-matched to the runtime by Gate GS-R1/R2-C.

## 7. Quantization boundaries relative to R2

- **v_proj/o_proj weights are W4-quantized AFTER the R2 fold**: the folded
  state dict is loaded, then `FakeW4A4Linear` computes
  `w_fake = W4(fold(W))` (executable audit `w4_after_fold`: quantizing
  before the fold differs by ~2.8 in max element — the boundary is
  unambiguous). Weight policy: SpinQuant `WeightQuantizer(bits=4,
  perchannel=True, sym=True, mse=True)` — per-output-channel symmetric
  RTN with the official MSE clip search.
- **The o_proj input activation IS A4-quantized after the R2 transform**:
  R2 lives folded in v_proj's output rows, so the attention output enters
  o_proj already in the R2 basis; `FakeW4A4Linear.forward` quantizes that
  input per-token-asymmetric (`ActQuantizer(bits=4, groupsize=-1,
  sym=False)`) and then GEMMs (`a4_after_r2_basis`). This is exactly why
  a draft-aware R2_D can matter: it reshapes both the W4 weight
  distributions of v/o and the A4 code distribution before o_proj.

## 8. Is R2 F0 (fully offline-folded)?

**Yes.** `audit_eagle_transform_foldability.py:145-148` classifies
`attention_R2_vo` as F0 ("absorbed into W_v, W_o offline"); the
model-level audit asserts no `[128,128]` buffer exists in any installed
runtime module (`f0_no_runtime_r2`) — the only online draft operators are
the R4 down_proj Hadamard and the PostProjectionR1 GEMM, both R2-free.
R_{2,D} keeps this property by construction: it is folded by the same
`conjugate_v_o` call (`ar_r2_override`), adding **no** inference operator.

## 9. Does the draft have only one relevant AR layer for R2?

**Yes** — one decoder layer (`layers.0`), hence exactly one V/O pair and
one R2. The `REQUIRED` module list and all folds hardcode `layers.0.*`;
the draft config has `num_hidden_layers = 1` (model-level audit
`single_ar_layer`).

## 10. R_{2,D} parameterization adopted (this study)

`src/eagle_spinquant/residual_rotation.py::ResidualR2Rotation`:
`R_{2,D} = R_{2,B} @ C(B)`, `B = W - Wᵀ` skew by construction,
`C(B) = (I - B/2)^{-1}(I + B/2)`, `W = zeros(128,128)` at init so
step 0 reproduces R_{2,B} **bitwise** (unit test
`test_r2d_init_reproduces_baseline_bitwise`; Gate R2-A
`r2a_init_bitwise`). Granularity = baseline (one shared [128,128]).
Runtime deployment consumes the saved fp64 `R2_D` tensor via
`build_spinquant_w4a4_draft_state(..., R2_override=ck["R2_D"])` — the
same matrix the trainer's exact path folds, giving trainer/runtime parity
(Gate R2-C legs in `scripts/check_r1r2_gs_parity.py`).

## 11. Gates covering this contract

| Gate | Script | What it proves |
|---|---|---|
| R2 contract | `scripts/audit_r2_contract.py` | items 1-9 above, executable |
| R2-A FP gauge | `scripts/check_r2_gates.py` | quant OFF: correct V/O refold preserves FP function (vs broken-fold control); init bitwise |
| R2-B quant sensitivity | `scripts/check_r2_gates.py` | W4A4 ON: R2_D moves v/o codes, logits, LK loss; dL/dB > 0 on multiple batches |
| R2-C / GS-R1 parity | `scripts/check_r1r2_gs_parity.py` | trainer==runtime folded weights + K=4 chain, 4 legs (base/R1D/R2D/joint) under GS m=4096^0.42; W_first pinned to R1_T in every leg; R2_D touches only v/o |
| R2-D orthogonality | trainer assert every step + logged every eval | ‖R2_Dᵀ R2_D − I‖ < 1e-3 |
| Param audit | trainer `param_audit__*.json` + weight-freeze hashes | only rotation generators trainable; model weights bit-identical before/after |
