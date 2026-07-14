# 01 — Rotated-EAGLE-1 Architecture (SpinQuant W4A4 x EAGLE-1)

All operator placements below were **verified against the actual code** (not assumed
from the paper diagram); citations are file:line in `third_party/`. Where the classic
paper-style description differs from the code, the code wins and the difference is
called out. Evidence trails: `docs/appendix/A1_*.md`, `A2_*.md`.

Conventions: row-vector math, activations right-multiply weights (`y = x @ W`,
`W:[in,out]`). PyTorch `nn.Linear` stores `weight:[out,in]` and computes
`y = x @ weight.T`; both forms are given where it matters. `D` = hidden_size (4096),
`d_h` = head_dim (128), `I` = intermediate_size (11008), `V` = vocab (32000).

---

## 0. System architecture map

```
                        ┌─────────────────────────────────────────────────────────┐
                        │   TARGET: LLaMA-2-7B-chat, SpinQuant-rotated, W4A4(KV4)  │
                        │   (fake-quant: QDQ ops, FP16 matmuls)                    │
  tokens ──────────────▶│ embed* (zero-centered, @R1) ──▶ 32x [rotated block]      │
                        │  ──▶ final RMSNorm (weight=1) ──▶ ĥ  ── lm_head*(@R1) ──▶│──▶ logits (verify)
                        └───────────────────────────────────────┬─────────────────┘
                                    residual stream is in R1 basis │ ĥ = (h ⊘ γ_f) @ R1
                                                                   ▼
                                              ┌──────────────────────────────────┐
                                              │  INTERFACE ADAPTER (this project)│
                                              │  A: h = (ĥ @ R1ᵀ) ⊙ γ_f (explicit)│
                                              │  B: fold map into draft fc W_h   │
                                              │  C: retrain draft on ĥ directly  │
                                              └────────────────┬─────────────────┘
                                                               ▼
   token ids ──▶ draft embed e ─┐            ┌──────────────────────────────────┐
   (draft's own FROZEN copy of  ├─ cat([e,h])│  EAGLE-1 DRAFT (FP16, unquantized)│
   the ORIGINAL, unrotated      │  [B,S,2D]  │  fc: 2D→D (bias) ──▶ 1x LlamaLayer│──▶ predicted feature f
   embedding — cnets.py:466-495)┘            │  (no final norm)                 │      │
                                              └──────────────────────────────────┘      │
                                                               ┌────────────────────────┤
                                                               ▼                        ▼
                                                original lm_head copy W_lm      recycle f as next-step
                                                (FP16, kept unrotated) ──▶ draft   draft input (tree)
                                                token distributions (top-k tree)  cnets.py:806-817
```

Key verified facts the map encodes:
- EAGLE-1 draft input is `fc(cat([e, h]))` — **embedding FIRST, hidden SECOND**, no
  activation (`cnets.py:492,593`). `h` is the target's **post-final-RMSNorm** top-layer
  feature (`ea_model.py:122-132` uses `outputs[0]`; ge_data saves `hidden_states[-1]`).
  It is NOT the second-to-top layer in this implementation.
- The draft has no lm_head; it reuses the target's (`topK_genrate(..., head=...)`,
  `cnets.py:780`). The draft's predicted feature is **never fed back into the target**;
  it is consumed only by (a) the draft itself recursively and (b) the head we pass in.
- SpinQuant's rotated model keeps the **entire residual stream in the R1 basis at
  runtime** (all rotations except R3/R4 are fused offline into weights).

---

## 1. SpinQuant rotation flow (verified)

Order of transforms at PTQ time (`eval_utils/main.py:28-29`): `fuse_layer_norms` →
`rotate_model` → wrap linears (`ActQuantWrapper`) + online R4 → GPTQ weights → K/V
quant config + R3 wrapper.

**Norm fusion first** (`utils/fuse_norm_utils.py`): every RMSNorm scale γ is folded
into the following linear(s) (`linear.weight = W ⊙ γ` column-wise, :26); the final
norm's γ_f is folded into `lm_head`; all norm weights are then set to 1, and embedding
rows are zero-centered (:43-45). A weight-1 RMSNorm (`x/rms(x)`) commutes with any
orthogonal R1 because `rms(x@R1) = rms(x)` — this is what makes the residual rotation
exact.

**Residual stream global rotation R1** (`D x D`, LEARNED via Cayley SGD, fused offline,
`eval_utils/rotation_utils.py:122-147`):
- `embed_tokens.weight ← W_E @ R1` → embeddings emitted in R1 basis (`X_R = X @ R1`)
- input-side of every consumer: `q/k/v/up/gate_proj.weight ← W @ R1` (PyTorch
  `[out,in]` right-mult = math-form `R1ᵀ · W`, i.e. the sketch's "R1⁻¹ → W")
- output-side of every producer: `o_proj/down_proj.weight ← R1ᵀ @ W` → block outputs
  re-enter the residual already in the R1 basis
- `lm_head.weight ← W @ R1` → the inverse residual rotation is **fused into the output
  head** (no explicit un-rotation op exists at runtime)

**Attention block** (input `[B,S,D]` in R1 basis):
- Q path: `(R1⁻¹ fused → W_q) → RoPE → R3` — R3 is a **fixed Hadamard on head_dim,
  applied ONLINE post-RoPE** via a monkeypatched wrapper around `apply_rotary_pos_emb`
  (`rotation_utils.py:150-228`, `monkeypatch.py`). Active only when `k_bits < 16`.
- K path: same as Q; the **K cache is fake-quantized immediately after R3**, per-head,
  asymmetric (`rotation_utils.py:186-202`). Attention scores are preserved because Q
  and K share the same orthogonal R3: `(QR3)(KR3)ᵀ = QKᵀ`.
- V path: `(R1⁻¹ fused → W_v) → R2` — R2 (`d_h x d_h`, LEARNED, per-layer) is fused
  into each head's output slice of `v_proj`; the **V cache is quantized in the R2
  basis** at the v_proj output quantizer (`eval_utils/main.py:110-117`).
- Output path: `(R2⁻¹ fused → W_o → R1 fused)` (`rotate_ov_proj`, o_proj per-head
  input dims by R2ᵀ, output side by R1ᵀ).

**MLP block**:
- Up: `(R1⁻¹ fused → W_up)`; Gate: `(R1⁻¹ fused → W_gate → SiLU)`
- The elementwise gated product is rotated by **R4 = fixed Hadamard over I=11008,
  applied ONLINE** (`ActQuantWrapper.online_full_had`, `quant_utils.py:249-255`;
  11008 = 172·64 Kronecker factorization). **The down_proj input is quantized AFTER
  the Hadamard** (`quant_utils.py:283-286`).
- Down: `(R4⁻¹ fused into W_down input dim → W_down → R1 fused on output)`
  (`rotation_utils.py:92-103`).

**What is quantized where (W4A4KV4, `scripts/2_eval_ptq.sh m 4 4 4`)**: weights GPTQ
4-bit per-channel (default; `--w_rtn` for RTN); inputs of every wrapped Linear
per-token dynamic asymmetric 4-bit; K post-RoPE+R3 per-head; V post-R2 per-head;
`lm_head` input stays FP16. The residual additions and norms themselves are FP16 —
"A4" means linear inputs, not the raw residual tensor. **All of it is
quantize-dequantize; matmuls stay FP16 (no INT4 kernels — see audit doc §4).**

---

## 2. The EAGLE-1 interface problem

EAGLE-1 drafts from target-model features: the concat input

```
z = cat([e, h])          e: draft's own token embedding  [B,S,D]
                         h: target post-final-norm hidden [B,S,D]
f = z @ W_fc + b         W_fc: [2D, D] (math form; nn.Linear weight is its transpose)
```

Split `W_fc = [W_e ; W_h]` (rows 0:D multiply `e`, rows D:2D multiply `h` — note the
order is **e-block first** because the code concatenates `(inputs_embeds,
hidden_states)`; the planning sketch's `concat(h, e)` has it reversed).

The rotated target does not expose `h`. Its final-norm output is

```
ĥ = (h_pre @ R1) / rms(h_pre) = (h ⊘ γ_f) @ R1
```

because γ_f (the original final-norm scale) was **fused into lm_head, not left in the
norm**. So the naive `h = ĥ @ R1ᵀ` is WRONG — the correct inverse map is

```
h = (ĥ @ R1ᵀ) ⊙ γ_f        (γ_f must be stashed BEFORE fuse_layer_norms sets it to 1)
```

In general `cat([e, ĥ]) @ W_fc ≠ cat([e, h]) @ W_fc`, so with an unmodified draft the
interface breaks. Exact full-precision preservation requires conjugating only the
h-block (the draft's `e` comes from its **own frozen copy of the original embedding**,
`cnets.py:466-495` — it is never rotated, so `W_e` stays unchanged):

```
W_h_rot = R1ᵀ · diag(γ_f) · W_h                     (math form [D,D])
fc.weight[:, D:2D] ← fc.weight[:, D:2D] @ diag(γ_f) @ R1   (nn.Linear form)
```

Draft output side: the planning sketch anticipated `W_out_rot = W_out @ R1` for a
draft-output projection. **Not needed in the actual EAGLE-1 code path**: the predicted
feature never re-enters the target; it feeds only the draft's own recursion (basis-
consistent by construction) and the lm_head handle passed into `topK_genrate`. We pass
a preserved FP16 copy of the ORIGINAL `W_lm` (0.26 GB), because the rotated model's
lm_head expects R1-basis inputs while the draft's features live in the original basis
(variants A/B). Equivalent alternative: rotate the draft interface fully and use the
rotated head — deferred; the original-head copy is simpler and exact.

Quantization caveat: all conjugations above are exact **only in full precision**. Under
W4A4 the captured ĥ carries quantization noise, so draft feature-prediction quality and
acceptance rate degrade — that degradation (and how much learned rotations mitigate it)
is the experiment.

---

## 3. Integration variants

### Variant A — interface-preserving unrotation (correctness baseline)
- Target runs rotated + quantized. At every feature capture point (prefill AND each
  post-verification step — `ea_model.py:145` and `utils.py:466-468`), compute
  `h = (ĥ @ R1ᵀ) ⊙ γ_f` explicitly, then run the stock EAGLE-1 draft + original-head
  copy.
- Safest: bit-identical to stock EAGLE-1 when quantization is disabled (up to the
  embedding zero-centering and fp64 fusion roundoff — verify with the FP sanity test
  below). Runtime overhead: one `[B,S,D]@[D,D]` GEMM + elementwise per target forward
  (S ≈ 26 tree tokens after prefill) — negligible vs the 7B forward, but nonzero.

### Variant B — draft-conjugated rotation (production path)
- Keep ĥ as-is; pre-fold the inverse map into the draft: patch `fc.weight`'s h-block
  once at load time (formula above). Zero runtime overhead.
- In full precision, A and B compute the SAME function (identical up to float
  associativity) — B is not a different experiment, it is the fused implementation of
  A. Ship B after A passes equivalence tests; keep A as the reference oracle.

### Variant C — rotated-interface draft retraining (acceptance recovery)
- Freeze the quantized rotated target and its R1/R2. Regenerate EAGLE training data by
  running the QUANTIZED target (`output_hidden_states` analog → store ĥ), then train
  only the draft (FP16/BF16) to consume ĥ directly — initialize from Variant-B
  conjugated weights so training starts at the FP-equivalent point.
- Losses as in `eagle/train/main.py`: SmoothL1 feature regression toward the quantized
  target's next-step ĥ, plus soft-CE against the **quantized target's** logits (that
  is the distribution verification actually samples from — matching it is what drives
  acceptance). Optional: acceptance-aware calibration (weight tokens by tree depth).
- This is the only variant that can recover acceptance lost to quantization noise,
  because the draft learns the noised feature dynamics.

Run order: A (validate) → B (measure) → C (improve). Report all three against
(i) FP16 EAGLE-1 baseline, (ii) quantized target WITHOUT speculative decoding.

---

## 4. Engineering plan of record (risks called out)

1. **One model object, two codebases**: EAGLE's tree decoding requires its vendored
   `modeling_llama_kv.py` (preallocated KVCache + `tree_mask` hook, transformers-4.31
   era); SpinQuant's runtime is a different vendored LLaMA (4.44 era). Plan: apply
   SpinQuant's *weight-level* pipeline (fuse_layer_norms → rotate_model → add_actquant
   + online R4 → GPTQ → K/V quant + R3 wrapper) **onto EAGLE's vendored model**.
   Module paths (`model.layers[i].self_attn.q_proj`...) line up. Two porting risks:
   (a) the R3 monkeypatch wraps `apply_rotary_pos_emb` — EAGLE's vendored copy uses the
   old 5-arg signature `(q,k,cos,sin,position_ids)` vs SpinQuant's 4.44-era call; the
   wrapper must be re-targeted and signature-adapted. (b) GPTQ/eval utils assume
   SpinQuant's module layout — use them at the weight level only.
2. **Draft embedding source**: when building the pipeline, the draft's `embed_tokens`
   copy must be loaded from the ORIGINAL checkpoint, never from the fused/rotated
   in-memory model (which is zero-centered and rotated). Same for the preserved
   original `W_lm` copy and γ_f stash.
3. **FP sanity tests before any quantized run** (gate for all later work):
   - rotated-FP16 target vs original target: max |Δlogits| small (limited only by
     embedding zero-centering + fp64 fusion roundoff) on a fixed prompt set;
   - Variant A FP16 (rotate → unrotate → stock draft) vs stock EAGLE-1: identical
     accepted tokens / acceptance lengths on greedy MT-bench samples;
   - Variant B vs Variant A: max |Δ draft logits| ~ float-roundoff.
4. **KV4 interaction**: K is quantized inside the R3 wrapper, V at v_proj output —
   both land in the target's (EAGLE-managed, preallocated) KV cache as dequantized
   FP16 values; the draft's own KV cache stays FP16 (draft is unquantized). No extra
   work expected beyond the R3 wrapper port, hence W4A4KV4 is "supported if the R3
   port works" — verify with PPL parity vs SpinQuant's own runtime on the same R.bin.
5. **Metrics validity**: fake quant → absolute tokens/s is not a deployment number.
   Primary metrics: wikitext-2 PPL (target), acceptance length/rate (draft), relative
   speculative speedup on the SAME quantized target, plus an analytical projection of
   real-kernel speedup from acceptance stats. See audit doc §4.
