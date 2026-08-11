# FIDI exact tensor / intervention-point map (from code audit @310462c)

All references are to dflash/model.py (PyTorch deployed path) unless noted.

## Four stages — exact capture locations

| stage | tensor | where (file:line) | notes |
|---|---|---|---|
| S1 | H_i, i∈{1,8,15,22,29} | `extract_context_feature` model.py:39-45; `hidden_states[l+1]` (offset+1: idx0=embeddings) | residual-stream OUTPUT of target layer l; concat order (1,8,15,22,29), cols [4096i,4096(i+1)) |
| S1' | concat [B,S,20480] | model.py:45 / prefill :99 / per-cycle :143 (sliced to accepted+1) | this exact tensor enters fc |
| S2 | W_c = `fc.weight` [4096,20480] | module built model.py:317 | per-source blocks 4096-wide |
| S3 | Z_t = fc(concat) (pre-norm) | inside model.py:334 (`self.fc(target_hidden)`) | capture requires hook on fc output or reimplementation |
| S3 | H_t = hidden_norm(Z_t) | model.py:334 | computed ONCE, shared by all 5 layers; ctx branch has NO per-layer norm |
| S4 | stored ctx K | k_ctx = k_proj(H_t) :226 → cat[ctx,noise] :230 → k_norm :232 → RoPE :234-235 → cache write :238 | stored POST-k_norm POST-RoPE, bf16, [1,8,S,128] |
| S4 | stored ctx V | v_ctx = v_proj(H_t) :228 → cat :231 → cache write :238 | stored RAW (no norm, no RoPE) |
| S4 | block (draft-token) K/V | same write :238; k_noise/v_noise from input_layernorm'd residual :227/:229 | TRANSIENT: `past_key_values_draft.crop(start)` model.py:120 discards them every cycle. Only ctx K/V persist. |

Cache mechanics: single write point per layer per forward (model.py:236-238,
`DynamicCache.update`); ctx entries persist & grow to full seq length; write is
gated on cache!=None, not use_cache. Draft queries never cached.
Under RotQuantDraft (deployed VSQ eval) there are NO nn.Linear modules —
weights are raw buffers, forward inlined (vsq_draft_rot.py:228-287); capture
there = monkeypatch `_aq`/`_wq` or edit forward. Under stock/RCDraft paths,
k_proj/v_proj pre-hooks fire ctx-first-then-noise (dkva_capture.py:476-494).

## Draft weights to audit (state_dict names, z-lab d3af30d)

Global: `fc.weight` [4096,20480], `hidden_norm.weight` [4096], `norm.weight` [4096].
Per layer i∈0..4 (`layers.{i}.`):
`self_attn.{q_proj[4096,4096], k_proj[1024,4096], v_proj[1024,4096], o_proj[4096,4096]}.weight`,
`self_attn.{q_norm,k_norm}.weight` [128], `mlp.{gate_proj[12288,4096], up_proj[12288,4096],
down_proj[4096,12288]}.weight`, `{input_layernorm, post_attention_layernorm}.weight` [4096].
Shared with target (stats only, not draft-owned): embed_tokens / lm_head [128256,4096]×2.
8 KV heads (GQA 4), head_dim 128, 32 Q heads, intermediate 12288, block_size 10, bidirectional attention (is_causal=False, no mask).

## Rotation configs (paired replay of the SAME fp16 trajectory: cyc__M0cyc__{mtbench,gsm8k})

- R0: stock fp16 + RTN references (target w4a4_norot for quant refs)
- R1: target vanilla SQ only — R.bin s1 (sha f433a88b); mandatory folds only (fold_wc is
  mathematically mandatory at the interface: cached target hidden arrives in rotated basis)
- R2: R1 + draft R1_D/R2_D (R1D_s1r1.pt.best, val CE 4.354); no R_C
- R3: R2 + P2 + R_C=R1_T (deployed M5 best) — ctx-specific folded K/V views
- R4 (diag): learned R_C = runs/dflash_seagle_transfer_20260807_180238/rotations/RC_L0.pt

NOTE: DKVA capture artifacts used the OLD w16a4kv16 rotation — NOT reusable for
R1/R2/R3-labeled FIDI results. Reusable from DKVA: C0 (rotation-free) reference
methodology + accumulator classes only. All FIDI captures re-run with s1.

## Gauge-invariance pre-analysis (I5)

Per-token absmax/minmax A4 + per-row absmax W4 ⇒ global scalar s (H_t·s, W_K/s, W_V/s)
is bit-exact gauge (codes identical). CLOSE I5 after one empirical verification
(s∈{0.25,0.5,1,2,4}, tables/scale_invariance_test.csv). Same argument: any per-ROW weight
rescale is gauge for per-row sym W4. NOT gauge (legitimate candidates): I6 branch-wise scalar
(changes intra-token relative magnitude under one shared per-token scale — note deployed
M5 uses P2 per-branch act scales at fc, under WHICH per-branch scalars become gauge again →
MP3-on-top-of-P2 is redundant, explains V_M4a+MP3 non-stacking), I7 per-channel diagonal S
at H_t→K/V (SmoothQuant-like, fold W_K S^{-1}).
