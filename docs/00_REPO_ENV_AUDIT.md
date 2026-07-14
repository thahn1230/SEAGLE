# 00 — Repository & Environment Audit

Planning-stage audit (2026-07-02). Every claim below was extracted from the code by
one inspection agent and re-verified by an independent adversarial agent against
file:line evidence. Full evidence trails: `docs/appendix/A1_EAGLE1_CODE_AUDIT.md`,
`A2_SPINQUANT_CODE_AUDIT.md`, `A3_MODEL_INTERSECTION_AUDIT.md`.

## 1. CUDA / PyTorch / GPU summary

| item | value |
|---|---|
| GPUs | 8x NVIDIA GeForce RTX 4090, 24 GB each (23.5 GiB usable), all idle |
| Driver / CUDA | driver 570.86.10, CUDA 12.8 (driver); PyTorch built for cu124 |
| Python | 3.12.4 (Anaconda) |
| torch | 2.6.0+cu124, `cuda.is_available()=True`, 8 devices visible |
| transformers | 4.51.3 installed |
| accelerate / datasets | 0.26.0 / 3.5.0 |
| fastchat | 0.2.31 (needed by EAGLE eval; present) |
| HF auth | token present (`~/.cache/huggingface/token`) — required for gated meta-llama |
| missing | `fast_hadamard_transform` (SpinQuant online R3/R4 — must `pip install fast-hadamard-transform`), `lm_eval` (optional) |

Version pins conflict three ways: EAGLE v1 pins `transformers==4.36.2, torch==2.0.1`;
SpinQuant pins `transformers==4.44.2`; installed is 4.51.3.
Empirically, all EAGLE v1 model modules (`eagle.model.{cnets,ea_model,utils,kv_cache,
modeling_llama_kv,modeling_mixtral_kv}`) **import cleanly under 4.51.3** because they
are self-contained vendored copies of transformers-4.31-era code. SpinQuant's vendored
`eval_utils/modeling_llama.py` imports private HF APIs
(`modeling_flash_attention_utils._flash_attention_forward`,
`modeling_rope_utils.ROPE_INIT_FUNCTIONS`) from the ~4.44 era — import under 4.51.3 is
**unverified** (worker task E-1). Plan: try the shared installed env first; fall back to
a dedicated SpinQuant venv with `transformers==4.44.2` if imports or Cache APIs break.
`torch 2.6` note: `torch.load` defaults `weights_only=True`; SpinQuant's `R.bin` and
EAGLE's `pytorch_model.bin` are plain tensor dicts and should load, but verify.

## 2. EAGLE: branch and relevant files for EAGLE-1

- Clone: `third_party/EAGLE`, **checked out on branch `v1`** (HEAD `4a9cf3a`). `main` is
  the EAGLE-2/EAGLE-3 runtime; `v1` is the EAGLE-1 snapshot (per main README, "v1" is
  the EAGLE-1 branch). Do not use `main` for EAGLE-1 semantics (dynamic-tree default,
  different code layout).
- Draft network: `eagle/model/cnets.py` — `class Model` (line 454).
  `self.fc = nn.Linear(2*hidden, hidden, bias=True)` (:492); forward combine is
  `fc(cat([inputs_embeds, hidden_states], dim=-1))` (:593) — **embedding first,
  hidden second, no activation** (activated variant commented out at :591). After the
  fc: one vendored `LlamaDecoderLayer` (`num_hidden_layers=1` in shipped configs;
  layer 0 has no input_layernorm; **no final RMSNorm** — raw feature out).
- Hidden-state source: target's **top-layer output AFTER the final RMSNorm**
  (`last_hidden_state`). Inference: `ea_model.py:122-132` (`outputs[0]` of
  `base_model.model`, post `self.norm` at `modeling_llama_kv.py:1074`). Training data:
  `ge_data_all_*.py` saves `hidden_states[-1]` (also post-norm). NOT second-to-top.
- Draft output: predicted next hidden feature, scored by the **target's lm_head**
  (head reuse — `topK_genrate(..., head=base_model.lm_head)`, cnets.py:780); the
  predicted feature is recycled autoregressively for multi-step tree drafting
  (cnets.py:806-817) with the draft's own FP KV cache.
- Tree/verify: static tree `mc_sim_7b_63` (choices.py; 25 paths, depth<=5, top_k=10);
  acceptance in `evaluate_posterior` (utils.py:320-412); `new_token += accept_length+1`
  (utils.py:470).
- Training: `eagle/train/main.py` (accelerate, bf16) — SmoothL1 feature loss (v_w=1.0)
  + soft-CE logit loss through frozen target head (p_w=0.1), uniform noise std 0.2 on
  stored hiddens. Data gen: `eagle/ge_data/ge_data_all_{vicuna,llama2chat}.py`
  (ShareGPT, 68k samples, `allocation.py` shards across GPUs; hardcoded model paths to edit).
- Evaluation: `eagle/evaluation/gen_ea_answer_llama2chat.py` (EAGLE) vs
  `gen_baseline_answer_llama2chat.py` (vanilla), MT-bench; `speed.py` computes the
  speedup ratio (hardcoded paths must be edited). Vicuna and Mixtral variants exist.
- **Batch size: the primary `eagle/model` path is batch=1 only** (explicit asserts,
  ea_model.py:269,373). Batch>1 lives in the separate `eagle/modelbsne1/` package
  (left padding). The batch-sweep deliverable must build on `modelbsne1`.
- v1 limitation: `modeling_llama_kv.py:227` hardcodes RoPE base 10000 (ignores
  `rope_theta`) → **LLaMA-3 is genuinely unsupported on v1** (GQA itself IS supported).

## 3. SpinQuant: scripts and relevant files

- Rotation optimization: `optimize_rotation.py` + `train_utils/`.
  Trains **R1 (hidden_size, global) + per-layer R2 (head_dim, per-head V/O)** only, via
  **Cayley SGD on the Stiefel manifold** (`SGDG`, train_utils/optimizer.py), 100 steps,
  lr 1.5, batch 1 x seqlen 2048, wikitext-2 next-token CE, with **fake-quantized (STE)
  forward at the target bit-widths in the loop**. R3/R4 fixed. Saves
  `<out>/R.bin = {"R1": ..., "model.layers.{i}.self_attn.R2": ...}`.
  Launcher: `scripts/10_optimize_rotation.sh <model> <w> <a> <kv>` → torchrun
  `--nproc_per_node=8` **DDP (memory is NOT sharded — full model + backprop per GPU;
  our 8x24GB helps throughput only)**. Estimated ~18-22 GB/GPU for 7B bf16 with grad
  checkpointing — plausible for a 4090 but **unproven** (verifier flagged the number as
  extrapolation; fp64 transient weight copies in `train_utils/quant_linear.py` are the
  wildcard). Fallbacks: published pre-optimized rotations (README Google Drive folder;
  base LLaMA-2 7B/13B/70B + LLaMA-3 8B/70B only, no chat variants), FSDP script 11,
  or shorter seqlen. README:72-73: if GPTQ will be used at PTQ, optimize rotations with
  `w_bits 16`.
- PTQ + eval: `ptq.py` via `scripts/2_eval_ptq.sh <model> <w> <a> <kv>`:
  **W4A4 = `4 4 16`, W4A4KV4 = `4 4 4`** — so **both target configs are directly
  supported by the same code path**. Flags: `--w_clip --a_asym --k_asym --v_asym
  --k_groupsize 128 --v_groupsize 128 --rotate --optimized_rotation_path .../R.bin`.
  Weights: **GPTQ by default** (128 wikitext-2 calib samples; `--w_rtn` for RTN).
  Activations: per-token dynamic asymmetric. K-cache: quantized inside the monkeypatched
  `QKRotationWrapper` **after RoPE + online R3 Hadamard**, per-head. V-cache: `v_proj`
  output quantizer, per-head, in the R2 basis. `lm_head` input stays FP16.
  Eval: **wikitext-2 PPL only** (`utils/eval_utils.py`, layer-offloaded); no lm_eval
  harness in-repo. `ptq.py` requires `torchrun` (unconditional NCCL init) but
  `--nproc_per_node=1` — single GPU is fine.
- Rotation placement (all verified in `eval_utils/rotation_utils.py` +
  `utils/fuse_norm_utils.py`): norm scales are folded into adjacent linears first
  (norms become weight-1; embedding rows zero-centered), then R1 fused offline:
  `embed_tokens←W@R1`, `q/k/v/up/gate←W@R1` (input side), `o_proj/down_proj←R1ᵀ@W`
  (output side), `lm_head←W@R1`. R2 fused into v_proj (out) / o_proj (in). R3 = fixed
  Hadamard **online post-RoPE** on Q,K over head_dim (only when k_bits<16; needs
  `fast_hadamard_transform`). R4 = fixed Hadamard **online** on down_proj input over
  intermediate_size, inverse fused into W_down; activation quantized after the Hadamard.
- **Runtime residual basis: after fusion the entire residual stream (all decoder-layer
  inputs/outputs and the final-norm input) is in the R1 basis.** This is the fact the
  whole EAGLE integration hinges on; see `docs/01_ROTATED_EAGLE1_ARCHITECTURE.md`.
- Architectures: **LLaMA-class only** (hardcoded custom `LlamaForCausalLM` +
  `LlamaTokenizerFast`; no Mistral/Qwen/Mixtral). Tied embeddings (LLaMA-3.2) handled by
  untying + cloning lm_head before rotation.

## 4. Can end-to-end speed be read as real W4A4 CUDA speedup?

**No.** Verified: every GPU execution path in SpinQuant is fake quantization —
`ActQuantizer`/`WeightQuantizer` do quantize-dequantize and all matmuls run in
FP16/BF16 (`utils/quant_utils.py`; GPTQ writes dequantized FP weights back). There is
**no INT4/low-bit GEMM kernel anywhere in the repo**; the only real-kernel path is the
ExecuTorch export (W4 group-wise + A8 dynamic) targeting on-device mobile backends, not
CUDA, and not A4.

Consequences for our experiment design:
- Absolute tokens/s of the quantized pipeline is NOT a deployment number; fake-quant ops
  (per-token scale search, online Hadamards, QDQ) make it *slower* than FP16.
- Metrics that ARE meaningful: (a) wikitext-2 PPL of the rotated-quantized target,
  (b) EAGLE acceptance length / acceptance rate under quantization (the core research
  question: does W4A4 noise break feature-level drafting, and do rotations help),
  (c) **relative** speculative speedup = EAGLE-on-quantized-target vs vanilla
  autoregressive-on-the-same-quantized-target (both carry identical fake-quant
  overhead, so the ratio isolates the speculative-decoding algorithmic gain),
  (d) projected real speedup = acceptance-derived analytical model (report-only).
- Any claim of "real W4A4 speedup" would require third-party INT4 kernels (e.g.
  QuaRot/Marlin-style) — out of scope; documented as a limitation.

## 5. Exact unresolved blockers

1. **Gated model access — RESOLVED**: `meta-llama/Llama-2-7b-chat-hf` is gated, but
   the HF token on this machine successfully fetched its config.json on 2026-07-02
   (`scripts/01_discover_model_intersection.py`), so license acceptance is already in
   place for this account. Full weight download still pending (task M-1). Documented
   fallback if weight download fails: non-gated `lmsys/vicuna-7b-v1.3` +
   `yuhuili/EAGLE-Vicuna-7B-v1.3`. Do not bypass gating.
2. **SpinQuant under transformers 4.51.3 unverified**: private-API imports
   (`_flash_attention_forward`, `ROPE_INIT_FUNCTIONS`) are from the 4.44 era; may need
   a dedicated venv with `transformers==4.44.2` (worker task E-1).
3. **`fast_hadamard_transform` not installed** (needed for R3/R4 online ops; CUDA
   build required). Worker task E-2.
4. **Rotation-optimization VRAM on a 24GB 4090 is unproven** (~18-22 GB estimate is an
   extrapolation; fp64 transients could push it over). Mitigations listed in §3.
5. **Two incompatible vendored LLaMA implementations**: EAGLE's tree-decoding runtime
   (`modeling_llama_kv.py`, transformers-4.31-era, preallocated KVCache + tree_mask
   hook, old 5-arg `apply_rotary_pos_emb`) vs SpinQuant's quantized runtime
   (`eval_utils/modeling_llama.py`, 4.44-era, HF Cache API + monkeypatched R3). The
   integration must port SpinQuant's *weight-level* transforms (norm-fuse, rotate,
   GPTQ, ActQuantWrapper, R3 wrapper, KV quant) onto EAGLE's vendored model — module
   names line up (`model.layers[i].self_attn.q_proj`...), but the R3 monkeypatch must
   be re-targeted at EAGLE's old-signature RoPE call. This is the main engineering risk
   (worker tasks I-1..I-3).
6. **`speed.py` / ge_data scripts have hardcoded paths** (`/home/lyh/...`, wandb
   entity) — must be parameterized in our wrappers, not edited in-place in third_party.
7. SpinQuant's published pre-optimized rotations cover **base** models only; for the
   chat target we must optimize R1/R2 ourselves (100 steps, tractable).
