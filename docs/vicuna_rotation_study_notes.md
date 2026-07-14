# Vicuna-7B-v1.3 rotation-interface study — model resolution & interface audit

Date: 2026-07-04. Purpose: reduce the "one model + one draft" generality gap of
`runs/rotstudy_20260704_1507` (Llama-2-7b-chat + EAGLE-llama2-chat-7B).

## 1. Model resolution

| Role | Resolved ID | Evidence |
|---|---|---|
| Target | `lmsys/vicuna-7b-v1.3` | EAGLE README model table (third_party/EAGLE/README.md:112); configs/model_candidates.yaml vetted it as the non-gated fallback pair in the original discovery pass |
| Draft | `yuhuili/EAGLE-Vicuna-7B-v1.3` | Same README table row: official EAGLE-1 weights for Vicuna-7B-v1.3 (0.24B non-embedding) |

Weights staged at `/data/thahn1230/hf_cache/` (root filesystem was at 100%
capacity, 4.1 GB free; `/data` is a separate 1.8 TB disk with 336 GB free).
Snapshot paths are pinned in `configs/vicuna_experiment.yaml`.

- Target snapshot: `236eeeab96f0dc2e463f2bebb7bb49809279c6d6` (pytorch_model
  2 shards, fp16, no safetensors published for v1.3)
- Draft snapshot: `11ed6a1022f89a8047b90af8853f189a31c99caa` (pytorch_model.bin,
  fp32, 367M params incl. 131M embedding = 0.24B non-embedding, matching README)

## 2. Interface audit (assumption-by-assumption vs the Llama-2 pair)

Draft state dict inspected on CPU (2026-07-04); draft config
(`config.json`, cached): LlamaForCausalLM, hidden 4096, 1 hidden layer,
32 heads, vocab 32000, `max_position_embeddings: 2048`.

| # | Assumption | Vicuna draft | Verdict |
|---|---|---|---|
| 1 | Draft input is `fc(concat([e, h]))` | Same code path — the draft is instantiated from the SAME `eagle/model/cnets.py` `Model` class used for Llama-2 (cnets.py:492/593); only weights differ | HOLDS |
| 2 | `e` first, `h` second in the concat | cnets.py code, model-independent | HOLDS |
| 3 | `fc.weight` shape `[4096, 8192]` (+ bias) | `fc.weight (4096, 8192)`, `fc.bias (4096,)` present in checkpoint | HOLDS |
| 4 | Hidden consumed = post-final-RMSNorm top-layer hidden | ea_model.py:122-132 capture site is model-independent; Vicuna-7B-v1.3 is LlamaForCausalLM with a final `model.norm` RMSNorm exactly like Llama-2 | HOLDS |
| 5a | Own frozen embedding | `embed_tokens.weight (32000, 4096)` in draft checkpoint, loaded strict | HOLDS |
| 5b | No independent lm_head; target head passed to `topK_genrate` | No `lm_head` key in draft checkpoint | HOLDS |
| 6 | Tree expansion recycles predicted features through the same fc | cnets.py:806-816, model-independent code | HOLDS |
| — | Draft layer 0 has no `input_layernorm` | Same as Llama-2 draft (only `post_attention_layernorm` in checkpoint) | SAME |

No interface adaptation of the math is required: R1 (4096²), gamma_f
(final-norm scale, 4096), and the fold formula
`fc.weight[:, 4096:8192] <- W_h @ diag(gamma_f) @ R1` carry over unchanged.

## 3. Architecture differences that do NOT touch the interface (documented)

- Vicuna-7B-v1.3 is **LLaMA-1-based**: `max_position_embeddings=2048`
  (vs 4096), rope_theta 10000, no GQA (7B Llama-2 also has none). Prompt +
  160 new tokens stays well under 2048.
- SpinQuant's official script list does not include Vicuna; the architecture
  is LlamaForCausalLM so `fuse_layer_norms` / `rotate_model` apply unchanged,
  but quantized results are NOT comparable to SpinQuant paper tables.
- Intermediate size 11008 = same online-R4 Hadamard decomposition as Llama-2.

## 4. Prompt template

Llama-2 grid used the `[INST] <<SYS>>...` template. Vicuna REQUIRES the
fastchat `vicuna_v1.1` template (EAGLE README: wrong template degrades output
and EAGLE performance). Implemented as
`eagle_bridge.build_vicuna_prompt`: `"{system} USER: {msg} ASSISTANT:"`,
matching `get_conversation_template("vicuna")` used by
`eagle/evaluation/gen_ea_answer_vicuna.py`. Selected via
`model.chat_template: vicuna` in `configs/vicuna_experiment.yaml` (or
`--chat-template vicuna`).

## 5. Rotations

- Random-Hadamard R.bin generated fresh per seed under
  `/data/thahn1230/eagle_spinquant_outputs/rotations_vicuna/` (never shared
  with the Llama-2 R.bin files, although a random Hadamard R1 is
  mathematically model-independent at equal dims).
- **No learned rotation for Vicuna**: the Llama-2 learned-R run (50 steps) was
  already too weak to support claims; training a Vicuna one is out of scope
  per the run plan (only after everything else completes). ppl.csv and
  acceptance shards therefore contain `random_hadamard` only.

## 6. Variant C (retrained draft)

SKIPPED for Vicuna: no retrained Vicuna draft exists and training one
(~hours on wikitext-style data, fp32) is not "already available and cheap".
The Llama-2 result (C = 2.826 < A/B2 = 3.28 under W4A4 at 80 prompts) already
shows retraining at this scale does not beat the frozen draft + corrected
interface; a full-recipe retraining remains future work.

## 7. Statistical plan (mirrors the Llama-2 grid minus learned-R/C)

- Smoke n=4 on GPU 4: stock/vanilla fp16; naive/A/B2/B rotate-only.
  Gates: naive collapses vs stock; A ~ stock; B2 ~ A.
- Main, GPUs 4-7 (one process per GPU, independent conditions):
  - acceptance_main: fp16 stock/vanilla n=80/40; rotate-only naive/A/B2/B n=80;
    W4A16, W4A4, W4A4KV4 naive/A/B2/B/vanilla n=80/40
  - gamma_ablation: naive/A/A_nogamma n=80
  - depth_sweep: stock + A/B2/B, depths 2-5, n=40
  - rotation_component_ablation: quant-OFF r1 / r1r2 / r1r2r3r4 (A) n=40;
    W4A4 none(stock) / r1 / r1r2 / r1r2r3r4 (A) n=40
  - robustness: rotation seeds 1/2 naive/A/B2 n=40; wikitext naive/A n=20
  - validate_rotation_interface: interface JSON + hidden_diagnostics.csv
  - eval_ppl_study: wikitext-2 PPL fp16 / rotate-only / W4A4 / W4A4KV4
- Bootstrap 95% CIs over prompts (summarize_results.py, 5000 resamples).

All quantized rows are fake quantization (QDQ, FP16 matmuls) — acceptance and
same-runtime relative speed only; no deployment speed claims.
