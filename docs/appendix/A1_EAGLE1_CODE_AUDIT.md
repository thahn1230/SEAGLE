# EAGLE-1 (branch v1) Audit — /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE

Branch confirmed: `git branch` shows `* v1` (HEAD 4a9cf3a "filter").

## 1. Draft model input interface (eagle/model/cnets.py)

The draft/autoregression-head network is `class Model(nn.Module)` at `eagle/model/cnets.py:454`.

**fc layer** — `eagle/model/cnets.py:492`:
```python
self.fc=nn.Linear(2*config.hidden_size,config.hidden_size,bias=bias)
```
- `bias` defaults to `True` in the signature `def __init__(self,config,load_emb=False,path=None,bias=True)` (cnets.py:455). At inference load time, `eagle/model/ea_model.py:42-46` reads `bias` from the EAGLE weight's config.json and falls back to `True` if absent: `try: bias=con["bias"] except: bias=True`, then `self.ea_layer = Model(config,bias=bias)`.

**Exact combine in forward()** — `eagle/model/cnets.py:591-593`:
```python
#hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
inputs_embeds=inputs_embeds.to(hidden_states.dtype)
hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
```
- **Concat order: token embedding FIRST, target hidden state SECOND**: `cat([e, h], dim=-1)` → `fc([e; h])`. So the first `hidden_size` columns of `fc.weight` multiply the embedding, the second `hidden_size` columns multiply the target-model hidden state.
- `inputs_embeds = self.embed_tokens(input_ids)` is computed under `torch.no_grad()` (cnets.py:558-559); `embed_tokens` is a frozen copy of the target's embedding (cnets.py:466-495, `param.requires_grad = False` at 494-495).
- **No activation** after fc: the activated variant is commented out at line 591; `self.act=ACT2FN[config.hidden_act]` (cnets.py:493) is defined but unused in forward (train config also sets `"act": "No"`, train/main.py:29).

**Structure after the fc**: yes — full Llama decoder-layer structure. `self.layers = nn.ModuleList([LlamaDecoderLayer(config,index) for index in range(config.num_hidden_layers)])` (cnets.py:491), where `num_hidden_layers: 1` in all shipped train configs (e.g. `eagle/train/vicuna_7B_config.json`). Each `LlamaDecoderLayer` (cnets.py:378-442) has `LlamaAttention` (self-attn with RoPE, q/k/v/o projections, cnets.py:190-326), `LlamaMLP`, and `post_attention_layernorm`; **the first layer (index 0) omits `input_layernorm`** (cnets.py:385-386 and 414-415: `if self.index != 0: hidden_states = self.input_layernorm(hidden_states)`). There is **no final RMSNorm** after the decoder layer — `forward` returns raw `hidden_states` (cnets.py:630-638). A tree mask is injected into the causal mask (`_prepare_decoder_attention_mask`, cnets.py:507-538).

## 2. Hidden state source

EAGLE-1 uses the **top (last) decoder layer output AFTER the final RMSNorm** (i.e. the model's `last_hidden_state`), NOT a second-to-top layer.

- Inference: `eagle/model/ea_model.py:122-132` — `outputs = self.base_model.model(input_ids=..., past_key_values=..., position_ids=...)`; `orig = self.base_model.lm_head(outputs[0])`; `hidden_states = outputs[0].clone()`. `outputs[0]` is `last_hidden_state`.
- In `eagle/model/modeling_llama_kv.py` (copied from transformers v4.31, per header line 1): after the layer loop, `hidden_states = self.norm(hidden_states)` (line 1074) and `return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)` (lines 1087-1092). So the feature is post-final-RMSNorm.
- Training data matches: `eagle/ge_data/ge_data_all_llama2chat.py:185-186` — `outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True); hidden_state_big = outs_big.hidden_states[-1]` (same in ge_data_all_vicuna.py:166-167). In HF, `hidden_states[-1]` is appended after the final norm, so it equals post-norm `last_hidden_state` — consistent with inference.

## 3. Draft output / head reuse

The draft network outputs a **predicted next hidden feature** and reuses the **TARGET model's lm_head** — it has no head of its own (an unused `Vhead` class exists at cnets.py:906-911; training uses a frozen copy of the target head, see Q4).

- Head passed in: `ea_model.py:145` — `ea_logits = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, logits_processor)`; and after each verification round, `eagle/model/utils.py:466-468` — `tree_logits = model.ea_layer.topK_genrate(accept_hidden_state_new, input_ids=..., head=model.base_model.lm_head, ...)`.
- Applied inside `topK_genrate` (cnets.py:762-845): `last_hidden = out_hidden[:, -1]; last_headout = head(last_hidden)` (cnets.py:779-781) and per tree level `last_headout = head(out_hidden[0])` (cnets.py:820). Multi-GPU variant uses a cloned weight: `ea_model.py:57` `self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)` with `F.linear(last_hidden,self.headweight)` (cnets.py:787).
- **Feature recycling for multi-step drafting** (cnets.py:806-817): the draft's own output `out_hidden` becomes the next step's input hidden state — `hidden_states = out_hidden[:, -1:]` (first level) or `hidden_states = out_hidden`, expanded per tree branching via `repeat_hidden` (cnets.py:811) with `self.tree_mask` and shifted `position_ids`, then `out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)` — using the draft's own KV cache (`self.stable_kv`, cnets.py:773-778).

## 4. Training path (v1 branch)

- Trainer: `eagle/train/main.py` (accelerate, bf16 mixed precision). Command per README.md: `accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir [data] --cpdir [ckpts] --configpath [config]`.
- **Losses** (`eagle/train/main.py:324, 355-367`):
  - Feature regression: `criterion = nn.SmoothL1Loss(reduction="none")` (line 324); `vloss = criterion(predict, data["target"])` (line 365) — smooth-L1 between predicted feature and target model's next-position hidden state (`target = hidden_state[:, 1:, :]` shift, main.py:162-168).
  - Logit loss: soft cross-entropy (equivalent to CE against target softmax; KL up to constant): `target_p = Softmax(head(data["target"]))` (frozen target lm_head, lines 75-98, 356-359); `out_logp = LogSoftmax(head(predict))` (360-361); `ploss = -sum(loss_mask * target_p*out_logp)/loss_mask.sum()` (363-364).
  - Combined: `loss = v_w*vloss + p_w*ploss` with `"v_w": 1.0, "p_w": 0.1` (main.py:24-25, 367).
  - Data augmentation: uniform noise on stored hidden states (`"data_noise": True, "noise": "uniform", "std": 0.2`, main.py:30-33; `AddUniformNoise`, main.py:123-132).
- **Data generation**: `eagle/ge_data/allocation.py` shards 68,000 ShareGPT samples (`e = 68000 - 1`) across GPUs, spawning `ge_data_all_vicuna.py` (hardcoded in the command string). `ge_data_all_vicuna.py` / `ge_data_all_llama2chat.py` load ShareGPT (`ShareGPT_V4.3_unfiltered_cleaned_split.json`, line 46), run the target with `output_hidden_states=True`, and save `{input_ids, hidden_state=hidden_states[-1], loss_mask}` per sample (llama2chat lines 185-190).
- **Targets supported by ge_data scripts**: only Vicuna (`ge_data_all_vicuna.py`, fastchat "vicuna" template, default path `vicunav13/13B`) and LLaMA2-Chat (`ge_data_all_llama2chat.py`, "llama-2-chat" template, hardcoded local 13B path at line 22). No Mixtral ge_data script, although train configs exist for vicuna 7/13/33B, llama2chat 7/13/70B and Mixtral 8x7B (`eagle/train/*_config.json`).

## 5. Tree drafting and verification

- **Static tree**: `mc_sim_7b_63` in `eagle/model/choices.py:1-3` — 25 paths, max depth 5, top_k=10 per node (cnets.py:52). It is the default `tree_choices` for `eagenerate`/`ea_generate` (ea_model.py:162, 262) and is attached to the draft via `init_tree` (cnets.py:498-500, called at ea_model.py:64). Target-side buffers via `generate_tree_buffers` (utils.py:90-227).
- **Verification/acceptance**: `evaluate_posterior` in `eagle/model/utils.py:320-412`. Greedy (temperature 0): `posterior_mask = (candidates[:,1:] == argmax(logits[:,:-1]))`; `accept_length = cumprod(posterior_mask).sum(1).max()` (lines 350-361). Sampling: per-token speculative acceptance `r <= p(x)/q(x)` with residual-distribution adjustment (lines 363-412). Tree forward pass in `tree_decoding` (utils.py:298-317); state update + KV compaction in `update_inference_inputs` (utils.py:415-472).
- **Acceptance length recording**: `new_token += accept_length + 1` (utils.py:470). Eval scripts record per-round `idxs`, `new_tokens`, `wall_time` into the answer jsonl (`gen_ea_answer_vicuna.py:268-270, 341`); average acceptance length = new_tokens/idxs. Per-depth acceptance rates ("alpha") are recorded by `eagle/model/utils_alpha.py:322-326` via `gen_ea_alpha_vicuna.py`/`gen_ea_alpha_llama2chat.py` (line 363 writes `alpha`/`alpha_num`) and aggregated by `eagle/evaluation/alpha.py`.

## 6. Supported models + official EAGLE-1 weights (README.md:112-115)

| Base model | EAGLE weights (HF repo) |
|---|---|
| Vicuna-7B-v1.3 | yuhuili/EAGLE-Vicuna-7B-v1.3 |
| Vicuna-13B-v1.3 | yuhuili/EAGLE-Vicuna-13B-v1.3 |
| Vicuna-33B-v1.3 | yuhuili/EAGLE-Vicuna-33B-v1.3 |
| LLaMA2-Chat 7B | yuhuili/EAGLE-llama2-chat-7B |
| LLaMA2-Chat 13B | yuhuili/EAGLE-llama2-chat-13B |
| LLaMA2-Chat 70B | yuhuili/EAGLE-llama2-chat-70B |
| Mixtral-8x7B-Instruct-v0.1 | yuhuili/EAGLE-mixtral-instruct-8x7B |

Evaluation scripts per family (`eagle/evaluation/`): Vicuna — gen_ea_answer_vicuna.py, gen_baseline_answer_vicuna.py, gen_ea_alpha_vicuna.py; LLaMA2-Chat — gen_ea_answer_llama2chat.py, gen_baseline_answer_llama2chat.py, gen_ea_alpha_llama2chat.py; Mixtral — gen_ea_answer_mix.py, gen_baseline_answer_mix.py. Plus speed.py and alpha.py.

## 7. Entry points

(a) **EAGLE MT-bench speed eval**: `python -m eagle.evaluation.gen_ea_answer_vicuna --ea-model-path [EAGLE weights] --base-model-path [target]` (README.md:263-265; llama2chat/mix variants analogous). Default `--bench-name mt_bench` (gen_ea_answer_vicuna.py:387); questions read from `eagle/data/mt_bench/question.jsonl` (line 447; file present). Loads `EaModel.from_pretrained` and runs `ea_forward` (lines 25-109).
(b) **Baseline (vanilla)**: `python -m eagle.evaluation.gen_baseline_answer_vicuna ...` (README.md:269-270). Its `ea_forward` is a plain autoregressive loop over `model.base_model(...)` with the preallocated KV cache (gen_baseline_answer_vicuna.py:62-75). Programmatic APIs: `EaModel.eagenerate` (ea_model.py:154), `EaModel.ea_generate` (255), `EaModel.naive_generate` (358).
(c) **Speedup ratio**: `eagle/evaluation/speed.py` — computes `mean(ea tokens/wall_time) / mean(baseline tokens/wall_time)` and prints `ratio` (line 50). NOTE: tokenizer path and the two jsonl filenames are hardcoded (lines 5-7) and must be edited. `eagle/outputs/speed.py` is a similar copy with sample jsonls.
- **Batch size**: the primary `eagle/model` path is **batch=1 only** — asserts "Only support batch size 1 for now!!" (ea_model.py:269, 373; gen_ea_answer_vicuna.py:26), `batch_size = 1` in `initialize_past_key_values` (eagle/model/kv_cache.py:88), and `current_length_data.fill_` comment "currently only support batch size is 1" (utils.py:451). Batch>1 is provided by the separate `eagle/modelbsne1/` package (`from eagle.modelbsne1.ea_model import EaModel` with left padding, README "Batch size > 1" section; its `eagenerate` handles per-batch finish flags, modelbsne1/ea_model.py:139-293). The generic `eagle/modeling_eagle.py` (`class EAGLE`, line 1555) also supports bs>1 and requires transformers > 4.36 per README.

## 8. Dependency risk

- Pins: `requirements.txt` — `torch==2.0.1`, `transformers==4.36.2`, `accelerate==0.21.0`, `fschat==0.2.31`, `gradio==3.50.2`, `protobuf==3.19.0`. `setup.py` (version 1.2.1) pins `accelerate==0.21.0`, `fschat==0.2.31`, etc. but leaves `torch`/`transformers` unpinned.
- Local environment: torch 2.6.0+cu124, transformers 4.51.3, accelerate 0.26.0 — all newer than the pins.
- Empirical check: `eagle.model.cnets`, `modeling_llama_kv`, `ea_model`, `utils`, `modeling_mixtral_kv` all import cleanly under transformers 4.51.3 (the modeling files are self-contained copies of transformers v4.31 code, see modeling_llama_kv.py:1).
- Likely breakage/risks with modern stacks:
  - **past_key_values format**: modeling_llama_kv.py uses the legacy tuple-of-(k,v) with a custom preallocated `KVCache` (`past_key_value[0].cat(key_states, dim=2)`, modeling_llama_kv.py:591-592; kv_cache.py:52-65), not the HF `Cache` API. Works only because the model classes are vendored; the tree-mask hook (`self.tree_mask` in `_prepare_decoder_attention_mask`) also depends on these vendored classes and cannot be transplanted onto stock modern LlamaModel.
  - **Rotary API**: cnets.py and modeling_llama_kv.py use the old per-layer `LlamaRotaryEmbedding(head_dim)` + `apply_rotary_pos_emb(q,k,cos,sin,position_ids)` with cos/sin caches indexed by position_ids (cnets.py:102-144) — incompatible with the modern HF rotary interface (position-embeddings-passed-in), but safe here because it is vendored.
  - **transformers >= 4.50**: `PreTrainedModel` no longer inherits `GenerationMixin`, so `base_model.generate()` would not exist on the vendored `LlamaForCausalLM`; EAGLE's own loops never call `.generate()`, so the main flow is unaffected, but the webui/any HF-generate usage could break.
  - **torch 2.6**: `torch.load` now defaults `weights_only=True`; `ea_model.py:104` (`torch.load(load_model_path, map_location=...)` of `pytorch_model.bin`) and train-data loading (`train/main.py:145`) load plain tensor dicts and should still work, but any pickled non-tensor payload would now raise. `torch.utils.checkpoint.checkpoint` is called without `use_reentrant` (cnets.py:614) — deprecation warnings / potential behavior change during training under torch 2.6.
  - accelerate 0.26 vs pinned 0.21 and wandb `entity="yuhui-li"` hardcoded in train/main.py:71 are additional training-path friction points; eval speed.py/alpha.py have hardcoded /home/lyh paths.

## KEY FILES
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/model/cnets.py — Draft (autoregression head) network: class Model with fc = nn.Linear(2*hidden, hidden, bias=True), fc(cat([embed, hidden])) at lines 492/593, one LlamaDecoderLayer stack, topK_genrate tree drafting with target lm_head reuse
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/model/ea_model.py — EaModel wrapper: extracts base-model last_hidden_state (outputs[0], post-norm), passes base_model.lm_head to draft, eagenerate/ea_generate/naive_generate entry points, batch=1 asserts
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/model/modeling_llama_kv.py — Vendored transformers v4.31 Llama with preallocated KVCache tuples and tree_mask hook; final self.norm at line 1074 makes the extracted feature post-norm
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/model/utils.py — Tree buffers, tree_decoding, evaluate_posterior (verification/acceptance), update_inference_inputs (accept_length accounting, re-drafting with target lm_head)
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/model/choices.py — Static draft tree mc_sim_7b_63 (25 paths, depth<=5)
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/train/main.py — EAGLE-1 draft training: SmoothL1 feature loss (v_w=1.0) + soft-CE logit loss via frozen target lm_head (p_w=0.1), uniform hidden-state noise
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/ge_data/ge_data_all_llama2chat.py — Training-data generation: ShareGPT, target forward with output_hidden_states, saves hidden_states[-1]; vicuna variant + allocation.py sharder alongside
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/evaluation/gen_ea_answer_vicuna.py — MT-bench EAGLE speed eval (records idxs/new_tokens/wall_time per round); baseline and alpha variants per model family in same dir
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/evaluation/speed.py — Speedup-ratio computation from ea vs baseline jsonl outputs (hardcoded paths)
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/eagle/modelbsne1/ea_model.py — Separate batch-size>1 implementation (left padding), per README
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/requirements.txt — Pins torch==2.0.1, transformers==4.36.2, accelerate==0.21.0 (local env has torch 2.6.0, transformers 4.51.3)
- /home/thahn1230/eagle_spinquant_w4a4/third_party/EAGLE/README.md — v1 README: yuhuili/* weight repo IDs (lines 112-115), eval/train commands, batch>1 and custom-model guidance

## BLOCKERS

## VERDICTS
### [CONFIRMED] The draft network combines token embedding e and target hidden state h as fc(concat([e, h], dim=-1)) — embedding FIRST — through nn.Linear(2*hidden_size, hidden_size, bias=bias) with bias defaulting to True, and applies NO activation after the fc.
Verified: eagle/model/cnets.py:492 defines self.fc=nn.Linear(2*config.hidden_size,config.hidden_size,bias=bias); cnets.py:593 'hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))' with inputs_embeds (embedding) first; the activated variant is commented out at cnets.py:591 and self.act (defined at 493) is never used in the live path. bias=True default at cnets.py:455 (Model.__init__ signature) and ea_model.py:43-46 (try con['bias'] / except bias=True).

### [CONFIRMED] After the fc, the draft passes through a stack of full Llama decoder layers (self-attention + MLP), configured as num_hidden_layers=1 in the shipped configs, with the first layer's input_layernorm removed and no final norm.
Verified: cnets.py:491 builds nn.ModuleList of LlamaDecoderLayer(config,index); each layer has self_attn + mlp + post_attention_layernorm (cnets.py:377-386). input_layernorm is only created when index!=0 (cnets.py:385-386) and only applied when index!=0 (cnets.py:414-415). All shipped train configs set num_hidden_layers: 1 (e.g. eagle/train/vicuna_7B_config.json:14, llama_2_chat_7B_config.json:14). Forward returns hidden_states directly with no final norm (cnets.py:635-638).

### [CONFIRMED] EAGLE-1 uses the target model's TOP (last) decoder layer output AFTER the final RMSNorm (last_hidden_state), not a second-to-top layer — both at inference and in training data generation.
Verified: ea_model.py:124-132 uses outputs[0] of self.base_model.model, which is last_hidden_state produced after 'hidden_states = self.norm(hidden_states)' at modeling_llama_kv.py:1074 and returned at 1088; the same tensor feeds lm_head (ea_model.py:131) and topK_genrate (ea_model.py:145). Training data uses outs_big.hidden_states[-1] (ge_data_all_llama2chat.py:185-186, ge_data_all_vicuna.py:167), and I confirmed in the installed HF transformers 4.51.3 LlamaModel.forward that the last hidden_states tuple entry is appended AFTER self.norm, i.e. post-final-RMSNorm. No hidden_states[-2]/second-to-top usage anywhere in the repo.

### [CONFIRMED] The draft outputs a predicted hidden feature that is scored by the TARGET model's lm_head (head reuse; no own head), and that predicted feature is recycled autoregressively as the next-step input hidden state during tree drafting.
Verified: ea_model.py:145 passes self.base_model.lm_head into topK_genrate; cnets.py:780 'last_headout = head(last_hidden)' (diff_device fallback uses a clone of the same lm_head weight, ea_model.py:57); cnets.py:807-816 recycles out_hidden as the next-step input hidden_states ('hidden_states = out_hidden[:, -1:]' / 'hidden_states=out_hidden' then self(hidden_states, ...)); utils.py:466-468 passes head=model.base_model.lm_head. The draft Model has no lm_head of its own (only embed_tokens, fc, layers).

### [CONFIRMED] Training (eagle/train/main.py) uses SmoothL1 feature regression (weight v_w=1.0) plus a soft cross-entropy logit loss against the frozen target lm_head distribution (weight p_w=0.1), on ShareGPT-derived (input_ids, hidden_states[-1]) data from eagle/ge_data (Vicuna and LLaMA2-Chat scripts only).
Verified: main.py:324 criterion = nn.SmoothL1Loss(reduction='none'); main.py:356-367 computes target_p from head(data['target']) with the head built from lm_head.weight and frozen (main.py:75-98, requires_grad=False at :98), ploss = -sum(target_p*out_logp)/..., loss = v_w*vloss + p_w*ploss at :367; weights p_w=0.1, v_w=1.0 at main.py:24-25. Data: ge_data_all_vicuna.py:46 loads ShareGPT_V4.3_unfiltered_cleaned_split.json and stores hidden_states[-1] (vicuna:167, llama2chat:186); allocation.py has e = 68000 - 1; eagle/ge_data contains only the vicuna and llama2chat scripts plus allocation.py/__init__.py.

### [CONFIRMED] Tree drafting uses the static tree mc_sim_7b_63 (25 paths, depth<=5, top_k=10); verification/acceptance is evaluate_posterior in eagle/model/utils.py, and acceptance length feeds new_token += accept_length + 1, with per-question idxs/new_tokens/wall_time written by the eval scripts (avg acceptance = new_tokens/idxs; per-depth alpha via utils_alpha.py).
Verified: eagle/model/choices.py:1-3 defines mc_sim_7b_63; programmatic parse confirms exactly 25 paths, max depth 5; top_k=10 at cnets.py:52; init_tree sets self.tree = mc_sim_7b_63 (cnets.py:499). evaluate_posterior at utils.py:320 (ends ~:412); new_token += accept_length + 1 at utils.py:470. gen_ea_answer_vicuna.py:336-341 appends idxs/new_tokens/wall_time per turn and writes them into choices (line 341). utils_alpha.py:322-326 (and 386-390) increment per-depth alpha/alpha_num counters. 'avg acceptance = new_tokens/idxs' is a derived interpretation (tokens per drafting step) consistent with these counters, not a literal code line.

### [CONFIRMED] The primary eagle/model path supports batch size 1 only (explicit asserts and batch_size=1 KV allocation); batch>1 requires the separate eagle/modelbsne1 package; requirements pin torch==2.0.1 and transformers==4.36.2 while the local env runs torch 2.6.0 / transformers 4.51.3, with the vendored legacy tuple+KVCache past_key_values format and old rotary API as the main modernization risks (modules still import cleanly under 4.51.3).
Verified: ea_model.py:269 and :373 'assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"'; kv_cache.py:88 'batch_size = 1'; eagle/modelbsne1 package exists and README.md:166-170 'Batch size > 1' section imports eagle.modelbsne1.ea_model; requirements.txt lines 1-2 pin torch==2.0.1 and transformers==4.36.2; installed env is torch 2.6.0+cu124 and transformers 4.51.3 (pip list + import check); modeling_llama_kv.py:591-592 uses the legacy custom KVCache tuple format 'past_key_value[0].cat(key_states, dim=2)'; import test of eagle.model.{ea_model,cnets,utils,kv_cache,modeling_llama_kv} under transformers 4.51.3 succeeded ('imports OK').

