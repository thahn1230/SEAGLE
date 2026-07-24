# EAGLE-1 Official Training Recipe Audit (v1 branch, executable-code truth)

Audited from the vendored official code `third_party/EAGLE` (v1):
`eagle/train/main.py`, `eagle/model/cnets.py`,
`eagle/ge_data/ge_data_all_llama2chat.py`,
`eagle/train/llama_2_chat_7B_config.json`. Every claim below cites code,
not README prose. Study: `exp/eagle1-official-fromscratch-ptq-vs-qat`,
run `runs/eagle1_official_fromscratch_ptq_vs_qat_20260724_213243`.

## 1. Exact official behavior (code citations)

### Fresh draft initialization (cnets.Model.__init__, cnets.py:454-497)
- `Model(config, load_emb=True, path=<target>)`: `embed_tokens` **copied
  from the target's safetensors** then **frozen**
  (`param.requires_grad = False`, cnets.py:494-495).
- `self.layers = [LlamaDecoderLayer(config, index=0)]` and
  `self.fc = nn.Linear(2H, H, bias=True)` are **plain nn.Module
  constructions with PyTorch DEFAULT initializers** (kaiming-uniform for
  Linear weights, uniform bias). No `_init_weights`, no
  `initializer_range=0.02` normal init is ever applied — the config's
  `initializer_range` is unused by the draft. Decoder layer index 0 has
  **no input_layernorm** (EAGLE's custom LlamaDecoderLayer).
- `gradient_checkpointing = True` by default and the training forward
  uses it (cnets.py:461, 605).
- LM head: the frozen **target** head, loaded in main.py (fp16), used for
  the soft-CE loss only; `for param in head.parameters():
  param.requires_grad = False` (main.py:98).

### Data pipeline (ge_data_all_llama2chat.py)
- ShareGPT V3 json → `ds.shuffle(seed=42)` → `select(range(start, end))`
  (official launcher `allocation.py` covers 0..68000 in 4 shards).
- fastchat `llama-2-chat` template with the pinned system prompt;
  `tokenizer(use_fast=False)`, `max_length=2048, truncation=True`.
- Loss mask: assistant turns only, with the `-2` tokenizer offset, `+2`
  per turn, `-1` legacy correction; `loss_mask[cur_len:] = 0`.
- Teacher record per conversation: `outs_big = bigmodel(input_ids,
  output_hidden_states=True)`; `hidden_state_big = hidden_states[-1]`
  (**post-final-RMSNorm**, gamma included) — stored fp16 with input_ids
  and loss_mask, one `.ckpt` file per conversation.

### Training loop (main.py)
- Records: `CustomDataset` shifts: model input token at row i is
  `input_ids[i+1]`; regression target at row i is `hidden_state[i+1]`
  (zero-padded tail, `loss_mask[-1]=0`).
- Noise: `AddUniformNoise(std=0.2)`: `(rand-0.5)*0.2*512/len` added to
  the INPUT hidden only (transform applied after target construction).
- Loss (main.py:357-367): `vloss = SmoothL1(predict, target)` masked
  token-mean; `ploss = -Σ softmax(head(target))·log_softmax(head(pred))`
  masked; `loss = v_w*vloss + p_w*ploss` with **v_w=1.0, p_w=0.1**
  (`head_w=0.1` exists in config but is never used in v1).
- Optimizer: `optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9,0.95))`
  — **weight_decay is left at the PyTorch default 0.01** (not 0).
- Grad clip: `accelerator.clip_grad_value_(params, 0.5)` (VALUE clip).
- Scheduler: `get_linear_schedule_with_warmup(2000,
  num_training_steps=800000)` — a **fixed 800k-step horizon**, NOT
  `len(loader)*epochs`. With the real run (~340k updates) LR ends at
  ≈58% of peak, never 0.
- Batch: `bs=4` per process, `gradient_accumulation_steps=1` (defaults);
  effective global batch = 4 × world_size — **the official world size is
  not pinned anywhere in code**.
- Mixed precision: `Accelerator(mixed_precision='bf16')`.
- Epochs: `for epoch in range(num_epochs + 1)` with num_epochs=20 →
  **21 passes** (off-by-one in official code).
- Data order: `list_files()` uses `os.walk` **without sorting** —
  filesystem-order dependent, not reproducible even officially. Split:
  train = first 95% of that order, test = last 5%.
- Validation: on epochs where `(epoch+1)%5 != 0`: top-1/2/3 head-argmax
  agreement + `getkacc(max_length=5)` (teacher-forced multi-step
  accuracy) on 10 test batches. Checkpoints: `accelerator.save_state`
  per epoch block (main.py:484). **No best-checkpoint selection exists
  in official v1 code.**

## 2. Local reproduction behavior

Same objective/optimizer/scheduler constants, official cnets.Model
(fresh init exactly as above), official record semantics and noise,
bf16 accelerate DDP, grad ckpt on, official 68k-conversation coverage,
21 epochs, official 95/5 split, official validation metrics, per-epoch
state saves. Effective global batch chosen by the §2B benchmark
(bs=4/process, world size from profiling; LR NOT rescaled — official
code does not scale LR by world size).

## 3. Unavoidable deviations (declared BEFORE training)

D1. **Storage — fused teacher instead of pre-generated .ckpt files.**
    The official pipeline stores ≈550–700 GB of fp16 hidden states for
    68k conversations; this host has 171 GB free (`/data`, root full).
    We therefore compute the identical teacher record (`hidden_states[-1]`
    of the frozen fp16 target, identical tokenization/masking) **online
    per batch**. Parity is PROVEN, not assumed: an 8-GPU-sharded audit
    subset (§7 of the study spec) writes real records with sample-level
    SHA hashes, and the fused path must reproduce them bitwise
    (`data_audit/`). This was validated once already in the
    2026-07-23 PTQ-vs-QAT study.
D2. **Deterministic data order.** Official `os.walk` order is
    filesystem-dependent (irreproducible by construction). We use the
    deterministic conversation order (shuffle(seed=42) → index), and the
    95/5 split over that order. This is strictly more reproducible; it
    cannot match an unknowable official order.
D3. **World size.** Official effective batch is undocumented; we pin
    bs=4/process and select world size by the pre-registered 100–200-step
    benchmark, recording the choice. LR stays 3e-5 (code does not scale).
D4. wandb → local jsonl logging. tqdm → periodic prints.
D5. Hardware: 8×RTX 4090 24 GB (bf16 supported natively).

## 4. Deliberate experimental extensions (NOT part of the FP16 reproduction)

- P3 alpha calibration per target/interface (study §14).
- Single-step low-LR QAT from the frozen anchor (§16-18).
- Multi-step quantized-rollout QAT (§19) — labeled non-official.
- Local residual rotation R_D (frozen weights; rotation calibration).
None of these touch the primary FP16 reproduction run.

## 5. Pre-registered reproduction gates (fixed before any test evaluation)

- Absolute: fresh MT-Bench tau within 0.20 of the public checkpoint.
- Relative: fresh tau ≥ 94% of public-checkpoint tau.
Public reference (this repo's official aggregation): C1 = 3.5762.
So the gate is: fresh tau ≥ max(3.3762, 3.3616) = **3.3762**.
