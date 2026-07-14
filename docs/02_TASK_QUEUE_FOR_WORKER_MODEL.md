# 02 — Implementation Task Queue (for the worker model)

Work top-to-bottom; do not start a section before its `GATE` passes. Read
`docs/00_REPO_ENV_AUDIT.md` and `docs/01_ROTATED_EAGLE1_ARCHITECTURE.md` first; raw
file:line evidence is in `docs/appendix/`. Never edit files inside `third_party/`
in place — wrap or monkeypatch from `src/`, or keep patches in `patches/` as git
diffs. Default model: `meta-llama/Llama-2-7b-chat-hf` + `yuhuili/EAGLE-llama2-chat-7B`
(fallback: vicuna-7b-v1.3 pair, see `configs/model_candidates.yaml`). Use 1 GPU unless
a task says otherwise. Log every run as JSONL under `outputs/<run_name>/`.

## E. Environment (targets 1-2)

- [ ] E-1 Resolve the transformers conflict: try `import eval_utils.modeling_llama`
      (SpinQuant) under installed transformers 4.51.3; if the private-API imports
      (`_flash_attention_forward`, `ROPE_INIT_FUNCTIONS`) fail, create
      `envs/spinquant` venv with `pip install -r third_party/SpinQuant/requirement.txt`
      and document activation in README. EAGLE modules already import clean on 4.51.3.
- [ ] E-2 `pip install fast-hadamard-transform` (CUDA build; needed for R3/R4).
      Verify: `python -c "import fast_hadamard_transform"`.
- [ ] E-3 Extend `scripts/00_env_check.py`: add the tiny-random-Llama rotation
      equivalence dry run (marked TODO in the file). Keep it download-free.
- [ ] E-4 Confirm gated access: `huggingface-cli whoami`; download
      `meta-llama/Llama-2-7b-chat-hf` config only. If license not accepted, switch
      configs to the vicuna fallback and record the decision in README.
      GATE: `python scripts/00_env_check.py` exits 0 with no WARN on hadamard.

## M. Models & intersection (target 3)

- [ ] M-1 Run `python scripts/01_discover_model_intersection.py` (fetches config.json
      only); fix any FAIL. Then download target + draft weights
      (`huggingface-cli download`), record disk paths in
      `configs/default_experiment.yaml: paths`.
- [ ] M-2 Smoke test stock EAGLE-1 FP16 (batch 1, 2-3 MT-bench questions):
      `python -m eagle.evaluation.gen_ea_answer_llama2chat --ea-model-path ...
      --base-model-path ...` from `third_party/EAGLE`. GATE: nonzero speedup vs
      baseline script on the same questions; save JSONLs to `outputs/smoke_fp16/`.

## R. SpinQuant rotation + PTQ (targets 4-5)

- [ ] R-1 Write `scripts/10_optimize_rotation.py` wrapper: invokes SpinQuant
      `optimize_rotation.py` via torchrun for the CHAT target. Start
      `--nproc_per_node=1`; on OOM escalate to 2/4/8 GPUs (DDP replicates memory —
      more GPUs only shortens wall-clock; if 1 GPU OOMs at bs=1+checkpointing, try
      seqlen 1024 or FSDP script 11). Per SpinQuant README: since PTQ uses GPTQ,
      optimize rotations with `w_bits 16` (`16 4 4` args); also produce a `4 4 4`
      variant for ablation. Output: `outputs/rotations/R.bin`. Record wall-clock+VRAM.
- [ ] R-2 Write `scripts/20_ptq_eval.py` wrapper: runs SpinQuant `ptq.py`
      (`2_eval_ptq.sh` args) for W16A16 (rotate-only sanity), W4A16, W4A4, W4A4KV4
      with `--optimized_rotation_path outputs/rotations/R.bin`.
      GATE: wikitext-2 PPL ladder is sane (rotate-only ≈ FP16 baseline; W4A4 for
      Llama-2-7B should land in the ~6-8 range per SpinQuant paper; >20 means broken).
      Save `outputs/ptq/ppl.jsonl` (target 11 partially satisfied here).

## I. Integration core (targets 6-8) — read architecture doc §4 first

- [ ] I-1 `src/rotated_target.py`: build EAGLE's vendored `modeling_llama_kv` model,
      then apply SpinQuant weight-level pipeline: stash γ_f (final-norm weight) and
      original `W_lm`/`embed_tokens` copies → `fuse_layer_norms` → `rotate_model`
      (load R.bin) → `add_actquant` + online R4 on down_proj → GPTQ (or `--w_rtn`
      first for speed) → V-cache quantizer on v_proj → port R3 `QKRotationWrapper`
      onto EAGLE's 5-arg `apply_rotary_pos_emb` call. Keep every stage flag-gated so
      each can be toggled for ablation.
- [ ] I-2 FP equivalence tests (`tests/test_fp_equivalence.py`), all with quant OFF:
      (a) rotated vs original target logits (fixed 8-prompt set): report max|Δ|;
      (b) hidden capture: `(ĥ @ R1ᵀ) ⊙ γ_f` vs original model's `outputs[0]`;
      (c) PPL parity of I-1's quantized model vs SpinQuant's own ptq.py on the same
      R.bin (W4A4) — this validates the R3/R4/quant port. GATE for everything below.
- [ ] I-3 `scripts/30_eagle_baseline.py` (target 6): parameterized wrapper around
      EAGLE MT-bench eval + baseline + speed ratio (no hardcoded paths; reuse
      gen_ea_answer_llama2chat internals via import, not copy-paste).
- [ ] I-4 `src/capture_hooks.py` (target 7): capture ĥ at BOTH sites — prefill
      (`ea_model.py:122-132` equivalent) and post-verification re-draft
      (`utils.py:466-468` equivalent) — as a wrapper class around EaModel, feeding the
      adapter from architecture doc §2.
- [ ] I-5 Variant A (target 8): `scripts/40_rotated_eagle_variant_a.py` — explicit
      `h = (ĥ @ R1ᵀ) ⊙ γ_f`, stock draft, original-head copy passed to topK_genrate.
      GATE: with quant OFF, accepted-token sequences identical to M-2 smoke run
      (greedy); with W4A4 ON, pipeline runs end-to-end and reports acceptance.

## V. Variants B & C (targets 9-10)

- [ ] V-1 Variant B (target 9): `scripts/41_rotated_eagle_variant_b.py` — patch
      `fc.weight[:, D:2D] @ diag(γ_f) @ R1` at load; draft consumes ĥ directly.
      GATE: draft logits match Variant A to float roundoff (<1e-3 rel) on 100 steps.
- [ ] V-2 Variant C data (target 10a): `scripts/42a_gen_rotated_data.py` — adapt
      `ge_data_all_llama2chat.py` to run the QUANTIZED rotated target and store ĥ
      (start with 10-20k ShareGPT samples, one GPU per shard via allocation.py
      pattern; 8 GPUs useful here).
- [ ] V-3 Variant C training (target 10b): `scripts/42_train_rotated_draft.py` —
      adapt `eagle/train/main.py`: init from V-1 conjugated draft, SmoothL1 on ĥ +
      soft-CE against the QUANTIZED target's logits (see architecture doc §3C);
      remove wandb hardcodes. Short run first (1 epoch, subset) before full training.

## X. Evaluation & reporting (targets 11-15)

- [ ] X-1 (target 11) `scripts/45_eval_ppl.py`: wikitext-2 PPL for every target config
      (FP16 / rotate-only / W4A16 / W4A4 / W4A4KV4) through OUR ported model (I-1),
      cross-checked against R-2 numbers.
- [ ] X-2 (target 12) `scripts/50_eval_speculative.py`: for {FP16, W4A4, W4A4KV4} x
      {no-EAGLE, A, B, C}: tokens/s, wall-time, avg acceptance length (new_tokens /
      drafting rounds), per-depth alpha (reuse utils_alpha counters), MT-bench.
      Report RELATIVE speedup vs no-EAGLE on the same target config (audit doc §4).
- [ ] X-3 (target 13) `scripts/51_batch_sweep.py`: batch sizes 1/2/4/8 via
      `eagle/modelbsne1` (primary path is batch=1 only). If modelbsne1 breaks under
      the rotated target, document and ship batch=1 results — do not sink >1 day.
- [ ] X-4 (target 14) `scripts/60_aggregate_results.py`: all run JSONLs →
      `outputs/results.csv` + `outputs/results.jsonl` (schema: run_name, variant,
      quant_config, ppl, accept_len, alpha_by_depth, tokens_per_s, rel_speedup, gpu,
      commit hashes of both third_party repos).
- [ ] X-5 (target 15) `docs/03_REPRODUCIBILITY_REPORT.md`: exact commands, seeds,
      versions, R.bin provenance, wall-clocks, the fake-quant disclaimer, and the
      acceptance-vs-quantization result table. State clearly which numbers are
      algorithmic (acceptance, PPL, relative speedup) vs NOT deployment speed.

## Known traps (from verified audit — do not rediscover these)

1. Concat order is `cat([embed, hidden])` — the h-block of `fc.weight` is columns
   `D:2D`, not `0:D`.
2. `h = ĥ @ R1ᵀ` alone is wrong — γ_f was fused into lm_head; multiply it back.
3. Draft embedding + head copies must come from the ORIGINAL checkpoint, not the
   fused/rotated in-memory model.
4. `ptq.py` needs torchrun even on 1 GPU (unconditional NCCL init).
5. torch 2.6 `torch.load(weights_only=True)` default — R.bin/pytorch_model.bin are
   tensor dicts (fine), but pass `weights_only=False` only if a load fails and you
   trust the file.
6. EAGLE `speed.py` and ge_data scripts have hardcoded `/home/lyh` paths and a wandb
   entity — never run them unwrapped.
7. R3 wrapper activates only when `k_bits < 16`; W4A4 (KV16) runs WITHOUT R3 — don't
   be surprised when K/V stay FP16 there.
8. Rotation optimization must use `w_bits 16` if PTQ will use GPTQ (SpinQuant README).
