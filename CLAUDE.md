# CLAUDE.md — eagle_spinquant_w4a4

EAGLE-1 speculative decoding x SpinQuant W4A4(-KV4) rotation quantization,
LLaMA-2-7B-chat class, 8x RTX 4090 24GB. Planning stage is DONE; you are the
implementation worker. Your queue: `docs/02_TASK_QUEUE_FOR_WORKER_MODEL.md` —
work it top-to-bottom, respect the GATEs.

## Read before coding
1. `docs/00_REPO_ENV_AUDIT.md` — environment, blockers, why speed numbers are
   fake-quant-only.
2. `docs/01_ROTATED_EAGLE1_ARCHITECTURE.md` — the interface math (the γ_f fusion
   subtlety, concat order, variants A/B/C). Do not re-derive; it is code-verified.
3. `docs/appendix/A*.md` — file:line evidence for every claim; consult before
   spelunking third_party yourself.

## Hard rules
- NEVER edit `third_party/EAGLE` or `third_party/SpinQuant` in place. Wrap from
  `src/`/`scripts/`, or keep git-diff patches in `patches/`.
- EAGLE stays on branch `v1` (EAGLE-1). Main branch is EAGLE-2/3 — different runtime.
- No gated-access workarounds. If meta-llama access fails, switch to the vicuna
  fallback in `configs/model_candidates.yaml` and note it in README.
- 1 GPU by default (`CUDA_VISIBLE_DEVICES`); multi-GPU only where the task queue
  says (rotation optimization DDP, Variant-C data sharding). Check `nvidia-smi`
  before grabbing GPUs — the box may be shared.
- Long jobs (rotation opt, GPTQ, draft training, MT-bench sweeps): run under
  `nohup ... > outputs/<run>/log.txt &`, then verify liveness before moving on.
- Every measurement run writes JSONL under `outputs/<run_name>/` including both
  third_party commit hashes and the exact command line.

## Environment gotchas (verified)
- Installed: torch 2.6.0+cu124, transformers 4.51.3. EAGLE v1 modules import clean
  (vendored 4.31-era code). SpinQuant is pinned 4.44.2 and imports private HF APIs —
  test first; make `envs/spinquant` venv if broken (task E-1).
- `fast_hadamard_transform` must be pip-installed (CUDA build) before any R3/R4 run.
- SpinQuant `ptq.py` requires torchrun even single-GPU (unconditional NCCL init).
- torch.load defaults weights_only=True on torch 2.6 (R.bin is a plain tensor dict).
- The "known traps" list at the bottom of the task queue doc is the distilled bug
  list — read it before tasks I-1..V-3.

## Validation discipline
- Run `python scripts/00_env_check.py` after env changes.
- FP-equivalence tests (task I-2) gate all quantized work: rotated-FP16 target must
  match the original to near-roundoff, and Variant A with quant OFF must reproduce
  stock EAGLE-1 accepted tokens exactly (greedy).
- Sanity anchors: W4A4 wikitext-2 PPL for Llama-2-7B should land ~6-8 (SpinQuant
  paper ballpark); FP16 EAGLE-1 MT-bench speedup ~2.5-3x. Far off → bug, not result.
