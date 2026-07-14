# eagle_spinquant_w4a4

Research pipeline integrating **EAGLE-1 speculative decoding** (SafeAILab/EAGLE,
branch `v1`) with **SpinQuant learned-rotation quantization** (facebookresearch/
SpinQuant) for **W4A4 / W4A4KV4** weight-activation quantization of LLaMA-class
models on CUDA GPUs.

Research question: does W4A4(-KV4) quantization noise on the target model's hidden
features break EAGLE-1's feature-level drafting (acceptance rate), and how much do
SpinQuant's learned rotations (vs plain quantization) preserve it?

**Scope note (verified):** SpinQuant's GPU path is *fake quantization* (quantize-
dequantize, FP16 matmuls, no INT4 kernels). All speed numbers here are algorithmic
(acceptance, relative speculative speedup on the same quantized target), NOT real
W4A4 deployment throughput. See `docs/00_REPO_ENV_AUDIT.md` §4.

## Status

**Implementation complete (2026-07-02).** Full pipeline built and run on
Llama-2-7b-chat + EAGLE-llama2-chat-7B. See `docs/05_FINAL_EXPERIMENT_REPORT.md`.

Headline results (avg EAGLE acceptance length; 1.0 = no speculative gain):

| target | naive (stock draft) | Variant A (unrotate) | Variant B (conjugate) | Variant C (retrained) |
|---|---|---|---|---|
| FP16 rotated (no quant) | 1.10 | **4.36** | 2.18 | — |
| W4A4 | 1.14 | **2.94** | 2.14 | 2.65 |
| W4A4KV4 | 1.12 | **2.85** | 2.13 | 2.57 |

- Naive SpinQuant rotation **breaks** the EAGLE-1 draft interface (4.36 → 1.10),
  caused by the hidden-basis mismatch (shown with NO quantization).
- **Variant A (unrotation) is the fix**: exact in FP, best under W4A4 (3.34× relative
  speculative speedup), 38 µs/call overhead.
- wikitext-2 PPL: FP16 6.95 → W4A4 10.63 (RTN+random-H) / 10.44 (learned R).
- FP16 EAGLE speedup 2.84× (real hardware). **W4A4 timing is fake-quant, NOT a
  deployment speedup** (no INT4 kernels; see report §11).
- Discovered finding: Variant B's fc-fold is exact per-token but diverges over full
  generation (EAGLE recycling), documented in `docs/04`.

Planning-stage audit (every claim re-verified against code): `docs/00`–`docs/02`.

## Layout

```
third_party/EAGLE        SafeAILab/EAGLE @ branch v1 (EAGLE-1) — do not edit in place
third_party/SpinQuant    facebookresearch/SpinQuant @ main    — do not edit in place
docs/00_REPO_ENV_AUDIT.md              environment + repo audit, blockers
docs/01_ROTATED_EAGLE1_ARCHITECTURE.md architecture map, interface math, variants A/B/C
docs/02_TASK_QUEUE_FOR_WORKER_MODEL.md implementation task queue (start here)
docs/appendix/                         raw verified code-audit reports (file:line evidence)
configs/model_candidates.yaml          EAGLE-1 x SpinQuant model intersection
configs/default_experiment.yaml        experiment defaults
scripts/                               numbered pipeline scripts (see scripts/README.md)
src/                                   (worker) integration library code
outputs/                               run artifacts (JSONL/CSV), rotations
```

## Quick start

```bash
python scripts/00_env_check.py                    # GPUs, packages, repos, HF auth
python scripts/01_discover_model_intersection.py  # verify model candidates (config.json only)
```

Recommended model pair: `meta-llama/Llama-2-7b-chat-hf` (gated — accept license +
HF token) with draft `yuhuili/EAGLE-llama2-chat-7B`. Non-gated fallback:
`lmsys/vicuna-7b-v1.3` + `yuhuili/EAGLE-Vicuna-7B-v1.3`.

## Hardware

8x RTX 4090 24GB. Default to 1 GPU (7B fits: ~13.5GB target + ~0.5GB draft, fp16).
Multi-GPU only for rotation optimization (DDP wall-clock) and Variant-C data
generation sharding.
