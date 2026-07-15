# SpinQuant PPL Expectation Audit

Purpose: determine precisely why the current target W4A4 WikiText-2 PPL is
10.3263 while the SpinQuant paper (ICLR 2025) reports ≈5.9–6.1 for
LLaMA-2-7B W4A4KV16 — WITHOUT assuming either number is wrong.

Branch: `exp/eagle1-spinquant-ppl-reproduction-fix` (from b44152d).
Vendored SpinQuant: 8f47aa3 (untouched). EAGLE: 4a9cf3a.

## Current numbers under audit (from the bitwidth-AL study grader run)

| config | CE (nats/tok) | PPL | n_tokens |
|---|---|---|---|
| chat FP16 | 1.92539 | 6.8578 | 32767 |
| chat random-Hadamard fake W8A8 (RTN+MSE clip) | 1.92799 | 6.8757 | 32767 |
| chat random-Hadamard fake W4A4 (RTN+MSE clip) | 2.33469 | 10.3263 | 32767 |

CE delta (W4A4 − FP16) = **+0.40930 nats/token**; PPL ratio = 1.5058.

## Paper anchors (LLaMA-2-7B **base**, WikiText-2, seq 2048, official eval)

| config | PPL |
|---|---|
| FP16 | ≈ 5.5 |
| learned SpinQuant W4A4KV16, RTN | ≈ 6.1 |
| learned SpinQuant W4A4KV16, GPTQ | ≈ 5.9 |

Implied paper-side CE delta at W4A4KV16-RTN ≈ ln(6.1/5.5) ≈ **+0.104**;
GPTQ ≈ ln(5.9/5.5) ≈ +0.070.

These are diagnostic references, NOT values to hard-code into any output.

**The paper values are for `meta-llama/Llama-2-7b-hf`. The EAGLE experiment
uses `meta-llama/Llama-2-7b-chat-hf`. Absolute PPL equality between those
models is not expected.** All comparisons in this study report (a) absolute
PPL, (b) CE delta from each model's own FP16 baseline, (c) relative PPL
ratio — never absolute PPL alone across base/chat checkpoints.

## Known differences: current run vs paper

| axis | current (10.3263 run) | paper |
|---|---|---|
| model | Llama-2-7b-**chat**-hf | Llama-2-7b-hf (base) |
| rotation | random Hadamard, seed 0, R1+R2 folded offline (R4 online in draft path only where stated) | learned Cayley-optimized R1/R2 (800 samples, 100 steps, lr 1.5) |
| weight quant | RTN + **MSE clip** (SpinQuant WeightQuantizer w_clip) | main: GPTQ; ablation: RTN |
| act quant | per-token asymmetric, last-dim reduction, project wrapper placement | official ActQuantizer placement inside ptq_model |
| KV | fp16 (KV16) | KV16 in the anchor rows |
| evaluator | project `wikitext_ce` (2048-token chunks off a 32767-token prefix, custom BOS handling) | official `utils.eval_utils.evaluator` via ptq.py (full test set, seq 2048, LlamaTokenizerFast add_bos=False add_eos=False) |
| harness | in-process `study.build_study_target` | official `eval_utils.main.ptq_model` |

## Hypotheses (to be separated, not assumed)

- **H-EVAL**: the project evaluator mismeasures CE (token count 32767 is a
  truncated prefix, not the full test set; BOS/window policy may differ).
  Test: §5 evaluator parity on a shared token tensor (GATE B).
- **H-MODEL**: chat checkpoint is intrinsically more W4A4-fragile and has a
  different FP16 baseline (6.86 vs 5.5 already shows base≠chat).
  Test: §6 vs §8 same-recipe base/chat pairs.
- **H-ROT-LEARNED**: random Hadamard vs learned rotation explains most of
  the gap (paper's own ablations show random-rotation variance).
  Test: §7 seed sweep + §8 learned chat rotation.
- **H-ROT-SEED**: seed 0 specifically is an unusually bad draw.
  Test: §7 10-seed sweep distribution.
- **H-QUANT-RECIPE**: RTN+MSE-clip vs official RTN clip / GPTQ.
  Test: §10 recipe ablations under a fixed rotation.
- **H-PIPELINE**: the in-process EAGLE integration diverges from official
  ptq_model (R2/R4 handling, quantizer placement, clipping).
  Test: §9 module-hash + layerwise divergence audit (GATE D).

## Sanity arithmetic

If H-MODEL accounts for the FP16 shift (5.5→6.86) and the paper's
learned-RTN delta (+0.104) carried over to chat, we would expect chat
learned-W4A4-RTN PPL ≈ 6.86 × e^0.104 ≈ 7.6 — far below 10.33. The current
CE delta (+0.409) is ~4× the paper's learned-rotation delta. Whether that
factor comes from the random rotation, the recipe, the pipeline, or the
evaluator is exactly what Phases 2–9 must separate.

## Execution deviations

- Physical GPU 7 absent from the bus (2026-07-15, persists); running with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6` (single GPU),
  physical GPUs 0–5 never used.
- `meta-llama/Llama-2-7b-hf` (base) is not in the local HF cache; it will
  be downloaded once to the /data-backed cache. If gated access fails, the
  vicuna fallback policy applies (CLAUDE.md) and will be recorded.
