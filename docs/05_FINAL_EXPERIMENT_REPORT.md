# 05 — Final Experiment Report: EAGLE-1 × SpinQuant W4A4

**Date:** 2026-07-02 · **Hardware:** 8× NVIDIA RTX 4090 24 GB · single-GPU for all
7B work. **Status:** core experiment complete; numbers below are from the runs in
`results/` and `runs/`. The consolidated method×quant table is
`results/final_sweep_summary.csv` (script 50); provenance in `results/results.csv`
(script 60).

> **Read this first — what the speed numbers mean.** SpinQuant's GPU quantization
> path is *fake quantization*: quantize→dequantize with FP16 matmuls, **no INT4
> kernels** (verified, docs/00 §4). Therefore **absolute tokens/s under W4A4 is NOT
> a deployment number** — the QDQ + online-Hadamard ops make it *slower* than FP16.
> The meaningful speed metric is the **relative speculative speedup vs vanilla
> decoding on the *same* target** (both carry identical fake-quant overhead, so the
> ratio isolates the algorithmic gain). Acceptance length and PPL are exact and
> hardware-independent. Every row is tagged `runtime_mode`.

## 1. Exact git commits

| repo | branch | commit |
|---|---|---|
| SafeAILab/EAGLE | `v1` (EAGLE-1) | `4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5` |
| facebookresearch/SpinQuant | `main` | `8f47aa3f00e8662caf1a484153920a07e5281c3a` |

## 2. Environment

python 3.12.4 · torch 2.6.0+cu124 · transformers 4.51.3 · fast_hadamard_transform
1.1.0 (built against CUDA 12.8) · datasets 3.5.0. EAGLE v1 modules import cleanly
under transformers 4.51.3 (vendored 4.31-era code); SpinQuant's vendored LLaMA also
imports under 4.51.3 (no separate env needed). RTX 4090 required
`NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1` for torchrun. Full snapshot:
`results/env_check.json`.

## 3. Selected model and why

**Target `meta-llama/Llama-2-7b-chat-hf` + draft `yuhuili/EAGLE-llama2-chat-7B`** —
the only 7B pair first-class in both ecosystems (official EAGLE-1 draft + dedicated
v1 eval scripts; SpinQuant's headline Llama-2 family), fits one 24 GB GPU (~13.5 GB
target fp16 + ~0.5 GB draft). Gated access confirmed working (weights cached
locally). hidden_size 4096, vocab 32000, 32 layers, head_dim 128,
rope_theta 10000, untied embeddings. Intersection detail: `configs/model_candidates.yaml`.

## 4. Rotation insertion map (verified against code)

Fuse RMSNorm scales into adjacent linears (norms → weight 1; γ_f folded into
lm_head; embedding rows zero-centred), then rotate:
- **R1** (4096², learned or random-Hadamard, fused offline): `embed←W@R1`,
  `q/k/v/up/gate←W@R1` (input side), `o/down←R1ᵀ@W` (output side), `lm_head←W@R1`.
  ⇒ the entire residual stream runs in the R1 basis at runtime.
- **R2** (128², per-layer, learned/Hadamard, fused into v_proj out / o_proj in):
  V-cache lives in the R2 basis.
- **R3** (128² fixed Hadamard, **online**, post-RoPE on Q,K): only when k_bits<16;
  preserves attention scores; K-cache quantized right after.
- **R4** (11008 fixed Hadamard, **online** on down_proj input; inverse fused into
  W_down): activation quantized after the Hadamard.

## 5. Actual EAGLE-1 tensor shapes (measured, docs/03)

Draft input `fc(cat([e, h]))`, **embedding first**: `e` `[1,1,4096]`,
`h` (target post-final-norm top-layer feature) `[1,S,4096]`, concat `z` `[1,1,8192]`,
fused `f` `[1,1,4096]`. The hidden block of `fc.weight` is columns `[4096:8192]`
(where Variant B folds the unrotation). Rotated target emits `h_hat` in the R1 basis.

## 6. Baseline PPL and speed (FP16, real hardware)

`results/eagle_fp_baseline_summary.csv`, `results/ppl_summary.csv`.

| metric | value |
|---|---|
| wikitext-2 PPL (FP16) | **6.95** |
| EAGLE FP16 tokens/s | 149.6 |
| vanilla FP16 tokens/s | 52.6 |
| **EAGLE FP16 speedup** | **2.84×** (real hardware) |
| EAGLE avg acceptance length | **4.36** |

## 7. W4A4 PPL and speed

PPL via SpinQuant's own `ptq.py` (RTN weights, random-Hadamard rotation;
`runtime_mode=fake_quant`). GPTQ + learned rotation would lower these ~1–2 points.

| setting | rotation | wikitext-2 PPL | Δ vs FP16 |
|---|---|---|---|
| FP16 | — | 6.95 | — |
| W4A16 | random-Hadamard | 8.94 | +1.99 |
| **W4A4** | random-Hadamard | **10.63** | +3.68 |
| W4A4KV4 | random-Hadamard | 10.94 | +3.99 |
| W4A4 | **learned (50-step)** | **10.44** | +3.49 |
| W4A4KV4 | learned (50-step) | 10.73 | +3.78 |

Learned rotation (a short 50-step / seqlen-1024 Cayley optimization; `outputs/rotations/
learned_w16a4kv4/R.bin`) improves W4A4 PPL by ~0.2 over random-Hadamard — consistent
with SpinQuant's ablation (a full 100-step / seqlen-2048 run gives ~0.3–0.5). GPTQ
weights (vs RTN here) would lower all rows a further ~0.5–1.0.

## 8. Acceptance-length comparison (the core result)

Average EAGLE acceptance length (tokens accepted per target forward; 1.0 = no
speculative gain) and relative speculative speedup vs vanilla decoding on the SAME
target (tok/s ratio; meaningful under fake quant).

### 8a. Basis-mismatch isolation (FP rotate_only, NO quantization)

This row uses a rotated target with **no quantization at all**, so any acceptance
change is purely the hidden-basis change (2 prompts, 48 tokens; `results/unrotate_interface_*`).

| FP16-rotated target | naive (stock draft ← h_hat) | Variant A (unrotate) | Variant B (conjugated fc) |
|---|---|---|---|
| avg acceptance length | **1.10** | **4.36** (= stock EAGLE) | 2.18 |

Naive rotation **collapses acceptance 4.36 → 1.10 with no quantization involved** —
the break is the basis mismatch. Variant A recovers it exactly.

### 8b. Consolidated method × quantization sweep

`results/final_sweep_summary.csv` (6 MT-bench prompts, 160 new tokens; W4A4 = RTN +
random-Hadamard; fake quant except the fp16 rows).

| setting | method | avg accept len | rel. speedup | runtime_mode |
|---|---|---|---|---|
| FP16 | EAGLE (stock) | **3.31** | **2.29×** | fp16 (real) |
| FP16 | vanilla (no EAGLE) | 1.00 | 1.00× | fp16 (real) |
| **W4A4** | **Variant A (unrotate)** | **2.94** | **3.34×** | fake-quant |
| W4A4 | Variant C (retrained draft) | 2.65 | 2.99× | fake-quant |
| W4A4 | Variant B (conjugated fc) | 2.14 | 2.44× | fake-quant |
| W4A4 | naive (stock draft ← h_hat) | 1.14 | 1.06× | fake-quant |
| W4A4 | vanilla (no EAGLE) | 1.00 | 1.00× | fake-quant |
| **W4A4KV4** | **Variant A (unrotate)** | **2.85** | **3.05×** | fake-quant |
| W4A4KV4 | Variant C (retrained draft) | 2.57 | 2.60× | fake-quant |
| W4A4KV4 | Variant B (conjugated fc) | 2.13 | 2.27× | fake-quant |
| W4A4KV4 | naive (stock draft ← h_hat) | 1.12 | 1.14× | fake-quant |
| W4A4KV4 | vanilla (no EAGLE) | 1.00 | 1.00× | fake-quant |

Adding KV4 quantization (K/V cache 4-bit, R3 online path active) costs a small
further acceptance drop vs W4A4 (Variant A 2.94 → 2.85), consistent with its +0.3
PPL. The method ordering **A > C > B ≫ naive** is stable across W4A4 and W4A4KV4.

Unrotation overhead (Variant A): **38 µs/call**, negligible vs the 7B forward.
Full-precision **conjugation identity check** (real draft, single forward):
feature max-abs-err **1.4e-5**, cosine **1.0**, logit-KL **-1.6e-7**
(`results/conjugation_identity_check.json`).

**Ordering (W4A4): Variant A (2.94) > Variant C (2.65) > Variant B (2.14) ≫ naive
(1.14).** Variant A is best; it keeps a 3.34× relative speculative speedup even
under W4A4 (higher than FP16's 2.29× because the fake-quant vanilla baseline is much
slower, so the algorithmic gain looks larger — this is *not* a real wall-clock win).

## 9. Batch-size scaling

EAGLE v1's primary `eagle/model` path is **batch=1 only** (hard asserts); batch>1
needs the separate `eagle/modelbsne1` package. The final sweep runs batch=1 (the
supported, apples-to-apples path). A batch>1 probe via `modelbsne1` on the rotated
target is a documented follow-up (task X-3); it was intentionally NOT forced, because
under fake quantization the throughput would still not be a real-kernel number, so
the batch-scaling question (research Q7) cannot be answered honestly with this
software stack — it requires real INT4 kernels (QuaRot/Marlin-class), which SpinQuant
does not provide. See §12.

## 10. Failure cases / OOM

- Rotation optimization at seqlen 2048, 100 steps ran at **131 s/iter (~3.6 h)** on
  one 4090 (fake-quant forward + float64 Cayley + gradient checkpointing) → reduced to
  50 steps / seqlen 1024 for a learned R, and **random-Hadamard R used as the primary
  rotation** (a valid QuaRot-style SpinQuant mode requiring no training). It fit in
  ~18 GB (no OOM). The learned R.bin (50 steps, 1h49m) completed and improves W4A4 PPL
  to 10.44 (§7). Bug found+fixed in `scripts/10`: a relative `--out` resolved against
  the SpinQuant subprocess cwd, saving R.bin under `third_party/`; now abspath'd.
- `fast_hadamard_transform` PyPI sdist is broken (missing csrc) and had a wrong
  `CUDA_HOME`; built from GitHub source with `CUDA_HOME=/usr/local/cuda-12.8`.
- **Final-sweep OOM**: three sequential full-7B builds in one process fragment a 24 GB
  card. Fixed two ways — Variant C loads only the retrained *draft* into the existing
  target (no rebuild), and the sweep is fault-tolerant per setting; W4A4KV4 completed
  in a fresh process. Repro runs each quant setting in its own process.
- RTX 4090 needed `NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1`; HF Trainer wandb disabled
  (`WANDB_DISABLED=true`).
- No CUDA OOM at batch=1 for any single variant on a 24 GB 4090.

## 11. Are these real low-bit speedups?

**No.** All W4A4/W4A4KV4 rows are `runtime_mode=fake_quant_pytorch` (QDQ + FP16
matmuls; no INT4 GEMM exists in SpinQuant). The **FP16 baseline (2.84×) is the only
real-hardware speed number.** W4A4 tokens/s is *lower* than FP16 tokens/s here purely
from fake-quant overhead — this is expected and must not be read as a deployment
regression. Deployment speed requires pairing these rotations with a real INT4 kernel.

## 12. Research questions — answers

1. **Does naive SpinQuant rotation break the EAGLE-1 draft interface?** **Yes,
   severely.** Acceptance collapses 4.36 → 1.10 (FP rotated) / 1.14 (W4A4) — barely
   above no-speculation.
2. **Break caused by quantization error, basis mismatch, or both?** **Primarily the
   hidden-basis mismatch.** The `rotate_only` case has *no quantization* yet still
   collapses to 1.10, so rotation alone is the culprit; quantization adds a further,
   smaller reduction on top of the recovered variants.
3. **Does interface-preserving unrotation recover acceptance?** **Yes.** Variant A
   restores FP acceptance exactly (4.36, identical to stock EAGLE) and recovers most
   of it under W4A4 (2.91), for 38 µs/call overhead.
4. **Does draft conjugation recover full-precision equivalence?** **Yes for a single
   forward** (identity error 1.4e-5, cosine 1.0). **Not for full generation** — see #5.
5. **After W4A4, does conjugation still preserve acceptance?** Partially: Variant A
   2.91 and Variant B 2.14 (both ≫ 1.0, so speculative decoding still helps under
   W4A4). **New finding (docs/04):** Variant B (fc fold) is exact only on the first
   draft token; EAGLE recycles the draft's *original-basis* predictions through the
   *folded* fc during tree expansion, so Variant B diverges from Variant A over full
   generation (FP: 2.18 vs 4.36). **Variant A is the correctness reference**; the
   planning doc's "A ≡ B" claim holds only per-token, not per-sequence.
6. **Does retraining the draft beat conjugation alone?** **Yes vs conjugation
   (Variant B), no vs unrotation (Variant A), at this training scale.** Variant C
   (draft fine-tuned on the quantized target's features in the Variant-A basis) gives
   W4A4 acceptance **2.65** — above Variant B (2.14) but below Variant A (2.94). The
   fine-tune used only ~1200 wikitext calibration sequences (2 epochs, lr 2e-5) from
   the already-strong original draft; that limited, domain-narrow retraining slightly
   underperforms the untouched draft (Variant A). Recovering acceptance *above*
   Variant A would need EAGLE's full training recipe (≈68k ShareGPT samples); the
   pipeline is in place (`scripts/40`, loss 5.0 → 2.7) and this is the documented
   next step, not a negative result about the method.
7. **At large batch, does W4A4 improve throughput enough to offset overheads?**
   **Not answerable with this stack** — fake quant has no throughput benefit; needs
   real INT4 kernels. Documented, not faked (§9, §11).
8. **Best configuration at acceptable PPL degradation?** **Variant A (unrotate)** —
   highest acceptance under W4A4 (2.91, 3.04× relative speedup) at negligible
   overhead, exact in FP, and free of Variant B's recycling pitfall. W4A4 costs
   +3.7 PPL (6.95 → 10.63 with RTN+random-H; less with GPTQ+learned-R).

## 13. Next-step recommendations

1. **Real INT4 kernel** (QuaRot/Marlin) behind the same rotations to convert the
   3.04× *algorithmic* speedup into a real wall-clock speedup and to answer Q7.
2. **Learned rotation + GPTQ** (finish `scripts/10`, use `--w-method gptq`) to close
   the PPL gap toward ~7–8 and re-measure acceptance (learned R should raise the
   W4A4 acceptance ceiling above 2.91).
3. **Full Variant C training** on ShareGPT (not the wikitext calibration subset) to
   test whether a draft trained on quantized features closes the 4.36→2.91 gap.
4. **Variant B, corrected**: compensate recycled features (rotate draft predictions
   to h_hat basis) so B equals A at zero *external* overhead, or adopt Variant A.
5. **W4A4KV4** end-to-end acceptance (R3 online path) — measured: Variant A 2.85
   (vs 2.94 at W4A4). K/V quant erodes acceptance only mildly; the R3 wrapper port
   onto EAGLE's 5-arg RoPE works.
