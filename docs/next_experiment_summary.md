# Next-experiment summary: Vicuna generality + real INT4 kernels

Date: 2026-07-04. Extends `runs/rotstudy_20260704_1507` (Llama-2-7b-chat).
New run dirs: `runs/rotstudy_vicuna_20260704_1824` (Experiment A),
`runs/rotstudy_realint4_20260704_1857` (Experiment B).
Tone: every number below is fake-quant, real-W4A16, or real-W4A4-linears as
labeled; nothing here is a deployment-speed claim.

## The ten questions

**1. Does Vicuna reproduce the Llama-2-chat basis mismatch?**
YES. n=80 MT-bench, quant OFF: stock 3.647 [3.52, 3.78] vs naive 1.134
[1.12, 1.15]. Robust to rotation seeds 1/2 (naive 1.137/1.145) and to the
wikitext prompt set (naive 1.126). Numeric interface check: naive hidden
cosine to true hidden 0.0021.

**2. Does gamma correction remain necessary?**
YES. A_nogamma = 1.071 [n=80] — as collapsed as naive (1.134); numerically,
unrotation WITH gamma_f reaches rel-L2 0.42% / cosine 0.99999 to the true
hidden, without gamma 44% / 0.966.

**3. Does B2 remain equivalent to A?**
YES, on both pairs and every condition measured. Vicuna quant OFF:
A 3.650 vs B2 3.647; W4A16 3.493/3.498; W4A4 3.462/3.469; W4A4KV4
3.457/3.457; rotation seeds 1/2 agree; level-wise diagnostics: B2 cosine
1.0 at all 5 tree levels. Real-kernel arms (Llama-2): tinygemm W4A16
A 3.434 vs B2 3.433; QuaRot W4A4 4.177 vs 4.180 (but see Q5 caveat on that
arm's quality).

**4. Does B still fail under recurrent feature recycling?**
YES. Vicuna depth sweep (n=40): B flat 2.139 -> 2.181 (+0.04) from depth 2
to 5 while A/B2/stock rise 2.66 -> 3.51 (+0.85); single-forward identity
still exact (validation JSON verdicts: B_single_forward_exact=true,
B_recycled_levels_diverge=true). Same failure signature as Llama-2
("first-layer-only folding failure under recurrent feature recycling" —
NOT an impossibility claim).

**5. Does fake W4A4 behavior generalize?**
YES for the fake-quant recipe: Vicuna W4A4 A 3.462 (from 3.650 quant-OFF;
-0.19) and W4A4KV4 3.457 — the same "reduces but does not destroy" pattern
(Llama-2: 3.573 -> 3.284/3.276). Component attribution under W4A4 largely
generalizes but is NOT strictly monotone on Vicuna: none 1.642 -> r1 3.011
-> r1r2 2.956 -> r1r2r3r4 3.297 (n=40; the r1-vs-r1r2 dip is within noise
at this n, the full set is clearly best; Llama-2 was monotone 1.428 ->
2.655 -> 2.903 -> 3.174).
Vicuna PPL sanity: fp16 6.903, rotate-only 6.902 (rotation lossless),
W4A4 8.69, W4A4KV4 8.84.
CAVEAT for REAL W4A4: our real-kernel recipe (per-channel symmetric RTN
weights, per-token symmetric A4, no GPTQ/clip search) degrades Llama-2-chat
badly — integrated-target wikitext PPL 66.5 and repetitive completions
(distinct-2 0.28 vs fp16 0.97). Its acceptance of 4.18 EXCEEDS fp16 stock
(3.58) precisely because degraded output is easier to predict. That number
is an artifact and must never be cited as an improvement.

**6. Is a real INT4 kernel path available?**
YES, two, both with kernel-name proof (docs/real_int4_kernel_audit.md):
- torch-native tinygemm: REAL W4A16 (packed int4 weights, bf16 activations),
  e2e-integrated into the EAGLE target; profiler shows
  `tinygemm_m16n8k16_chunk_kernel<..., BLayout_TC_int4<...>>` inside real
  tree-decoding runs; 4-bit weight memory confirmed (~6.8 GiB peak vs 15.3
  fp16).
- QuaRot CUTLASS (built for sm_89 at /data/thahn1230/quarot): REAL W4A4 for
  the 7 per-layer linears (int4 weights AND per-token int4 activations),
  e2e-integrated; profiler shows the CUTLASS `integer_subbyte<4>` GEMM +
  `sym_quantize_f16_i4_kernel`. KV cache/lm_head/attention stay fp16 in all
  arms; KV4 is NOT real anywhere in this project.

**7. If yes, does B2 improve wall-clock speed over A?**
NO measurable improvement — and no measurable penalty (the honest headline).
Llama-2, n=80 paired prompts:
- tinygemm W4A16: B2 - A = -0.18 tok/s, 95% CI [-0.84, +0.35] (contains 0).
- QuaRot W4A4: main run showed B2 - A = -3.96 tok/s (CI excluding 0), but
  the gap sat in the TARGET phase, which B2 cannot causally touch; a
  reversed-order rerun (B2 first) flipped the sign (B2 90.2 vs A 89.6
  tok/s) => order/thermal drift, not a B2 cost. Verdict: unresolved-at-zero;
  no direction claimable.
Mechanistically both overheads are noise at 7B batch-1: A's unrotation GEMM
costs 143-152 us per draft entry; B2's two-path dispatch (Python branch +
.data pointer swap) costs 0.5-0.7 ms per PROMPT. B2's value is therefore
architectural (removes a runtime GEMM and its basis-coupling from the
interface), not a measurable latency win at this scale. Do NOT claim "B2 is
faster" and do NOT claim "B2 has no overhead" — it has a measured, tiny one.

**8. If no, what exactly blocks real INT4 deployment?**
The kernel path exists, so the blockers are integration-quality ones
(docs/real_int4_limitations.md):
- The unfused per-linear chains make batch-1 decode SLOWER than fp16 e2e:
  vanilla 37.7 tok/s (W4A16) and 22.6 (W4A4) vs 43.3 (fp16). tinygemm is a
  decode-shape kernel (0.4-0.76x at EAGLE's M~26 tree batch); QuaRot's
  chain is 3+ kernel launches vs 1 fp16 GEMM.
- EAGLE+W4A16 still beats its OWN vanilla by 2.59x (97.4 vs 37.7 tok/s) —
  the speculative win survives real quantization — but the fp16 stock
  pipeline (120 tok/s) remains fastest end-to-end on this hardware.
- Real W4A4 needs a proper weight quantizer (GPTQ + clip search) before its
  quality is usable (Q5); KV4 needs a portable real KV-cache kernel.

**9. Which claims are now safe?**
- "The rotated-residual basis mismatch collapses EAGLE-1 acceptance on two
  independent model+draft pairs (Llama-2-7b-chat, Vicuna-7B-v1.3)."
- "The gamma-corrected inverse restores stock acceptance on both pairs."
- "The two-path folded interface (B2) is equivalent to runtime unrotation
  (A) on both pairs, under fake quant AND under real packed-INT4 kernels."
- "First-layer-only folding fails under recurrent feature recycling on both
  pairs (flat depth curve; exact single-forward identity)."
- "The interface fix carries over unchanged to real INT4 execution, with
  kernel-name proof of INT4 dispatch."
- "At 7B batch-1, the unrotation GEMM removed by B2 is microseconds-scale;
  neither A nor B2 has a measurable wall-clock advantage."

**10. Which claims remain unsafe?**
- Any real-W4A4 QUALITY claim from our RTN-based integration (PPL 66.5).
- Any deployment tokens/s claim (batch-1, unfused, no serving stack).
- "B2 is faster than A" (measured: indistinguishable) or "B2 is free"
  (measured: ~0.6 ms/prompt dispatch overhead).
- KV4 anything (never real here).
- Learned-rotation conclusions (never trained for Vicuna; 50-step Llama run
  remains inconclusive).
- Generality beyond LLaMA-1/2-class 7B EAGLE-1 pairs (two pairs ≠ all
  speculative decoders; both drafts share the same cnets.py architecture).

## Recommended next experiment (one, concrete)

Make the real-W4A4 arm quality-credible, then re-ask Q7 where INT4 actually
wins: (a) replace per-channel RTN with GPTQ + clip-searched weight quant
(SpinQuant's own recipe) inside `realint4.swap_target_linears`, target
integrated-PPL <= 12; (b) batch the QuaRot quant+GEMM+dequant chain (their
fused epilogue or CUDA graphs) so the M~26 tree forward stops being
launch-bound; (c) rerun the A-vs-B2 paired comparison with interleaved
variant order on both pairs. Secondary: a third, architecturally different
pair (e.g. Mixtral EAGLE or an EAGLE-2 runtime) to break the shared-cnets
confound in the generality claim.
