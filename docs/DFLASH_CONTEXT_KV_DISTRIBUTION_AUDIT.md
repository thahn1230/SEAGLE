# DFlash Context-K/V Distribution Audit (mechanistic follow-up, 2026-08-08)

Follow-up to `DFLASH_SEAGLE_TRANSFER_STUDY.md`: raw-tensor-level dissection
of the target-context H_t W4A4 K/V-projection bottleneck and of what the
context rotation R_C (= R_T reuse) actually changes. No AL numbers here —
measurement + visualization + causal decomposition only. Run:
`runs/dflash_context_kv_distribution_audit_20260808_150523`
(§30 gate: **missing = 0, mismatch = 0**; 29 scheduler jobs, paired
teacher-forced replay of the C0 greedy trajectory through C0-C4).

Setup identical to the headline study: Llama-3.1-8B-Instruct @0e9e39f2 +
DFlash-UltraChat @d3af30d, learned R_T (sha 99ed53f2), layers [1,8,15,22,29],
B=10, per-token asym A4 / per-channel sym W4, 160 prompts (4×40),
27.9k H_t rows / ≥13.8k concat rows / 2.4k+ draft rows per layer per shard,
seed-0 reservoir 20k. Hook map: `tables/tensor_hook_map.md` (file:line for
every tensor; `identical_kv_input=true` verified at runtime in every shard).

## §28 answers

**1. H_t before R_C** — RMS 0.94, absmax 26.5 (weighted across datasets;
mtbench shard 27.1), |x| p99.9 = 5.98, max/RMS 26.9, kurtosis 66.2.
Z_t (pre-RMSNorm fc output) is far wilder: absmax 1552, kurtosis 714 —
hidden_norm tames scale but not shape. Channel structure: the top 0.1 % of
channels carry **20.8 %** of squared energy (gini 0.335).

**2. After R_C (= R_T)** — absmax 5.69 (−79 %), p99.9 3.27 (−45 %),
max/RMS 5.77 (−79 %), kurtosis 3.06 (−95 %: essentially Gaussian).
Top-0.1 % channel energy 20.8 % → **0.26 %** (gini 0.158): the rotation
does exactly what it should — it destroys channel-outlier concentration.

**3. Per-token** — row-absmax median 20.4 → 3.7; row-range p99 45.1 → 8.6.
The difficulty was within *every* token, not a few bad tokens.

**4. A4 scale** — per-token scale median 2.33 → 0.476 (−80 %); the grid
step shrinks 5× at equal bit-width.

**5. Zero-point** — distribution centered near mid-range in both, mildly
narrower after R_C (histograms in plots/qparams/). For this asymmetric
quantizer zero-int-code == exact-zero-dequant (verified equal: 0.8077 both).

**6. Code utilization** — zero-code ratio 80.8 % → 18.8 %; code entropy
1.46 → 3.18 bits of 4 (mean unique codes per token rises accordingly);
saturation ≈ 0 in both (per-token asym absorbs range).

**7. Activation NMSE** — A4(H_t) NMSE 0.405 → **0.019** (−95.3 %).

**8. qparams shared?** — **No.** ctx and draft branches are quantized in
separate calls with per-token parameters (by construction in both QLinear
and RCContextKV paths; verified per call site). H0 is dead on arrival.

**9. Independent qparams, yet ctx harder?** — Yes = H1. Same weights, own
qparams: ctx K AW-NMSE 0.089-0.099 vs draft-side K 0.035-0.222. The ctx
error is *uniform* across layers (one shared H_t, no per-layer norm) while
draft-side error is layer-idiosyncratic. Within-token dynamic range
(row absmax/RMS ~22 vs H_d ~... see tables) is the driver, not scale
sharing.

**10. Worst layer gap** — K: l0 and l2 (ctx/draft ratio 2.55× and 2.46×);
l1 inverts (0.45×) because that layer's own draft input is unusually hard
(V_draft l1 NMSE 0.586). The ctx bottleneck claim is about the *injected
context path being uniformly bad + persistent*, not about every layer
losing to its draft branch.

**11. K vs V** — V is ~2.3× more sensitive than K on the ctx branch
(V 0.196-0.292 vs K 0.089-0.099); K additionally passes k_norm+RoPE which
partially re-normalizes. R_C fixes both by ~10× (K → 0.006-0.009,
V → 0.019-0.031).

**12. Activation vs weight contribution of R_C** — decomposition (l0 K,
deployed policies): A-only 0.0793 → 0.0033 (**24×, ~96 % of the gain**),
W-only 0.0075 → 0.0043, interaction ≈ +0.002/-0.001. The RTN-vs-MSEclip
policy difference is negligible (AW 0.0888 vs 0.0851 control), so the
prior study's clip-search asymmetry did not confound the R_C effect.

**13. W_c branch imbalance** — block RMS [0.142, 0.099, 0.118, 0.076,
0.061], ratio 2.33× (deeper source ⇒ smaller block), confirming the
headline study; W4 handles it via per-row scales (W_c W4 NMSE small
relative to the activation term).

**14. K/V weight outliers** — none to speak of: k_proj kurtosis 3.5,
absmax 1.21 — the weights are benign; the ctx problem is activation-borne.

**15. R_C weight fold effect on W4** — mildly positive: k_proj view absmax
1.21 → 0.73, RTN W4 NMSE 0.0289 → 0.0247. The fold does not damage the
weight grid.

**16. K/V projection output error** — deployed: K 0.089-0.099 → 0.006-
0.009 (−92 %), V 0.196-0.292 → 0.019-0.031 (−90 %).

**17. Error at the BF16-attention boundary** — with everything downstream
of the projections in bf16 (all arms), post-norm+RoPE ctx-K NMSE 0.115 →
0.008, attention-score NMSE 0.054 → 0.011, softmax KL 0.246 → 0.070,
top-1 attended-position agreement 60.1 % → 82.0 %. The damage is fully
formed *before* attention; bf16 attention merely transports it.

**18. Verdict** — yes: the tensor evidence supports "DFlash target-context
injection quantization bottleneck" precisely. One outlier-concentrated,
kurtosis-66 fused context tensor is quantized at 4 bits into an effective
~1.5-bit code book, injected into every layer's K/V, and cached; rotating
it into the R_T basis Gaussianizes it (kurtosis 3.06), recovers 3.2 bits
of effective code use, and removes ~95 % of the activation error — which
propagates to ~90 % lower K/V output error and a 22-point gain in top-1
attention agreement.

## Headline figures

FIG-1/2 concat before/after R_T · FIG-3/4 H_t before/after R_C (same
z-axis, + `_autoscale`, + heatmaps) · FIG-5 per-channel absmax/RMS/sorted
curves · FIG-6 per-token scale/zero/NMSE · FIG-7 per-layer K/V NMSE (ctx
RC-off / RC-on / draft) · FIG-8 + `DFLASH_WC_WEIGHT_3D_*` (stock/folded ×
fp/W4, source-block separators) · `DFLASH_CTX_VS_DRAFT_ACT_3D_layer{0..4}`
(ctx 0-4095 | draft 4096-8191 split panels; caption notes independent
qparams) · representative tokens (median/p99/worst by pre-registered rule)
in plots/qparams/token_*.png.

## Interpretation guardrails honored (§27)

Per-token qparams are independent (measured); aggregate absmax was never
used as a scale; zero-code was cross-checked with entropy/NMSE/unique-codes;
the bottleneck is projection quantization, not KV-cache storage precision
and not bf16 attention (both explicitly measured); H_t's absmax 26.5 is a
new post-W_c/RMSNorm representation, not the raw target outlier (324)
passed through; R_C's gain is decomposed (96 % activation-side, weight fold
mildly positive); C4 (learned R_C) captures exist for parity with the AL
study — no claim of superiority over R_T reuse is made.
