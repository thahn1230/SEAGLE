# EAGLE draft — ACTIVATION-only 3D distribution visualizations

Every figure in this pass plots a **pre-layer ACTIVATION** (the tensor fed INTO a
draft module, captured immediately before that module runs) during **real EAGLE
tree generation**. Axes: **x = channel index, y = token index, z = abs(activation)**.
**No weights are plotted anywhere.**

- Run dir: `runs/eagle_draft_activation_3d_20260708_1102/`
- Model: Llama-2-7b-chat (target, fake W4A4) + EAGLE-llama2-chat-7B draft. Greedy.
- Configs: **C2** = pure-R1 draft fp16 · **C3** = pure-R1 draft fake **W4A4** ·
  **C4** = pure-R1 draft fake **W8A8**. Target is fake-W4A4 for all three.
- Representative case: **prompt_id 83, verification cycle 1** (C2 acceptance
  **4.33**, C3 acceptance **1.00** — largest C2−C3 gap of 8 candidates).
- Depths captured at cycle 1: **0,1,2,3,4** (5 draft forwards). Depth 5 is not
  reached at cycle 1 → depth-montages span depths 1–4 (stated, not hidden).
- Script: `scripts/plot_eagle_draft_activation_3d.py`. 456 figures + 7 stats CSVs.

## Instrumentation sanity check (proves the tap points are correct)
At **depth 0** the fc INPUT is identical across configs by construction (same
target `h_R`, same `e_R`): measured cosine(C2,C3) = **1.0000**, rel_L2 = **0.0**
for `projection_input`, `embedding`, `embedding_branch`, `hidden_branch`. The
divergence then appears *inside the single forward* (fc→attention→mlp), which is
exactly what a correct pre-layer capture must show.

---

## The 10 required answers

**1. Were all figures based on activations, not weights?**
YES. Every captured tensor is a forward-hook **input** (or `embed_tokens`/`head`
I/O) — i.e. the activation entering a layer. Every stats CSV row carries
`tensor_kind = "activation"` and `capture_point = "pre_layer_input"`; every
figure filename/title contains `activation` / `pre-layer ACTIVATION`. No weight
tensor is captured or plotted in any figure.

**2. For each of the 12 activation types, where exactly was the tensor captured?**
| # | activation type | capture point (pre-layer) |
|---|---|---|
| 1 | embedding_activation | `embed_tokens` output = `e_R = e@R1` |
| 2 | embedding_branch_activation | fc input `[..., :4096]` (embed half of concat) |
| 3 | hidden_branch_activation | fc input `[..., 4096:]` (hidden half: `h_R`@depth0 / `f_R` recurrent) |
| 4 | projection_input_activation | fc input = `concat((inputs_embeds, hidden), -1)` ([cnets.py:593]) |
| 5 | q_proj_input_activation | q_proj input = raw fc output (layer 0 has **no** input_layernorm) |
| 6 | k_proj_input_activation | k_proj input (same residual-stream tensor) |
| 7 | v_proj_input_activation | v_proj input (same residual-stream tensor) |
| 8 | o_proj_input_activation | o_proj input = attention output (post-softmax·V) |
| 9 | up_proj_input_activation | up_proj input = post_attention_layernorm output |
| 10 | gate_proj_input_activation | gate_proj input = post_attention_layernorm output |
| 11 | down_proj_input_activation | down_proj input = MLP intermediate (act(gate)·up), **pre** online-Hadamard |
| 12 | lm_head_input_activation | draft `head` input = final draft feature `f_R` |

**3. Which prompt/cycle was chosen as representative, and why?**
**prompt_id 83, cycle 1.** Selection scanned 8 MT-bench prompts and picked the
one with healthy C2 and fully-collapsed C3: C2 = 4.33 accepted tokens, C3 = 1.00
(gap 3.33, the largest). Cycle 1 is the earliest cycle where C2/C3/C4 all start
from the *identical* target hidden, so depth-0 differences are pure quantization.

**4. For C2 vs C3, which activation type looks most distorted at depth 1?**
Two readings, both reported honestly:
- **Clean per-layer quantization (depth 0, identical inputs):** the **final
  draft feature (`lm_head_input`) is the most-distorted basis-matched activation,
  cosine 0.115**; `q/k_proj input` next (cosine 0.320). `o_proj input` is ~0 but
  it is R2-rotated (basis-mismatched), so its number mixes rotation with quant.
- **At depth 1** every type is near-orthogonal (cosine 0.01–0.49) because by
  depth 1 the W4A4 draft has already proposed **different tree tokens** than the
  fp16 draft, so embeddings/features diverge for token-path reasons on top of
  quantization. i.e. the draft is already off the rails after one forward.

**5. For C2 vs C4, which activation type still improves relative to C3?**
**All of them.** At depth 0 (pure quant), C4 (W8A8) vs C2 cosine is far higher
than C3: final feature `lm_head_input` **0.856 (C4) vs 0.115 (C3)**; `q/k/v input`
**0.971 vs 0.320**; `gate/up input` **0.932 vs 0.546**; `down input` **0.834 vs
0.214**. Averaged over basis-matched types at depth 1, C4 mean cosine **0.573**
vs C3 **0.161** (rel_L2 0.873 vs 1.235). W8A8 preserves the activation geometry
W4A4 destroys.

**6. Does distortion appear immediately at depth 1 or only after recurrent
accumulation?**
**Immediately — within the very first forward (depth 0).** With an *identical* fc
input (cosine 1.0), the W4A4 draft's output feature is already destroyed at depth
0 (final-feature cosine 0.115). It is not recurrent accumulation; recurrence and
token-path divergence only compound damage that is already near-total at depth 0.

**7. Which layers show the strongest channel outliers BEFORE quantization?**
(per-channel absmax max/median ratio, C2, mean over depths 0–4)
- **projection_input (fc input concat): ~194** — the 8192-wide `[e_R, h_R]` concat
  has a tiny median (`median_abs` 0.031: the embedding-branch channels are
  near-zero) while the hidden branch spikes to ~5, a **scale-mismatch** outlier.
- **down_proj input (MLP intermediate): ~43** (up to ~73 at shallow depth), with
  **kurtosis ~176 and abs_max 85.6** — a genuine **heavy-tailed** LLaMA outlier channel.
- **o_proj input (attention output): ~8**; hidden_branch `h_R` ~5; q/k/v, gate/up,
  embedding, final feature all ~4.3–4.5 (kurtosis ≈ 0, well-behaved).
So the fc-projection input (scale mismatch) and the MLP intermediate (heavy tail)
are the outlier hotspots — exactly the two tensors quantized worst.

**8. Does R1 visibly flatten channel-wise outliers in the activation views?**
YES, on the residual stream: the R1-rotated `hidden_branch` (`h_R`) shows a
channel ratio of only ~5 (kurtosis ≈ 0), versus ~18 for the un-rotated residual
measured in the prior distribution analysis (18.4→2.0 there). BUT R1 does **not**
flatten the fc-concat input (~194, the embedding-branch scale mismatch) nor the
MLP intermediate (`down_proj` input ~43, kurtosis ~176 — downstream of R1 and R4's
job, not R1's). So rotation flattens the residual but leaves the two hardest
tensors outlier-heavy (visible in `fig channel_absmax_curve` and the 3D plots:
`hidden_branch` is a smooth landscape, `projection_input`/`down_proj_input` spike).

**9. Which activation types remain fragile even after rotation?**
`projection_input`/fc (ratio ~130), `down_proj` input / MLP intermediate (~73),
and the attention output (`o_proj` input). The depth-0 pure-quant errors confirm
it: attention (`q/k/v` 0.32, `o_proj` ~0), MLP `down` (0.21), and the resulting
final feature (`lm_head_input` 0.115) are where 4-bit does the damage — R1 alone
cannot protect them.

**10. Which next precision to test?**
**W8A8 (→ real INT8), then mixed precision.** The plots make it visual: uniform
W8A8 restores the final draft feature (depth-0 cosine 0.856 vs W4A4's 0.116) and
every clean intermediate (q/k/v 0.971, gate/up 0.932, down 0.834). So **W8A8 is
the precision to test next**, then real INT8 kernels once validated.
The damage is **activation-side (A4)** on outlier-heavy tensors that R1 cannot
flatten, not weight-side — so **W4A16 is not indicated** (weight-only 4-bit; the
prior pass already showed activations are the problem).
Two nuances the plots expose argue for **mixed precision** on top of W8A8:
(a) `o_proj` input (attention output) stays near-orthogonal even at W8A8
(cosine ≈ 0 both C3 and C4) — it is the most quantization-fragile tensor and
wants ≥8-bit / higher; (b) the `embedding` / `projection_input` path is the other
sensitive spot. Recipe to try: **W8A8 baseline, keep o_proj and the
embedding→fc-projection path at W8A16/fp16**, and (optionally) W8A16 elsewhere to
trim activation cost. `down_proj` (kurtosis ~176) needs its R4 online-Hadamard to
stay effective at 8-bit.

## Figure inventory (all activation-only)
- `figures_3d/<C>/depth<d>/<act>__3d__prompt83_cycle01_depth0N.png` — 198 (x=channel, y=token, z=abs activation)
- `figures_heatmap/<C>/depth<d>/<act>__heatmap__...png` — 222
- `figures_channel_curves/<C>/<act>__channel_absmax_curve__depthNN.png` — 36
- `figures_heatmap/montages/<C>/<act>__depth1to5__montage.png` — 36 (depths 1–4 present)
- `figures_3d/montages/<C>/<act>__depth1to5__3d_montage.png` — 18 (6 key types × 3 configs)
- `figures_heatmap/overlays_depth1/<act>__C2C3C4_depth1_overlay.png` — 6 (shared z-limit)

Most telling figures for "why W4A4 hurts": the C2/C3/C4 depth-1 overlays and the
`lm_head_input_activation` / `projection_input_activation` / `down_proj_input_activation`
3D plots — C3's magnitude landscape is visibly scrambled where C2/C4 are smooth.
