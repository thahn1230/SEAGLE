# EAGLE draft — WEIGHT + ACTIVATION distribution visualizations (descriptive config names)

Every figure explicitly states whether it shows an **ACTIVATION** (pre-layer input
captured during real EAGLE tree generation), a **WEIGHT** (the weight actually used
by the draft module), or a **WEIGHT QUANTIZATION ERROR** (w_q − w_fp). No C2/C3/C4
names appear in any output; the in-script quality checks (`quality_checks.txt`)
and an independent path scan both PASS.

- Run dir: `runs/eagle_draft_weight_activation_3d_20260711_1127/`
- Model: Llama-2-7b-chat target (fake W4A4) + EAGLE-llama2-chat-7B draft. Greedy. GPU 4 only.
- Configs (used verbatim in every filename, title, CSV row):
  - `target_W4A4__draft_FP16_pureR1` — healthy reference (scan mean 3.02)
  - `target_W4A4__draft_W4A4_pureR1` — collapse case (scan mean 1.016)
  - `target_W4A4__draft_W8A8_pureR1` — recovery case (scan mean 2.225)
- Representative case: **prompt_id 83, verification cycle 1** — acceptance
  FP16 **4.33** / W4A4 **1.00** / W8A8 **3.50** (healthy + collapsed + recovered,
  largest FP16−W4A4 gap of 8 candidates; cycle 1 = earliest cycle where all
  configs start from the identical target hidden).
- Depths: 0–4 captured (depth 0 = first draft forward from target hidden h_R;
  depth k = k-th recurrent forward from f_R).
- 431 activation figures + 116 weight/quant-error figures + 11 stats CSVs.
- Comparability caveat (also applies to the error CSVs): only **depth-0** rows are
  strict apples-to-apples across configs (identical input, verified cosine 1.0000
  on the fc input); at depth ≥ 1 the quantized drafts have already proposed
  different tree tokens, so rel-L2/cosine there conflate token divergence with
  quantization error. All headline comparisons below use depth 0.

## Why the collapse / recovery (one paragraph, figure-backed)
The draft's **as-used weights are benign** after R1 — near-Gaussian, channel
ratios ≤ 2.7 — so 4-bit weight RTN produces a *uniform* ~0.11 rel-L2 error
(fc worst, 0.131). The draft's **activations are not benign**: the fc concat
input has a 126× channel absmax ratio (embedding-branch scale mismatch) and the
MLP intermediate 32×. At 4-bit, weight+activation error compounds inside the
single draft layer and destroys the feature EAGLE verifies against
(depth-0 final-feature cosine 0.116 vs fp16 draft) — acceptance 1.0. At 8-bit,
weight error drops ~10× (fc 0.0137, SNR 17.6→37.3 dB) and every intermediate
activation stays close to fp16 (q/k/v 0.971, gate/up 0.932, final feature 0.856)
— acceptance recovers to 3.50 on the representative prompt (2.225 mean).

## The 12 required answers

**1. Are activation plots and weight plots separated?**
YES. Activation figures live only under `figures_activation_3d/`,
`figures_activation_heatmap/`, `figures_activation_channel_curves/`,
`figures_comparison_activation/`; weight and quant-error figures only under
`figures_weight_3d/`, `figures_weight_heatmap/`, `figures_weight_channel_curves/`,
`figures_comparison_weight/`. Every activation filename contains `activation`,
every weight filename contains `weight`; `figure_manifest.csv` carries
`tensor_kind` ∈ {activation, weight, quantization_error} per figure. No figure
mixes the two; no shared filename stems.

**2. Do all filenames and titles use descriptive names instead of C2/C3/C4?**
YES. Config directories/titles/CSV `config_name` values are exactly
`target_W4A4__draft_FP16_pureR1`, `target_W4A4__draft_W4A4_pureR1`,
`target_W4A4__draft_W8A8_pureR1`. The in-script quality check walks every output
path and asserts the substrings C2/C3/C4 are absent — PASS (independently
re-checked with `find`: 0 hits).

**3. Which activation tensors were captured and where?** (all from real EAGLE
generation; forward-hook input[0] unless stated)
| activation type | capture point |
|---|---|
| embedding_output_activation | `embed_tokens` **output** (e_R = e@R1) |
| projection_input_activation | fc input = concat(embeds, hidden) ([cnets.py:593]) |
| projection_embedding_branch_activation | fc input `[..., :4096]` |
| projection_hidden_branch_activation | fc input `[..., 4096:]` (h_R at depth 0 / f_R recurrent) |
| q/k/v_proj_input_activation | q/k/v input = raw fc output (draft layer 0 has no input_layernorm) |
| o_proj_input_activation | o input = attention output (R2-rotated in quant configs → flagged basis-mismatched) |
| up/gate_proj_input_activation | post_attention_layernorm output |
| down_proj_input_activation | MLP intermediate, **pre** online-R4-Hadamard (basis-matched across configs) |
| down_proj_input_activation_post_R4 | after `matmul_hadU_cuda` (quantized configs only; 13th type) |
| lm_head_input_activation | head pre-hook = final draft feature f_R |

**4. Which weight tensors were captured and where?**
For each config, the 10 as-used draft weights: `projection_fc_weight` (ea_layer.fc),
`q/k/v/o_proj_weight`, `up/gate/down_proj_weight` (draft layer 0),
`embedding_weight` (embed_tokens, rows@R1), `lm_head_weight` (W_lm@R1).
For the quantized configs, each of the **8 fake-quantized linears** was captured
as both `w_fp` (R1/R2/R4-conjugated pre-quant weight, from the adapter's
`_replaced` originals) and `w_q` (`FakeW4A4Linear.w_fake`, the fake-quantized
weight actually used), plus the error `w_q − w_fp`. Embedding/lm_head are fp16
in ALL configs (never fake-quantized) and are labeled so.

**5. Which representative prompt/cycle was selected and why?**
**prompt_id 83, cycle 1.** Scan over 8 MT-bench prompts × 3 configs; selection
rule = FP16 healthy (4.33 ≥ 2.0), W4A4 collapsed (1.00 ≤ 1.6), W8A8 recovered
(3.50 ≥ 1.00 + 0.3), maximizing the FP16−W4A4 gap (3.33). Cycle 1 is the
earliest cycle where all three configs receive the identical target hidden
(depth-0 fc-input cosine = 1.0000 across configs — the built-in sanity check).

**6. In `target_W4A4__draft_W4A4_pureR1`, which activation is most distorted at depth 0?**
The **final draft feature (`lm_head_input_activation`): cosine 0.116 / rel-L2 1.30**
vs the fp16 draft — with an identical input (fc input cosine 1.0000). Next worst:
`down_proj_input` 0.214, `q/k/v_proj_input` 0.320, `gate/up_proj_input` 0.546.
Error is spread across channels (top-10-channel error share <1%), i.e. the whole
feature geometry is scrambled, not a few channels.

**7. Which weight has the largest quantization error?**
**`projection_fc_weight`: rel-L2 0.1312 (SNR 17.6 dB)** — the 2D→D projection.
The other seven linears cluster at 0.110–0.115 (o_proj 0.115 second).
Weight-side error is *uniform* — no single catastrophic weight — consistent with
the flat post-R1 weight distributions (channel ratios ≤ 2.7).

**8. Does W8A8 visibly recover activation structure vs W4A4?**
YES. Depth-0 cosine vs the fp16 draft: final feature **0.116 → 0.856**,
q/k/v 0.320 → 0.971, gate/up 0.546 → 0.932, down 0.214 → 0.834. Visually
(`figures_comparison_activation/depth0/…heatmap_overlay.png`, shared scale):
the FP16 panel's channel structure is flattened/scrambled in the W4A4 panel and
restored in the W8A8 panel.

**9. Does W8A8 visibly reduce weight quantization error vs W4A4?**
YES — ~10× on every module: fc 0.1312 → 0.0137, others 0.110 → 0.0085–0.0091;
SNR +20 dB (17.6→37.3 fc). The quant-error heatmaps
(`figures_weight_heatmap/<config>/<w>__weight_quant_error__heatmap.png`) show the
same spatially-uniform error at ~1/10 the magnitude.

**10. Which layer should be kept higher precision based on ACTIVATION plots?**
The **fc projection input path** (126× channel ratio — embedding-branch scale
mismatch) and the **MLP `down_proj` intermediate** (32× ratio, heavy tail), plus
**o_proj** (attention output; near-zero cosine even at W8A8 — the most
activation-fragile tensor). These are the activation-quantization hotspots R1
does not flatten.

**11. Which layer should be kept higher precision based on WEIGHT plots?**
**`projection_fc_weight`** — largest quant error (0.131) and the largest
input-channel absmax ratio (2.74; its embedding-branch columns carry larger
weights to compensate the small embedding activations, visible in
`projection_fc_weight__weight__3d.png`). Everything else is weight-flat; weights
alone do not justify mixed precision beyond fc.

**12. Recommended next precision experiment?**
**W8A8 → real INT8** as the baseline (both weight error ~0.009 and activation
structure recover, acceptance 1.0 → 2.2–3.5). Then a **mixed-precision probe**:
keep the fc projection (worst weight AND worst activation input) and o_proj
(activation-fragile even at 8-bit) at W8A16/fp16 while testing lower bits
elsewhere; keep R4's online Hadamard on `down_proj`. **W4A16 is not indicated** —
weights are flat/easy and the observed damage is dominated by activation-side
outliers.

## Figure inventory
- activation 3D / heatmap: `figures_activation_3d|_heatmap/<config>/depth<d>/<type>__pre_layer_activation__{3d|heatmap}__prompt83_cycle01_depth0N.png`
- activation channel curves (depth 0) + comparisons: `figures_activation_channel_curves/`, `figures_comparison_activation/depth0/`
- weight 3D / heatmap / curves: `figures_weight_3d|_heatmap|_channel_curves/<config>/<w>__weight__*.png`
  (+ `__weight_quant_error__heatmap.png` for the 8 quantized linears per quant config)
- weight comparisons: `figures_comparison_weight/<w>__draftFP16_vs_draftW4A4_vs_draftW8A8__weight_heatmap_overlay.png`
- every figure indexed in `stats/figure_manifest.csv` (tensor_kind, config_name, layer_name, figure_type, path, sampling_rule)
