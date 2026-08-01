# EP3-P Projection Geometry Visualization

**Run**: see `runs/EP3PVIZ_RUN_DIR` (`runs/ep3p_projection_visualization_20260801_104655`)
**Source study**: commit `d1cefba` (Generic-QAT / EP3-P / RCAL study);
no training, no checkpoint changes, no synthetic data — every panel is
rendered from tensors measured on the validated W4A4-target EP3-P
deployment.

## 1. What is being visualized

EAGLE-1's draft feeds its feature-fusion projection with the
concatenation of a token embedding (e) and a hidden feature (h). The
completed study showed the raw e-slice is ~2 orders of magnitude
smaller than the h-slice, which wrecks W4A4 quantization. EP3-P
(exponent P3, pathwise) migrates scale from the weight into the
activation with path-specific factors:

    m_first     = D ** beta_first     = 4096 ** 0.40 = 27.8576
    m_recurrent = D ** beta_recurrent = 4096 ** 0.45 = 42.2243

    e' = m * e            (activation, embedding slice only)
    W_e' = W_e / m        (weight, embedding-side input columns only)
    h, W_h unchanged

The product W_e'·e' is unchanged, so the FP function is preserved;
only the *numerical geometry* that the 4-bit quantizers see changes.

## 2. Why one complete projection

The projection is a single linear layer consuming one 8192-channel
input. Its quantizers act on the *complete* tensor: the per-token A4
quantizer computes one scale across all 8192 input channels, and the
per-output-channel W4 quantizer sees each full 8192-wide row. Plotting
W_e and W_h separately would hide exactly the phenomenon that matters —
the scale disparity *inside one quantization group*. Every main figure
therefore shows one continuous surface with a marked boundary plane at
input channel D = 4096:

    channels [0, 4096)    = embedding input region
    channels [4096, 8192) = hidden-feature input region

## 3. Axes and orientation (audited)

The repository stores the projection weight in PyTorch `nn.Linear`
convention, `[out_channel, in_channel] = [4096, 8192]`, and the forward
is `Y = X @ W.T` (+ bias; the fold carries a bias, which is not
migrated — it cancels in every before/after comparison and is included
in all output computations). Audit lives in
`metadata/collection_manifest.json`.

- Activation 3D: x = projection input channel (0–8191), y = token
  index, z = absolute activation value.
- Weight 3D: x = inner/input channel (0–8191), y = outer/output channel
  (0–4095), z = absolute weight value.
- Output 3D: x = output channel, y = token, z = absolute value/error.

Grouped mode (recorded in `metadata/visualization_manifest.json`):
activations use 16-channel groups (mean abs; every collected token
retained), weights use 16×16 input×output blocks (mean abs). Raw
tensors are stored losslessly in `plot_data/ep3p_tensors.pt`; each
figure's exact plotted matrix is in `plot_data/<figure>.npz` with a
metadata JSON (shape, grouping, aggregation, scale, color-norm range,
betas, m, path, depth, SHAs).

## 4. Data provenance

- Deployed system: W4A4 SpinQuant target (`learned_chat_w4a4kv16`),
  EP3-P draft adapter (`gamma_R1` first interface; recurrent hidden is
  the draft's own hidden state — **no** target final-RMSNorm gamma),
  the exact validated interface of the completed study.
- Calibration: the same held-out manifest as the beta search — C4
  calib pool (offset-500, disjoint from all test sets), 16 prompts;
  prompt IDs recorded in the manifest.
- Collection point: the split-projection forward input, *before* the
  recurrent e-slice rescale, so the shared embedding table's m_first
  factor can be divided out to recover the RAW e for both paths.
  Deterministic prefix sampling — no random subsampling.
- Recurrent depth: EAGLE-1's tree draft runs one recurrent projection
  forward per tree depth; depth = recurrent-call count since the last
  first-path call. Depths 1–4 are kept separate; the aggregated
  figures concatenate tokens in depth order with annotated depth
  boundaries (no depth-destroying averaging).
- Shapes collected: first `[4096, 8192]`; recurrent depth 1/2
  `[924, 8192]`, depth 3/4 `[231, 8192]` (tree width shrinks with
  depth); weight `[4096, 8192]` fp32 unmigrated fold.

## 5. What the figures show

- `activations/first_projection_input_before_3d` — the embedding
  region is a barely visible blue floor next to the hidden region: the
  structural imbalance.
- `..._after_3d` (same color normalization) — the embedding region
  rises by ×27.86 (first) / ×42.22 (recurrent); the hidden region is
  pixel-identical. The `difference` surface is exactly 0 over the
  hidden region; the `ratio` surface is ≈ m over the embedding region
  and ≈ 1 over the hidden region.
- `weights/projection_weight_*` — the mirror image: the embedding-side
  input columns of the ONE weight matrix drop by 1/m, hidden columns
  untouched. First-vs-recurrent shows the two migration strengths side
  by side under one normalization.
- `outputs/*` — FP outputs before vs after are visually identical;
  the error surface is at fp32-roundoff level (numbers in §6).
- `quantized_effect/*` — explicitly labeled W4A4 comparison (never
  mixed with the FP-equivalence panels): abs(Y_q − Y_ref) for naive
  W4A4 vs EP3-P W4A4 on identical scales.
- `summaries/` — four publication figures assembling the measured
  panels: per-path mechanism, first-vs-recurrent, and the end-to-end
  mechanism chain (embedding activation ↑, embedding-side weight ↓,
  hidden unchanged, FP output unchanged, W4A4 error ↓).

## 6. Numerical results (filled from tables/)

All measured (tables/ep3p_viz_summary.json; per-region detail in the
four CSVs):

| quantity | first path | recurrent path |
|---|---|---|
| activation e/h RMS ratio, before | 0.01274 | 0.00887 |
| activation e/h RMS ratio, after | 0.35503 (= ×27.86) | 0.37470 (= ×42.22) |
| FP output max abs error | 1.91e-6 | 9.54e-7 |
| FP output NMSE / cosine | 5.4e-14 / 1.0 | 1.1e-14 / 1.0 |
| A4 NMSE before → after | 0.0191 → 0.0319 | 0.0196 → 0.0331 |
| W4 NMSE before → after | 0.0217 → 0.0196 | 0.0217 → 0.0204 |
| **projection-output W4A4 NMSE, naive → EP3-P** | **0.7560 → 0.0415 (18.2×)** | **0.5833 → 0.0330 (17.7×)** |

Weight e/h RMS ratio (one matrix, region-wise): 7.654 before →
0.2748 (first migration) / 0.1813 (recurrent migration).

Reading: per-tensor A4/W4 NMSE barely moves (A4 even rises slightly —
the migrated e-slice now actually participates in the per-token scale),
but the *output* NMSE collapses by ~18x. The naive fold quantizes the
e-slice to near-annihilation (its contribution sits below the A4/W4
step size next to the h-slice); migration moves it into the
quantizers' dynamic range. That is the entire EP3-P mechanism, and it
is exactly what the quantized-effect figures show. FP outputs are
bit-identical up to fp32 roundoff (fp64 test: relative < 1e-12).

## 7. Why first and recurrent use different factors

The two paths feed the same projection with differently distributed
inputs: the first path's h comes from the (rotated, gamma-folded) W4A4
target, the recurrent path's h is the draft's own fp16 hidden state.
Measured e/h RMS ratios differ (~0.0127 vs ~0.0089 under the int4
interface), so the NMSE-optimal migration differs (beta 0.40 vs 0.45).
EP3-P implements this with one shared embedding table (carrying
m_first) plus a single elementwise rescale (m_rec/m_first) on the
recurrent e-slice — measured at 1.001× runtime, +0 MiB in the source
study.

## 8. Limitations of 3D visualization — and the companions

3D surfaces communicate the *shape* of the geometry but suffer
occlusion, camera dependence, and z-scale compression by outliers.
Therefore every 3D figure ships with:
- a **2D heatmap companion** (exact same matrix and color
  normalization — reliable channel/token localization),
- a **log10 companion** (`log_scale/`, explicitly labeled; never a
  replacement for the linear primary) because the pre-migration
  embedding region is invisible on a linear scale next to the hidden
  region — which is precisely the point being made,
- the exact plotted matrix (NPZ) + metadata for regeneration, verified
  end-to-end by `scripts/verify_ep3p_visualization_data.py` and the
  7 contract tests.

Grouped rendering (16-channel / 16×16 mean-abs) is an *explicit,
recorded* aggregation, not silent downsampling; full-resolution
statistics are computed on ungrouped tensors.

## 9. Artifacts

- Scripts: `collect_ep3p_projection_tensors.py`,
  `plot_ep3p_projection_3d.py`, `plot_ep3p_projection_summary.py`,
  `verify_ep3p_visualization_data.py`
- Tests: 7 `tests/test_ep3p_*.py` (orientation, boundary, migration
  values ×2, hidden-unchanged, FP equivalence, reproducibility)
- Bundle: `ep3p_projection_visualization_<ts>.tar.gz` + `_latest`
