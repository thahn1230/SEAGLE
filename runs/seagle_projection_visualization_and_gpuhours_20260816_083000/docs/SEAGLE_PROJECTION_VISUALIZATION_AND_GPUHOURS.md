# SEAGLE Projection-Layer Visualization & GPU-Hours Accounting (2026-08-16)

Analysis run: `runs/seagle_projection_visualization_and_gpuhours_20260816_083000`
All tensors are REAL tensors from the canonical deployed configurations,
captured during actual MT-Bench generation replay (no toy data).

## Section 1. Canonical method mapping

See `tables/canonical_method_mapping.csv`. Summary:

| Public name | Internal canonical config | Headline |
|---|---|---|
| SEAGLE-PTQ | GS+R5 PTQ: public draft, D4P3 fold, analytic GS scale (beta=0.42 -> alpha=32.8996), R5=RD_HYB_s2, frozen weights | mean4 3.4788 |
| SEAGLE-QAT | CAN_T6_gsr5_conv: plain chain QAT, LR 1e-6, 3000 steps, 3 seeds, calib-selected cadence; GS+R5 deploy | mean4 3.6108 (seed mean) |
| SEAGLE-RT  | Official-recipe from-scratch retraining (fresh init, 21 epochs, 43.9k steps) + W4A4-target D4P3 deploy (gamma o R_T fold, alpha=32.0) | FP16 3.5341; W4A4 deploy 2.9666 mtbench (+Q1 QAT 3.29-3.33) |

**Mapping resolution (explicit, per task rule).** The repository's only
completed "trained from the beginning" draft is the official-recipe
from-scratch reproduction; its scratch phase consumes the FP16 target
hidden per the official recipe, and the rotated/native target feature
is consumed at deployment through the exact gamma o R_T fold (the
native rotated interface). No completed run trains a draft from
scratch directly on rotated features. The nearest variant — the July
study's C7 cell (full original recipe applied to the CONVERGED public
draft on the int4 interface) — is HARMFUL (-0.65 tau vs PTQ) and is
listed as an appendix variant, not the canonical RT.

## Section 2. What tensors were visualized

For each method, from the actual deployed adapter during MT-Bench
generation (12 prompts, greedy, mc_sim_7b_63):
- first / recurrent projection INPUT activations z = cat([embedding-side
  (4096 ch) | hidden-side (4096 ch)]), deployed ("after") form captured
  at the projection module input;
- analytic ORIGINAL views: e-half unscaled by alpha; h-half restored to
  the stock basis (first: h_t=(a_t R^T) o gamma_f; recurrent: h R_used^T);
- projection weights: original fc halves [W_e | W_h] from the method's
  own state dict, and the deployed pre-quant folded W_first / W_rec
  captured via the weight-quantizer spy.

## Section 3. Selection rule for activation windows

Median-energy contiguous 128-row window over up to 4096 captured rows
per path (rule + prompt IDs + window indices saved in
`artifacts/<method>_manifest.json`). No cherry-picking: the window is
the MEDIAN-energy one, not best/worst.

## Section 4. Figures

`plots/` — FIG1/2/3 per method (first + recurrent) x (activation,
weight) x (original, after) x (same-z, autoscale), PNG+PDF, 300 dpi,
coolwarm (blue=small, red=large), black translucent wall = the
embedding|hidden channel boundary at channel 4096. FIG4/FIG5:
three-method side-by-side (first-path activation/weight, original,
same-z). FIG6: summary-stat bars. Max-abs pooling (act 1x16,
weight 32x16) preserves outlier structure; factors in
`artifacts/plot_metadata.json`.

## Section 5. Summary stats (tables/projection_stats.csv)

First-projection input, ORIGINAL basis (the geometry every method
inherits):

| method | RMS | absmax | kurtosis | top-0.1% ch energy |
|---|---|---|---|---|
| SEAGLE-PTQ | 1.274 | 73.9 | 83.4 | 11.5% |
| SEAGLE-QAT | 1.274 | 77.4 | 66.2 | 8.2% |
| SEAGLE-RT  | 1.269 | 77.4 | 63.6 | 8.7% |

After the deployed treatment (alpha migration + rotation fold), all
three flatten to absmax ~5, kurtosis ~1.5, top-0.1% ~0.6%. The
embedding-side block sits orders of magnitude below the hidden-side
spikes in the original basis (the boundary wall in every "original"
panel) — the heterogeneous two-block geometry that makes naive W4A4
collapse (tau 1.05-1.28 across studies).

## Section 6. GPU-hours methodology

GPU-hours = wall-clock x GPUs occupied, summed per job. Sources, in
priority order: (i) in-log elapsed prints of the chain trainer
([lk] ... (Ns)) — MEASURED; (ii) per-prompt gen_seconds recorded in
every evaluation shard + 240 s model-load per job — MEASURED;
(iii) study-report cost sections (from-scratch study section 11) —
DOCUMENTED; (iv) step-count x measured same-trainer step rate on
identical hardware, or setup-based bounds — ESTIMATE (flagged). No
unflagged guesses. Old-server logs are not present on this machine;
everything not measurable here is tied to a documented source.

**Counting policy.** SEAGLE-PTQ = scale-selection grid + fold/deploy
(+ its rotation artifacts). SEAGLE-QAT = the 3 seed trainings + the
cadence-selection calibs that define the defended checkpoints (+ the
same rotation artifacts; LR-frontier and anchored-QAT research are
excluded). SEAGLE-RT = the documented from-scratch reproduction +
alpha recalibration (+ shared target rotation); the optional Q1 QAT
arm and its selection are listed separately. Research studies that
merely USE these checkpoints (qanchor causal study, closure audit,
B-grids, LK arms) are excluded and listed in
`tables/gpuhours_excluded_jobs.csv`.

## Section 7. GPU-hours main table (tables/gpuhours_main.csv)

| Method | Core GPU-h (no rotation) | Rotation-matrix GPU-h | Total | Notes |
|---|---:|---:|---:|---|
| SEAGLE-PTQ | 3.0 | 11.8 | **14.8** | rotation = R_D 1.8 [EST] + shared R_T ~10 [EST] |
| SEAGLE-QAT | 8.0 | 11.8 | **19.8** | trainings 5.45 + selection 2.58 [MEASURED] |
| SEAGLE-RT  | 158.0 | 10.0 | **168.0** | 155+3 [DOCUMENTED]; +40 QAT arm +6 selection optional |

Rotation-matrix breakdown:

| Rotation job | GPUs | GPU-h | Used by | Provenance |
|---|---:|---:|---|---|
| Target R_T (SpinQuant, 100 steps) | 8 | ~10 [EST 8-12] | all three | trainer_state.json + setup bound |
| Draft R_D (RD_HYB_s2, 3000 steps) | 1 | ~1.8 [EST] | PTQ, QAT | measured same-trainer step rate |

## Section 8. Detailed accounting

`tables/gpuhours_detailed.csv` (substage rows with n_gpu, wall hours,
source flags) and `tables/gpuhours_excluded_jobs.csv`.

## Section 9. Interpretation

1. **How much does each method cost?** PTQ ~15 GPU-h total (only ~3
   outside rotation learning), QAT ~20, RT ~168 (~214 with its QAT
   arm).
2. **Rotation share:** For PTQ/QAT the rotation matrices are the
   DOMINANT cost (11.8 of 14.8 / 19.8); for RT the from-scratch
   training dwarfs rotation (158 vs 10).
3. **RT vs PTQ: ~11x more total GPU-h (52x core-only).**
4. **RT vs QAT: ~8.5x more total GPU-h (20x core-only).**
5. **Quality per compute:** SEAGLE-QAT attains the best acceptance
   (3.61) at ~20 GPU-h; SEAGLE-RT's W4A4 deployment (2.97 mtbench,
   ~3.3 with its QAT arm) does NOT exceed the cheaper methods under
   this deployment contract. Stated carefully: the RT numbers are
   mtbench-only from its study, and RT serves a different purpose
   (recipe reproduction) — but nothing in the data suggests scratch
   retraining buys quantization robustness.
6. **Why the projection stays sensitive (the figures' message):**
   the projection input is a concatenation of two statistically
   incompatible blocks — a small-scale embedding block and a
   heavy-tailed target-hidden block (original kurtosis 64-83, absmax
   ~74-77, top-0.1% channels carrying 8-12% of energy). This geometry
   comes from the TARGET's hidden feature, so retraining the draft
   from the beginning (SEAGLE-RT) inherits it unchanged (RT original
   stats ~= PTQ's). RT fixes interface SEMANTICS (the draft learns to
   read the feature) but not quantization GEOMETRY (the feature's
   two-block heterogeneity); only the deployment-side treatment
   (alpha migration + rotation) flattens it (kurtosis 83 -> 1.5,
   absmax 74 -> 5).

## Section 10. Caveats

- All quantization is fake-quant; no real-INT4 kernel timings.
- R_T / R_D / scale-grid GPU-h are flagged ESTIMATES (original-server
  logs not preserved here); measured and documented rows are flagged.
- SEAGLE-RT figures use the fresh anchor deployed through the same
  D4P3 adapter (its study's W4A4 arm); its "after" is the deployed
  form (there is no separate learned treatment beyond the fold).
- Activations from 12 MT-Bench prompts (manifest recorded); the
  median-energy window rule avoids cherry-picking but is one window.
- RT headline W4A4 numbers are mtbench-only (its study's contract).
