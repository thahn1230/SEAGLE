# Why draft fake W4A4 collapses EAGLE acceptance (3.0 → 1.0) — distribution analysis

Compares C2 (target W4A4 + pure-R1 draft **fp16**) vs C4 (target W4A4 + pure-R1
draft **fake W4A4**) on the REAL EAGLE draft generation path. Run dir
`runs/w4a4_distribution_analysis_<ts>_n20/`. 8 figures + 7 CSVs. **n=20 confirmed**
(depth-1 cosine 0.144, identical to the n=2 pilot).

## One-line finding
**W4A4 destroys the draft feature at the VERY FIRST forward** — depth-1
feature cosine(C4, C2) = **0.146** and the W4A4 draft's top-1 token agrees with
the fp16 draft only **1.1%** of the time. It is NOT gradual accumulation over
tree depth; the damage is immediate and near-total, because the 0.24B single-
layer draft has 8 quantized linears (fc weight rel-L2 0.13, q/k/v activation
rel-L2 0.16) whose errors compound within one forward. R1 does suppress channel
outliers (max/median absmax **18.4 → 2.0**, 9×), but 4-bit still wrecks the fine
feature geometry that EAGLE acceptance depends on.

## The 10 required answers (n=20)
1. **Largest W4A4 quant error modules?** Activation: attention `o/v/k_proj`
   (rel-L2 0.16-0.17, highest), then fc/mlp (~0.10). Weight: **fc** highest
   (rel-L2 0.131), others ~0.11. `quant_error_by_module.csv`, `weight_quant_error.csv`.
2. **Does R1 reduce channel outliers in draft activations?** YES — hidden
   per-channel max/median absmax ratio 18.4 (before R1) → 2.0 (after R1)
   (`fig02`, `per_channel_stats.csv`).
3. **Does fake W4A4 reintroduce large feature drift despite R1?** YES — depth-1
   final-feature cosine 0.146, rel-L2 1.28 (`fig04`, `feature_drift_by_depth.csv`).
4. **At which tree depth does C4 diverge from C2?** **Depth 1** (immediately) —
   cosine already 0.146, top-1 agreement 1.1%.
5. **Does feature error accumulate over recurrent depth?** NO — it is near-total
   at depth 1 and stays low (cos 0.15→0.07→0.04...); depths ≥2 are additionally
   confounded by token-path divergence. The collapse is a depth-1 phenomenon.
6. **Main suspect module for the collapse?** No single module — the whole
   1-layer draft is corrupted. Highest contributors: `fc` (largest weight error,
   the 2D→D projection) and `q/k/v_proj` (largest activation error).
7. **Weight, activation, or both?** BOTH. Per-tensor error is larger for
   ACTIVATION (q/k/v 0.16 vs weight 0.11); but for ACCEPTANCE the prior
   localization showed weight-only 4-bit (1.23) is marginally worse than
   activation-only (1.57) — each independently collapses acceptance.
8. **Does logit top-k agreement drop before acceptance collapses?** YES — depth-1
   top-1 agreement 1.1%, top-5 4%, top-10 5.4% (`fig05`, `logit_drift_by_depth.csv`).
   The draft proposes wrong tokens from the start.
9. **Does final-feature cosine correlate with accepted length?** YES — cosine
   0.146 (destroyed) ↔ C4 acceptance ~1.0; fp16 cosine 1.0 ↔ C2 acceptance ~3.0
   (`fig06`, `acceptance_vs_error.csv`).
10. **Which precision next?** **W8A8** (8-bit weight rel-L2 ≈0.008 vs 4-bit ≈0.13;
    already recovering acceptance to ~2.3–2.46 in the parallel W8A8 pass), or
    W8A16 / mixed (keep fc + attention higher precision). W4A16 (weight-only
    4-bit) is insufficient alone (acceptance ~1.23).

## Hypotheses
- **H1 (fc most sensitive)** — PARTIAL: fc has the highest WEIGHT error but q/k/v
  have higher ACTIVATION error; fc is a top suspect, not uniquely.
- **H2 (activation drift > weight drift)** — MIXED: activation per-tensor error is
  larger, but weight-only quant is marginally more harmful to acceptance.
- **H3 (error accumulates with depth)** — NOT SUPPORTED: damage is immediate at
  depth 1.
- **H4 (R1 reduces outliers but A4 destroys geometry)** — SUPPORTED: ratio
  18.4→2.0 yet cosine 0.146.
- **H5 (collapse correlates with cosine / top-k drop)** — SUPPORTED: cosine 0.146
  and top-1 1.1% coincide with acceptance ~1.0.

## Figures (`figures/`)
fig01 activation histograms · fig02 per-channel absmax (before/after R1/W4A4) ·
fig03 activation quant error by module · fig04 feature cosine by depth ·
fig05 top-k agreement by depth · fig06 acceptance vs feature error ·
fig07 layer-error heatmap · fig08 weight quant error by module.
