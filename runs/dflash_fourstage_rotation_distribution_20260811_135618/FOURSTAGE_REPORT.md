# Four-Stage Rotation Distribution Map — Final Report (§26 answers)

Run: this directory. Pipeline: session-autonomous scheduler DAG
(traj→capture×4→plots→finalize→verify→commit), verify gate
missing=0 / mismatch=0 (tables/verify_fourstage_output.txt). All numbers
recomputed from THIS run (pooled over mtbench/gsm8k/humaneval/sharegpt,
20 prompts each, frozen fp16 replay trajectories; seed 0; conditions
FP / QOFF(=M3) / QON(=M5a); basis contract per FIDI basis_audit verdict D).

1. **How severe are the five target-source hidden outliers before R1_T?**
   Extreme: concat[H1..H5] pooled absmax 326.0, kurtosis 171,433, A4 NMSE
   0.472 (Llama massive-activation channels; see FIG_A left / FIG_A2).

2. **How much does learned R1_T flatten them?**
   absmax 326→26.8 (12×), kurtosis 171,433→344 (500×), A4 NMSE 0.472→0.051
   (9×). Residual kurtosis is token-driven, not channel-driven (FIG_A2).

3. **How does the mandatory R1_T fold change W_c geometry?**
   Mildly: RMS unchanged (0.1034), absmax 2.078→1.047, kurtosis 4.77→3.99 —
   the fold mixes columns within each source block, diffusing weight
   outliers (FIG_B).

4. **Does W4 of folded W_c become better or worse?**
   Better: per-row RTN W4 NMSE 0.1025 (stock) → 0.0469 (folded).

5. **After W_c folding, does H_t become heavy-tailed again?**
   Yes — this is the core mechanism: the fold cancels R1_T (basis audit),
   so H_t re-concentrates outliers regardless of upstream flattening.

6. **H_t kurtosis / absmax / energy concentration immediately before R_C?**
   kurtosis 189.1, absmax 33.5 (unit-RMS scale), heavy top-channel energy
   (FIG_C left; per-dataset stats in tables/fourstage_rotation_stats.csv).

7. **How much does R_C change those values?**
   kurtosis 189→2.9, absmax 33.5→5.1 (deployed runtime Ht@R_C tap pair,
   same tokens, same z-scale: FIG_C).

8. **How much does H_t A4 NMSE change?**
   0.456 → 0.018 (25×; A4-dequant twins visualized in FIG_C2 — the coarse
   pre-R_C grid visibly quantizes most channels to zero codes).

9. **Does FP K/V cache change under R_C?**
   No, by algebra and by measurement setup: R_C is a function-preserving
   reparameterization (basis-audit §4/§7 identities, rel ≤1e-6), so the FP
   condition needs no RC arm — the FP cache is the shared reference.

10. **Does the W4A4-produced K/V cache become closer to the FP cache?**
    Yes, substantially (elementwise NMSE vs FP, same trajectory):
    ctx K 0.356→0.190, ctx V 0.702→0.442; transient draft-side K
    0.213→0.107, V 0.787→0.471. This — not a distribution change of the
    cache itself — is R_C's cache-level effect.

11. **Are stored K/V cache tensors themselves KV4/KV8 friendly?**
    Yes: FP stored ctx caches have kurtosis K 4.0–11.1 / V 3.5–5.5;
    hypothetical KV4 NMSE mean 0.0127 (max 0.0204), KV8 mean 4.5e-5
    (diagnostic only; deployment cache stays bf16).

12. **Which of the four stages is the strongest low-bit bottleneck?**
    Stage 3 (H_t): worst A4 NMSE (0.456) at a quantized interface, created
    AFTER the model-local rotations by the fold-cancellation, and fixed
    only by the context rotation. Stage-1 severity (NMSE 0.472) is already
    handled by standard R1_T; Stage-2 is weight-benign (W4 ≤0.10, improved
    by folding); Stage-4 is intrinsically low-bit-friendly (KV4 ~0.013) —
    its W4A4-production error is inherited from Stage 3.

Interpretation guardrails honored (§25): W_c pair = stock-vs-mandatory-fold
(not a rotation pair); H_t is NOT in R1_T basis; FP cache unchanged by
construction; cache was verified (not assumed) to be a downstream victim,
not the bottleneck; no cache-precision deployment change.
