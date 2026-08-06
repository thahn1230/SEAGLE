# Runtime & Folding Contract (pre-registered hypotheses + audit facts)

Classes: F0 folded offline (zero runtime op) · F1 math-foldable but
explicit op remains in current impl · F2 kernel-fusable (no separate
tensor materialization needed) · NF math-unfoldable · NA.

## Audit facts (current ConcatSelectiveDraftAdapter, file:line in BASELINE_AUDIT)

| transform | math | current | note |
|---|---|---|---|
| R_T into W_first hidden block | foldable | **F0** | M = diag(γ_f)·R_T fold |
| γ_f (first path) | foldable | **F0** | in M; exactly once |
| γ_f (target LM head) | foldable | **F0** | SpinQuant fuse_layer_norms |
| Stock-draft restore (a_t@R_Tᵀ)⊙γ | foldable | **F1** in RestoredInterfaceCSAdapter; **F0** via gamma_R1 fold (equiv. verified 3.5703==3.5703) | this study uses the F0 path |
| EP3-P m_first (activation side) | foldable into E table | **F0** (pre-scaled table) | A4 codes bitwise equal |
| EP3-P 1/m_f, 1/m_r (weight side) | foldable | **F0** (two W_e views) | first/rec weight views required |
| EP3-P m_r (activation side, ONE shared table) | NOT co-foldable with m_f into a single table | **F2 today** (e-slice scalar mul per recurrent call) | F1 option: dual table (+250 MiB) |
| EP3-G m (single) | foldable both sides | **F0 achievable** — verify impl; report F1 if runtime mul remains | hypothesis: zero runtime arithmetic |
| PostProjection R1/R_D (y@R) | required output-basis change; foldable INTO consumers? — the projection output feeds AR layers whose weights already absorb R (q=W_q·R etc.), so the explicit y@R is math-REMOVABLE only by re-deriving the fold; current impl | **F1/F2** (dense fp32 4096² GEMM every draft forward) | measure ms/call; biggest single explicit op |
| R_D (all draft weights, KV basis) | foldable (residual reparam) | **F0 for weights** (stash R1 swap refolds W_rec/AR/head); PostProjectionR1 remains explicit with R_D | Gate-P bitwise PASS |
| R2 (V/O per-head) | foldable | **F0** | deployed |
| R4 (down_proj, post-SiLU) | NF weight-only | **F2/F3** online Hadamard CUDA op | prologue-fusion candidate; cost reported separately |
| QAT | no runtime operator | **NA** | same exported structure as PTQ; offline values only |
| Duplicate embedding lookup / concat materialization / quant-dequant | — | measured in Phase 8 profiler pass | |

## Pre-registered hypotheses (Phase 8 tests these)

1. Naive PTQ / Generic QAT: no P3, no R_D → runtime = baseline
   adapter cost (PostProjectionR1 + R4 remain — shared by all arms).
2. EP3-G: fully pre-foldable → measured overhead vs naive ≈ 0.
3. EP3-P: + one fused e-slice scalar multiply per recurrent call (or
   dual-table memory trade) → overhead ≈ 0 latency, +ΔMB if dual.
4. EP3-P+R_D: same runtime class as EP3-P (R_D refold offline);
   PostProjectionR1 cost identical (same GEMM shape).
5. PTQ vs QAT exported runtime: identical.
6. Explicit-op profiler scan must find: PostProjectionR1 GEMM,
   rec_embed_rescale mul, R4 Hadamard, (restored arms only) interface
   restore — and NOTHING else rotation/scale-related.

Table D (final report) columns: Method | Transform | Math foldable |
Actually folded | Extra weight views | Extra memory | Explicit runtime
op | Measured overhead.
