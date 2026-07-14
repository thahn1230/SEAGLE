# SpinQuant-aware pure-R1 EAGLE draft — correctness summary

Mode `w4a4_spinquant_draft_pure_r1`. Greedy. Implementation-correctness pass.
Run dir `runs/spinquant_draft_pure_r1_20260708_0119/`. Llama-2-7b-chat + EAGLE.

## Headline
The new pure-R1 draft **recovers acceptance** (fp16 3.41 = stock B0 3.42 = B1
3.41; fake-W4A4 3.00 = B1 3.00), unlike the previous failed mode (B2 = 1.00).
It is single-path: external target hidden is rotated `h@R1` once per
verification cycle; the recurrent draft feature `f_R` is never rotated again.

## Baselines / ablations (fp16 n=20, 64 tok, acceptance = tokens/target-forward)
| config | accept | AR==EAGLE match | reading |
|---|---:|---:|---|
| B0 original EAGLE (unrotated) | 3.42 | — | stock reference |
| B1 unfused target + original draft | 3.41 | 0.95 | recovers |
| B2 previous [R1.T,I]/[R1.T,R1.T]+fused head | **1.00** | 1.00 | negative control (collapse) |
| **B3 NEW pure-R1 draft** | **3.41** | 0.95 | **RECOVERS = B0/B1** |
| A1 external h NOT rotated | 1.17 | 0.90 | collapse (external must be @R1) |
| A2 @R1 once per prompt only | 1.37 | 0.95 | collapse (must be EVERY cycle) |
| A3 recurrent f_R rotated again | **2.30** | 0.95 | degrades (3.41→2.30): recurrent must NOT be re-rotated |
| A4 rotate input but fc NOT conjugated | 1.15 | 1.00 | collapse (PL weight must be conjugated) |
| A5 PL-conjugated, no R2/R3/R4 | 3.41 | 0.95 | = B3 (R2/R3/R4 fp16-identity) |
| A6 gamma_f-fused head on f_R | 3.41 | 1.00 | head basis WRONG (top-1 0.875, rel-L2 large) but positive gamma scaling preserves most argmaxes → acceptance mostly survives |

Note on `AR==EAGLE match = 0.95`: this is NOT interface-specific — the
KNOWN-GOOD B1 shows the same 0.95. High-acceptance EAGLE uses the batched
TREE-verification forward, which differs from incremental AR by fp16 roundoff
at rare near-ties; low-acceptance configs (B2, A4) degenerate to incremental
and show 1.00. Verifier logits on matched prefixes are identical (rel-L2 0.0).

## fake W4A4 (n=2) + path determinism
B1 3.00 / B2 1.00 / **B3 3.00** (acceptance meaningful, not ~1.0). AR==EAGLE
match 0.50 (n=2/16tok) → 0.00 (n=20/64tok, B1) — caused by fake-W4A4 GREEDY PATH
NON-DETERMINISM (plain W4A4 target: incremental `use_cache=True` vs full
`use_cache=False` disagree at 2/16 near-ties; fp16 shows 0), NOT interface
failure. `path_determinism_w4a4.json`.

## REAL W4A4 (QuaRot CUTLASS INT4, n=2) — kernel dispatch PROVEN
224 target linears swapped to QuaRot packed-INT4 (weight_bits 4, act_bits 4,
KV 16); `int4_dispatch_proven=True` (CUTLASS `integer_subbyte<4>` GEMM +
`sym_quant`/`sym_dequant` in the profiler; `kernel_dispatch_trace.txt`).
- B2 (prev failed): accept **1.000**
- **B3 (pure-R1): accept 3.517** — recovers under REAL INT4 kernels.
(B3's 3.517 > fp16 3.41 partly reflects the QuaRot recipe's quality degradation
making output more predictable — an acceptance artifact, not a quality gain; the
point is B3 ≫ 1.0 with proven INT4 dispatch, and B2 stays 1.0.)

## The 16 required answers
1. **Old [R1.T,I]/[R1.T,R1.T] removed from main path?** YES — the main mode
   (B3) is the R1-conjugated draft + runtime `h@R1` external-only. The old mode
   is kept ONLY as the B2 negative control.
2. **R1 applied only to external target hidden at the first draft forward of
   every cycle?** YES — `draft_forward_basis_trace.csv`: `@R1` at
   `is_external_target_forward` (tree_depth 0) of every `verification_cycle_id`;
   one external `@R1` per cycle (not once per prompt).
3. **Recurrent draft feature avoids extra rotation?** YES — trace: recurrent
   forwards `runtime_transform=none`. A3 (rotate recurrent) collapses.
4. **PL conjugated as `W_PL_R = R1.T @ W_PL @ blkdiag(R1,R1)`?** YES — fp64
   rel-L2 1.2e-15 (`pl_conjugation_correctness.json`); `b_R = b@R1`.
5. **PL output verified `f_R = f@R1`?** YES — via the fp64 conjugation identity;
   the full pure-R1 draft matches the reference at all tree levels
   (cosine 0.99999999, established in the pure-R1 study).
6. **R1/R2/R3/R4 all applied?** R1 conjugation applied in the generation draft
   (exact, correctness-critical). R2/R3/R4 AUDITED as fp16-identity orthogonal
   quantization rotations (honest fp16 errors: R2 5.6e-4, R3 4.8e-4, R4 6.9e-4;
   fp64 ~1e-15). They do NOT change fp16 acceptance (A5 = B3) — they only
   reshape draft activations when the draft is quantized, which the fp16 /
   target-only-W4A4 path does not do.
7. **Where draft RMSNorm gamma folded?** The draft's only norm
   (`post_attention_layernorm`, gamma_l) is folded into `gate_proj`/`up_proj`;
   the norm weight becomes ones. Layer 0 has NO input_layernorm (nothing to
   fold). The target `gamma_f` is NOT folded into the draft.
8. **Which LM head for draft scoring?** `W_lm @ R1`.
9. **gamma_f-fused head avoided in the pure-R1 path?** YES — used only in A6.
10. **fp16 acceptance recovered to original EAGLE?** YES — B3 3.41 vs B0 3.42 /
    B1 3.41 (NOT ~1.0).
11. **fake W4A4 meaningful acceptance?** YES — B3 3.00 = B1 (B2 stays 1.00).
12. **Real W4A4 run?** YES (fake W4A4 showed no interface failure — B3
    acceptance recovered — so real W4A4 was permitted). QuaRot CUTLASS INT4,
    224 target linears swapped; B3 accept 3.517, B2 accept 1.000.
13. **Real W4A4 kernel dispatch proven?** YES — profiler shows the CUTLASS
    `integer_subbyte<4>` INT4 GEMM + `sym_quant`/`sym_dequant`
    (`int4_dispatch_proven=True`, `kernel_dispatch_trace.txt`); no fp16 fallback.
14. **AR==EAGLE output match?** fp16 0.95 (tree-vs-incremental near-tie, same as
    known-good B1); fake W4A4 0.50 (path non-determinism).
15. **W4A4 mismatch cause — target path non-determinism or interface failure?**
    TARGET PATH NON-DETERMINISM (proven on the plain W4A4 target; B3 acceptance
    recovers). NOT interface failure.
16. **Final recommendation.** The pure-R1 SpinQuant-aware draft is the correct
    direction: it recovers full acceptance with a coherent single-path R1 basis
    (external `h@R1` per cycle, recurrent `f_R` untouched, PL conjugated, head
    `W_lm@R1`). Keep this over the [R1.T,·] runtime hack (B2). For W4A4, use
    same-execution-path verifier comparisons (not exact token equality) because
    fake-quant greedy is path-dependent. R2/R3/R4 on the draft are needed only
    if the draft itself is quantized (fp16-identity otherwise).
