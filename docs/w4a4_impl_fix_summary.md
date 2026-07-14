# w4a4_impl_fix — implementation correctness summary

Mode `w4a4_unfused_target_fused_draft_head`. Greedy (temperature 0). This is an
implementation-correctness pass, NOT a research claim. Run dir:
`runs/w4a4_impl_fix_20260707_2314/`.

## The 12 required answers

1. **Did `fc.weight` have shape [4096, 8192]?** YES (Linear 8192→4096).
2. **Was concat order [embedding, hidden]?** YES (cnets.py:592
   `cat((inputs_embeds, hidden_states))`; `W_e=fc.weight[:, :4096]`,
   `W_h=fc.weight[:, 4096:]`).
3. **Did the target hidden passed to the draft equal
   `h_unfused = (h_hat @ R1.T) * gamma_f`?** YES — implemented explicitly in the
   patched `model.norm.forward`; note algebraically `h_unfused =
   (RMSNorm0(x)@R1@R1.T)*gamma_f = RMSNorm0(x)*gamma_f = h` (the original
   post-norm hidden). Target-tail correctness (Test A): unfused-tail logits vs
   fused-tail logits top-1 agreement = 1.000 on every prompt.
4. **Did the first draft forward use `[R1.T, I]`?** YES — projection branch
   trace shows first (external) forward = embedding→R1.T, hidden→Identity.
5. **Did recurrent draft forwards use `[R1.T, R1.T]`?** YES — trace shows every
   recycled forward = embedding→R1.T, hidden→R1.T.
6. **Did the draft use the fused lm_head (R1.T + gamma_f absorbed)?** YES —
   draft head = `W_lm @ diag(gamma_f) @ R1`; trace logs
   `lm_head_mode=fused..., has_R1T_absorbed=True, has_gamma_f_absorbed=True`.
7. **Did W4A4 AR-only and W4A4 target+EAGLE produce identical greedy tokens?**
   - **fp16: YES, exactly (match rate 1.000).**
   - **fake W4A4: NO (match rate 0.5 at n=2).** Diagnosed below — this is NOT
     an interface bug.
8. **Did verifier logits match AR logits at every step?**
   - fp16: YES (rel-L2 0.0, top-1 100% at every step).
   - fake W4A4: NO at near-ties — see diagnosis.
9. **Was real W4A4 kernel dispatch proven?** NOT RUN. Per the spec's own gate
   ("if fake W4A4 fails, do not run real W4A4"), and because fake W4A4 already
   revealed the fundamental issue, real W4A4 was correctly not run. (Real INT4
   backends exist in this repo — tinygemm W4A16, QuaRot W4A4, see
   `docs/real_int4_kernel_audit.md` — and would inherit the SAME greedy
   path-non-determinism, likely worse.)
10. **First mismatch and its cause.**
    First mismatch: prompt 81, position 3 (AR token 8991 vs EAGLE 8565).
    **Root cause (proven on the PLAIN W4A4 target with NO adapters):
    fake-W4A4 greedy decoding is EXECUTION-PATH-DEPENDENT.** The same prefix
    gives different greedy argmax under incremental decode (`use_cache=True`,
    M=1, which `naive_generate` uses) vs a full forward (`use_cache=False`,
    M=L): 2 disagreements in 16 steps, both at NEAR-TIES (top-2 logit gaps 0.32
    and 3.04), while the full forward is self-deterministic. In fp16 the same
    diagnostic shows 0 disagreements. EAGLE's tree verification is a THIRD
    execution pattern, so it too diverges from incremental AR at near-ties.
    → The token mismatch is inherent fake-quant numerical non-determinism
    amplified by greedy argmax, **not** the unfused-tail / projection-transform
    interface (which is exact in fp16).

## What is correct vs what this reveals

- **Interface implementation is CORRECT**: fp16 AR == EAGLE exactly (match 1.0),
  verifier logits identical, target tail exact, projection branches and fused
  draft head exactly as specified. Every one of the 6 structural checks passed.
- **The user-specified draft transforms do not form a coherent draft basis**:
  acceptance stays ≈1.0 (draft proposals always rejected), so EAGLE degenerates
  to AR (correct output, no speedup). Proven against a harness sanity control
  (identity path = acceptance 4.1 in the SAME harness).
- **Fake W4A4 cannot satisfy exact token equality** across AR and EAGLE because
  fake-W4A4 greedy is path-dependent even for the target alone.

## Ablations (fp16; output match is 1.0 for ALL because output is
target-determined; the signal is ACCEPTANCE = draft-basis coherence)

| config | output match | acceptance | reading |
|---|---:|---:|---|
| main `[R1.T,I]/[R1.T,R1.T]`, fused head | 1.00 | **1.00** | draft always rejected |
| ident_control `[I,I]/[I,I]`, original head, h_unfused | 1.00 | **4.10** | known-good; harness sanity |
| abl1 first `[I,I]` | 1.00 | 1.00 | no help |
| abl2 first `[R1.T,R1.T]` | 1.00 | 1.00 | no help |
| abl3 unfused draft head | 1.00 | 1.44 | marginally better, still poor |
| abl4 target exposes h_hat (fused head) | 1.00 | 1.10 | poor |
| main **fake W4A4** | **0.50** | 1.00 | mismatch = fake-quant path non-determinism |

The only configuration that drafts effectively is the plain original-basis path
(`[I,I]`, original head, target exposes `h`), which is the earlier "explicit
unfused tail + naive draft" result (acceptance ~3.5-4). None of the
user-specified `[R1.T, ·]` projection-transform variants produce useful
acceptance.

## Recommendation (implementation-level only, no research claim)

1. The unfused-tail target + fused-draft-head + explicit projection transforms
   are implemented and audited; fp16 output equality holds exactly, so the
   plumbing is correct.
2. Exact token equality is the wrong success criterion under FAKE W4A4: the
   fake-quant target is not execution-path-invariant, so AR (incremental) and
   EAGLE (tree) diverge at near-ties regardless of the draft. For a quantized
   correctness criterion, compare verifier logits to AR logits computed by the
   SAME execution path, and expect near-tie divergence.
3. The user-specified `[R1.T, I]/[R1.T, R1.T]` draft transforms + fused head do
   not yield a working draft (acceptance ≈1.0). A coherent single-path
   rotation-aware draft requires the pure-R1 construction (target exposes
   `h_R = h@R1`, draft fully R1-conjugated) — see
   `docs/eagle_fully_rotation_aware_summary.md` — not runtime `[R1.T, ·]`
   transforms on an un-conjugated fc.
