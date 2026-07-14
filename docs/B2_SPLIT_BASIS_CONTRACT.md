# B2 SPLIT BASIS CONTRACT

Row-vector convention throughout (PyTorch Linear): `Linear(x; W, b) = x Wᵀ + b`,
weights stored `[out, in]`. `D = 4096`. `R = R1` (orthogonal, random Hadamard,
`outputs/rotations/random_hadamard/R.bin`). `D_gamma = diag(target_final_rms_gamma)`
(the LEARNED final-RMSNorm vector — never to be confused with any
`weight_quant_scale` / `activation_quant_scale`).

## Feature definitions

    n_t              unit-RMS-normalized target feature, NO gamma:  n(x) = x / rms(x)
    h_t   = n_t D_gamma                 feature stock EAGLE-1 expects (gamma INCLUDED, original basis)
    a_t   = n_t R                       what the FUSED SpinQuant target tail exposes
                                        (gamma EXCLUDED — folded into head/downstream; rotated basis)
    h_t_R = h_t R = n_t D_gamma R       complete gamma-included rotated feature
    h_d                                 draft AR-head recurrent feature (imitates h_t, gamma INCLUDED semantically)
    h_d_R = h_d R                       recurrent feature of the R1-conjugated draft

Orthogonality gives `n(xR) = n(x)R` **exactly** (unit-RMS norm has no
per-channel scale, and R preserves row norms). But gamma does NOT commute:

    (n_t R) D_gamma  ≠  n_t D_gamma R          in general
    C = D_gamma R − R D_gamma ≠ 0              (commutator diagnostic, measured below)

so `a_t · D_gamma` is WRONG (this is negative control FP-N2/"gamma_elementwise"),
and one permanently gamma-folded projection is WRONG for recurrent steps
(negative control FP03/"single_folded"; measured fp16 acceptance 2.17 vs 3.37).

If only `a_t` is available, the complete rotated feature is recovered by

    h_t_R = a_t M_gamma ,   M_gamma = Rᵀ D_gamma R
    h_t   = a_t Rᵀ D_gamma

## EAGLE projection (actual layout)

`cnets.py:593`: fc input = `concat(e, f)` → weight `W_proj[:, :D] = W_embedding`,
`W_proj[:, D:] = W_feature`, bias `b` present. Original computation:

    y = e W_embeddingᵀ + h W_featureᵀ + b

## Architecture B (both rotated): the two projections

Desired rotated output `y_R = y R` for both paths.

FIRST forward (input feature = `a_t`, gamma NOT included):

    a_t W_feature_firstᵀ = (h_t W_featureᵀ) R
    ⇒  W_feature_first = Rᵀ W_feature D_gamma R

RECURRENT forwards (input feature = `h_d_R`, gamma already included):

    h_d_R W_feature_recurrentᵀ = (h_d W_featureᵀ) R
    ⇒  W_feature_recurrent = Rᵀ W_feature R

Both paths share:

    W_embedding_rot = Rᵀ W_embedding R      (token branch NEVER receives D_gamma)
    b_rot           = b R

Invariant: `W_feature_first ≠ W_feature_recurrent` (they differ by the inner
D_gamma); gamma_f enters EXACTLY ONCE, through `projection_first`.

Mapping to this repo's validated primitives
(`rotation_aware.in_fold(W,R,γ) = W·diag(γ)·R`, `out_fold(W,R) = Rᵀ·W`):

    convert_draft_state(sd, R1, γ, mode='r1') returns
      out['fc.weight']  = Rᵀ·[W_e·R | W_f·R]        = projection_recurrent   ✓
      extra['fc_ext']   = Rᵀ·[W_e·R | W_f·D_γ·R]    = projection_first       ✓
      out['fc.bias']    = b·R                        = shared rotated bias    ✓

## Architecture A (target-only): original-basis split

Draft stays original; first-step input is `a_t` from the fused target; desired
output = ORIGINAL-basis y.

    FIRST:      W_feature_first_A = W_feature D_gamma R        (consumes a_t, emits original y)
    RECURRENT:  original W_feature unchanged                    (consumes original h_d)
    embedding block & bias unchanged in both.

(= this repo's Variant B2 `TwoPathAdapter`: `W_h ← W_h @ (D_γ R)` for the first
call only. A-explicit alternative: runtime `h_t = (a_t Rᵀ)·γ` + fully stock
draft = Variant A `UnrotateAdapter`.)

## Draft head (Arch B)

Recurrent/rotated features approximate `h·R`; the stock draft head applies
`W_lm` to `h`-like features with no extra norm, so the rotated head is

    W_head_rot = W_lm R      (in_fold(W_lm, R), NO gamma — gamma belongs to the
                              TARGET tail fold, not the draft loop)

## Target tail (fused, exact)

    norm.weight ≡ 1 after SpinQuant fusion → tail emits a_t
    W_lm_fused = W_lm D_gamma R → logits = a_t W_lm_fusedᵀ = h_t W_lmᵀ  (exact)

## Dispatch contract

Per `topK_genrate` cycle (verified: `cnets.py:772-820`): fc call #0 consumes
ONLY target-originated rows; calls #1.. consume ONLY draft `out_hidden` rows —
no mixed batches. Dispatch key = per-cycle fc-call index, reset at every
wrapped `topK_genrate` entry:

    idx == 0 → projection_first     (feature_origin=target_first,  gamma_already_included=false)
    idx  > 0 → projection_recurrent (feature_origin=draft_recurrent, gamma_already_included=true)

## Commutator diagnostic (filled by tests/test_rms_gamma_rotation_noncommutativity)

    ||C||_F, ||C||_2, ||C||_F/||D_gamma R||_F  → see artifacts/b2_split_study/commutator.json
