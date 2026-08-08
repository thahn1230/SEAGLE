# R_D / R_C boundary analysis (§17-18, RD3 proof) — 2026-08-07

Row-vector convention, F.linear y = xW^T. Target rotation R_T (global R1).

## 1. Draft dataflow bases

```
noise path : embed_tokens(ids) ──────────────► draft residual stream H_d
                                               (5 layers, residual adds)
             draft residual ─── norm ────────► lm_head
ctx path   : H_cat(target residual, m=5) ─► fc ─► hidden_norm(γ) ─► H_t
             H_t ─► k_proj/v_proj (ctx view)   [NO per-layer norm on ctx]
shared     : k_proj/v_proj also consume input_layernorm_i(H_d) per layer
```

## 2. RD3: zero-overhead R_D ≠ R_T is impossible under shared embed/head

Claim: with `target.model.embed_tokens` and `target.lm_head` fixed (shared,
unmodifiable), a draft-internal residual basis change to R_D ≠ R_T cannot be
realized by draft-internal weight folds alone.

Proof. Write one draft layer as h ← h + f(h) where f reads h only through
linears preceded by RMSNorm and writes through o_proj/down_proj. A basis
change h' = hB (B = R_T^T R_D orthogonal) is implementable inside the layer:
RMSNorm_bare is rotation-equivariant (‖hB‖=‖h‖ ⇒ n(hB) = n(h)B), γ can be
pre-folded into the following linear, so setting W_in ← B^T(D_γ W_in) and
W_out ← W_out B gives f'(hB) = f(h)B, hence hB + f'(hB) = (h+f(h))B. The
residual identity path transforms consistently ONLY because both summands
carry the same B — i.e., folding propagates the basis, it never creates or
removes it.

At the boundaries the stream basis is pinned by the shared modules:
- input: noise embedding is W_E-rows in the R_T basis (rotated target) ⇒
  the stream enters in basis R_T. To obtain basis R_D the map x ↦ xB must be
  applied to the embedding OUTPUT. There is no draft-internal weight
  multiplying the raw embedding before the first residual add — the first
  add is `residual + attn_out` with residual = the embedding itself. Any
  fold into first-layer W_in changes f's input read, not the residual term.
- output: lm_head expects basis R_T; the final residual (basis R_D if
  internally rebased) must be mapped by B^T after the last residual add and
  before lm_head — again a position with no draft weight.

Since B ≠ I (R_D ≠ R_T) and the two boundary positions carry no foldable
draft parameter while embed/head are unmodifiable, at least the two boundary
GEMMs (or equivalent weight VIEWS of embed/head) remain. ∎

Costs: RD1 explicit = 2 fp32 GEMMs (10-50 tok × 4096²) per block;
RD2 views = rotated embed copy + rotated head copy = 2 × 128256×4096×2B
≈ 1.96 GiB (tie_word_embeddings=false ⇒ both needed).

## 3. Structural finding: ANY draft-side rebasing needs ctx-specific K/V views

k_proj/v_proj are shared by two branches with different upstream transforms:
- noise branch: input_layernorm_i(H_d)  (per-layer γ_i)
- ctx branch:   H_t (post hidden_norm, no per-layer norm)

If the draft residual is rebased (RD0: R_D = R_T included!) the noise-branch
input becomes n(H_d)B, requiring W_k ← D_γi W_k B. The ctx branch input H_t
is NOT in the rebased coordinates, so the same folded W_k is wrong for it —
unless H_t is also rotated by B, which cannot be folded through hidden_norm's
γ into fc (γ sits between fc and the rotation; folding γ into the shared
k_proj corrupts the noise branch). Hence any of RD0/RD1/RD2/R_C requires
per-branch K/V weight views:
  cost = (k_proj 1024×4096 + v_proj 1024×4096) × 5 layers × 2B ≈ 84 MiB
This is 23× cheaper than RD2's 1.96 GiB and is the deployment argument for
context-only rotation R_C.

## 4. R_C exact placement

H_t' = H_t R_C with R_C applied AFTER hidden_norm. FP-invariance at init and
foldability: ctx views W_k^ctx = W_k R_C ... wait, y = (H_t R_C)(W_k^ctx)^T
must equal H_t W_k^T at init ⇒ W_k^ctx = W_k R_C (then (H_t R_C)(W_k R_C)^T
= H_t W_k^T ✓, pure gauge in FP; the quantization grids of both sides change
— that is the entire effect, exactly like R1 at the target interface).
Deployment: either fold R_C into a ctx-K/V view (F0, +84 MiB) or one runtime
GEMM on H_t per accepted-token batch (F2). hidden_norm untouched; noise
branch untouched; embed/head untouched ⇒ no RD3 obstruction.

## 5. RD0 (R_D = R_T) note

Even the "free" choice R_D = R_T needs: (a) no embed/head restore (native
basis match — removes the PostProjectionR1-analog transforms), but (b) draft
internal γ-fusion + conjugation folds, and (c) ctx-specific K/V views per §3
(or additionally rotating H_t by R_T, which needs the same views because of
hidden_norm γ). So RD0's deployment cost ≈ R_C's (+84 MiB views), while its
effect is a *basis alignment* rather than a learned optimization.
