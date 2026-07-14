# 04 — Draft Conjugation Notes (Variant B)

## The transform

EAGLE draft first projection (cnets.py:592-593): `fc(cat([e, h]))`, an
`nn.Linear(2D -> D)`. In nn.Linear terms `y = x @ W^T`; with `x = [e | h]` the
embedding occupies input columns `[0:D]` and the hidden occupies `[D:2D]`.

To consume the rotated hidden `h_hat` directly (h = (h_hat @ R1^T) * gamma_f):

    W_h_new = W_h_old @ (diag(gamma_f) @ R1)        # fc.weight[:, D:2D]

The embedding block is left untouched (the draft's embedding is the original,
unrotated Llama embedding, so `e` is already in the right basis).

## Full-precision identity check (real draft, single forward)

    feature_max_abs_err = 1.431e-05
    feature_rel_err     = 2.214e-06
    feature_cosine_sim  = 1.000000
    logit_kl            = -1.611e-07

The conjugated draft consuming `h_hat` reproduces the stock draft consuming `h`
to floating-point roundoff on the first forward. Variant B == Variant A there.

## Recycling caveat (discovered from the code)

EAGLE tree drafting (cnets.py:806-817) recycles the draft's OWN predicted
features `out_hidden` back into the same `fc` for subsequent tree levels. The
draft is trained to predict ORIGINAL-basis features, so `out_hidden` is in the
original basis — but after folding, `fc`'s hidden block now expects the ROTATED
basis. Hence the fc fold is exact only for the FIRST (external) hidden; across the
recycled tree-expansion steps Variant B diverges from Variant A unless the
recycled features are also rotated to the h_hat basis (an online op of the same
cost as Variant A's unrotation).

Consequence: Variant A (explicit unrotation at the draft entry, applied once per
draft call, with the internal recycling naturally consistent in the original
basis) is the correctness reference for end-to-end generation. Variant B is the
zero-overhead form ONLY for single-step feature prediction; for full generation
it is either (a) an approximation, or (b) equivalent to A once recycled features
are compensated. The end-to-end numbers in `results/conjugated_rotation_*` show
this divergence directly.
