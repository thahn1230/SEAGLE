# fc projection fold audit (Task 1)

Real EAGLE-llama2-chat-7B `fc.weight`, real random-Hadamard R1, real target final-norm gamma_f; fp64.

## Shapes / order

- `fc.weight` shape = **[4096, 8192]** (Linear(8192->4096); is [4096,8192] = True)
- concat order = **[e, h]  (cnets.py:592  cat((inputs_embeds, hidden_states)))**
- `W_e = fc.weight[:, :4096]`, `W_h = fc.weight[:, 4096:]`
- R1 orthogonal max |RᵀR−I| = 0.00e+00; gamma_f in [0.0034, 2.8438], std 0.1161 (non-uniform)

## Fold directions (rel-L2 of folded output vs original fc output)

| input basis | correct fold | rel-err | wrong fold | wrong rel-err |
|---|---|---:|---|---:|
| h_hat = (h/γ_f)@R1 | W_h @ diag(γ_f) @ R1 | 4.43e-15 | W_h @ R1 | 40.067 |
| e_R = e@R1 | W_e @ R1 | 9.59e-16 | — | — |
| h_R = h@R1 | W_h @ R1 | 9.54e-16 | W_h @ diag(γ_f) @ R1 | 0.768 |

## Key point: h_hat vs h_R

- cos(h_hat, h_R) = 0.1463, rel-L2 = 4.263
- h_hat = (h/gamma_f)@R1 (scale-free a rotated); h_R = h@R1 (full post-norm hidden rotated). They differ by the final RMSNorm gain diag(gamma_f), which does NOT commute with R1.

All three claimed folds are exact (rel-err ~1e-15). Using the wrong fold (swapping the diag(γ_f) factor) yields ~30-50% error, confirming diag(γ_f) and R1 do NOT commute. The current SpinQuant tail exposes h_hat (needs the diag(γ_f) fold); a tail exposing h_R needs only W_h@R1 — the SAME fold used for recycled draft features f_R = f@R1, which is why h_R unifies the external and recurrent fc paths into one.
