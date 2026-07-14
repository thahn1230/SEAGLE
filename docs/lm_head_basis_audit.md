# lm_head basis audit (Task 2)

Real target `lm_head.weight`, real R1/gamma_f; fp64. logits reference = f @ W_lm.T (original feature, original head).

| feature basis | correct head | rel-err | wrong head | wrong rel-err |
|---|---|---:|---|---:|
| original f | W_lm | 0.00e+00 | — | — |
| f_R = f@R1 | W_lm @ R1 | 9.32e-16 | W_lm (orig) | 1.398 |
| f_hat = (f/γ_f)@R1 | W_lm @ diag(γ_f) @ R1 | 4.17e-15 | W_lm @ R1 | 3.263 |

Wrong-head top-1 agreement: f_R scored by original W_lm = 0.000 (vs correct W_lm@R1 = 1.000).

Each feature basis has exactly one correct head; using another basis's head causes large logit error (~30-50% rel-L2, near-zero top-1). A fully-R1 draft (features f_R) MUST be scored by W_lm @ R1, not the original head and not the SpinQuant fused head W_lm@diag(γ_f)@R1.
