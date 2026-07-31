# EAGLE Acceptance-Length Convention Audit (study §16)

Audited code paths:

1. **Official repo evaluator contract** (`eagle/model/utils.py`
   `update_inference_inputs`): after a cycle accepting `accept_length`
   proposal tokens, `input_ids` grows by
   `candidates[best, :accept_length+1]` — the `+1` is the cycle's root
   token (the previous cycle's verifier-sampled token), NOT a bonus of
   the current cycle. Per generation step the sequence therefore grows
   by `accept_length + 1` tokens.
2. **This repository's official tau** (`eval_eagle_acceptance_length.py`
   + `aggregate_micro_al.py`): tau = pooled mean of per-cycle sequence
   growth = mean(accept_length + 1) = **1 + accepted_draft_length**.
   Verified numerically: N1 naive tau 1.04 with near-zero acceptance
   (pure verifier progress = 1/cycle).

Conclusion: official tau **includes exactly one verifier-generated token
per cycle** on top of accepted draft proposal tokens.

RCAL convention (this study):

- `R_q, R_0, R_RC` and therefore `AL_q, AL_0, RCAL, SAL, LAL, P_A,
  R_A, AFS` are computed on **accepted draft proposal tokens only**
  (no verifier token).
- The tau-compatible views are valid and may be reported as:
  `AL-tau = 1 + AL_q`, `RC-tau = 1 + RCAL`.
- No silent addition/removal anywhere; conversions are explicit.
