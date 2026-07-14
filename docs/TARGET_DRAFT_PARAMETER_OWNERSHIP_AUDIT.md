# TARGET/DRAFT PARAMETER OWNERSHIP AUDIT

Runtime evidence: `artifacts/bitwidth_al_component_causality/logs/parameter_ownership.json`
(ids + data_ptrs captured on the live stock build; regenerated every stock-group
matrix run).

| tensor | ownership | basis | evidence |
|---|---|---|---|
| target `model.embed_tokens.weight` | target-owned storage | original (stock) / rotated (fused builds mutate ONLY this storage) | data_ptr differs from draft's |
| draft `ea_layer.embed_tokens.weight` | **independent copy** (cnets loads it from the checkpoint file at init) | original — bit-exact vs untouched checkpoint (rel-L2 0.0, verified 2026-07-14) | `shared_embed_storage=false` in the JSON |
| target `lm_head.weight` | target-owned | original / fused (W_lm·D_γ·R1) per build | — |
| draft scoring head | **stock EAGLE: ALIAS** — `topK_genrate(hidden, ids, head=bm.lm_head, …)` receives the target module itself | n/a | `ea_model.py:145` |
| draft scoring head under our adapters | **isolated copy** — every adapter substitutes `adapter.head` (own `nn.Linear`, own storage; `W_lm` or `W_lm·R1`) inside the wrapped `topK_genrate` | original or R1 | adapter code + data_ptr check |
| draft final norm | does not exist in the decode path (no `D_FINAL_NORM` op) | — | cnets topK_genrate |

## Experimental ownership modes

- `native_shared_or_copied_behavior`: stock T16_D16 runs with the native alias
  (head) and native copied embedding — labeled as such; used ONLY for the
  stock baseline cell and the optional native-sharing control.
- `isolated_component_behavior`: all draft-side quantization experiments use
  the isolated `adapter.head` (never the target verifier head) and the draft's
  own embedding storage (never the target's). Therefore:
  - "Draft-only LM-head quantization" quantizes ONLY the isolated scoring head;
    the target verifier head is untouched (verified by data_ptr).
  - "Draft-only embedding quantization" mutates ONLY `ea_layer.embed_tokens`
    (weight fake-quant applied to the draft state dict; target storage
    untouched; restored on uninstall).
  - Target-side embedding/LM-head ablations patch ONLY the target modules and
    run the STOCK draft.
