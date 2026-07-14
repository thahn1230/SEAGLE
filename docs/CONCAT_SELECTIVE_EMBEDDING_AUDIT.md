# CONCAT-SELECTIVE EMBEDDING AUDIT

## Provenance & ownership (measured)

- `ea_layer.embed_tokens.weight` is loaded at draft init by
  `cnets.Model(load_emb=True, path=<target checkpoint>)` — read DIRECTLY from
  the untouched `meta-llama/Llama-2-7b-chat-hf` safetensors
  (`model.embed_tokens.weight`), never through any fused/rotated target module.
- Measured: draft embedding vs untouched checkpoint tensor → rel-L2
  **0.000e+00** (bit-exact). Frozen (`requires_grad=False`), OWN storage
  (separate data_ptr from the target's embedding; the target's R1 fusion
  mutates only the target module's storage).
- Therefore NO restoration was needed: the primary architecture uses the
  as-loaded original-basis table. (If it had been fused, the preferred fix —
  checkpoint reload — is what `build_standalone_draft`/cnets already do.)

## Object identity table

| tensor | storage | basis |
|---|---|---|
| target embedding before rotation | target module (checkpoint load) | original |
| target embedding after R1 fusion | same storage, mutated in fusion | rotated |
| draft embedding | SEPARATE storage (cnets copy at init) | original (bit-exact ckpt) |
| untouched checkpoint embedding | safetensors on disk | original |

## Runtime guarantees in the primary architecture

- No `e→e·R1`, `e→e·R1ᵀ`, `e→e·γ` anywhere; `W_e` block bit-untouched in both
  folded projections (`tests/test_concat_embedding_unchanged.py`).
- Adapter `install()` asserts the loaded embedding checksum equals the original
  checkpoint copy's (primary modes; the `embedding_rotated` NEGATIVE control
  and the explicit `quant_embed` ablation are the only modes that may modify
  it, and both are labeled).
- Previous-B2 comparison: that architecture rotates the table (`E·R1`) inside
  `install()` and restores the original on `uninstall()` — its rotated table is
  never shared with, or leaked into, the concat-selective runs (state dicts are
  reloaded from per-adapter backups; verified by the equality of F0 reruns).
