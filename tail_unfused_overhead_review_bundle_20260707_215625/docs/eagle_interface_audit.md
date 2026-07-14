# EAGLE-1 Interface Audit (for the rotation study)

Repo: `third_party/EAGLE` @ branch `v1`, commit `4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5`.
All line numbers re-verified on 2026-07-04. This document answers the eight
inspection items required before implementing the rotation-study variants.

## 1. Where EAGLE obtains the target hidden `h`

Two capture sites, both returning the same tensor kind:

- **Prefill / round start** — `eagle/model/ea_model.py:122-132` (`EaModel.forward`):
  ```python
  outputs = self.base_model.model(input_ids=..., past_key_values=..., position_ids=...)
  if output_orig:
      orig = self.base_model.lm_head(outputs[0])
  hidden_states = outputs[0].clone()
  ```
  `outputs[0]` is the base LlamaModel's `last_hidden_state`.
- **Post-verification re-draft** — `eagle/model/utils.py:466-468`
  (`update_inference_inputs`): `model.ea_layer.topK_genrate(accept_hidden_state_new, ...)`
  where `accept_hidden_state_new` is sliced from the tree-decoding hidden output
  (`utils.py:454-455`).

Both flow into the single chokepoint `ea_layer.topK_genrate(hidden_states, ...)`
— this is where all variant adapters are installed (instance-level wrap, no
third_party edits).

## 2. Post- or pre-final-RMSNorm?

**Post-final-RMSNorm.** `eagle/model/modeling_llama_kv.py:1074`:
`hidden_states = self.norm(hidden_states)` is applied before
`last_hidden_state` is returned (line ~1088). Training data generation matches
(`ge_data_all_llama2chat.py:185-186` stores `hidden_states[-1]`, which HF appends
after the final norm).

Consequence for rotation: SpinQuant folds the final-norm scale `gamma_f` into
`lm_head` and zeroes the norm scale, so the rotated model's post-norm output is
`h_hat = (h / gamma_f) @ R1` — the interface inverse must re-apply `gamma_f`.

## 3. Exact draft `fc` input order

`eagle/model/cnets.py:492`: `self.fc = nn.Linear(2*hidden, hidden, bias=bias)`
(bias=True for this checkpoint). Forward combine at `cnets.py:593`:
```python
hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
```
**Order is `[e, h]` — embedding FIRST.** Therefore in `fc.weight` ([4096, 8192]):
- columns `[:, 0:4096]` multiply the embedding branch `e`
- columns `[:, 4096:8192]` multiply the hidden branch `h`  ← fold target for B/B2

No activation follows the fc (the activated variant is commented out at :591).

## 4. Draft embedding / head status

- **Own frozen embedding**: `cnets.py:466` creates `self.embed_tokens`;
  `cnets.py:494-495` freezes it. It is loaded from the DRAFT checkpoint
  (`ea_model.py:106`, `load_state_dict(..., strict=True)`; verified the yuhuili
  `pytorch_model.bin` contains `embed_tokens.weight [32000, 4096]`). It is a
  physically separate tensor from the target's embedding, so rotating the target
  never touches `e`. **`e` is always original-basis.**
- **No lm_head of its own**: the head is passed per call
  (`topK_genrate(..., head, ...)`, applied at `cnets.py:781/820`; multi-device
  fallback uses `self.headweight`, `cnets.py:787/826`). Variants supply a
  pre-rotation copy of `lm_head` here, so draft scoring is original-basis.

## 5. Where tree expansion recycles predicted features

`eagle/model/cnets.py:806-816` inside `topK_genrate`'s level loop:
```python
if i==0: hidden_states = out_hidden[:, -1:]
else:    hidden_states = out_hidden
hidden_states = self.repeat_hidden(hidden_states, ...)
out_hidden, past_key_values = self(hidden_states, input_ids=..., past_key_values=..., ...)
```
`out_hidden` is the draft's OWN predicted feature (original basis). Call
structure per `topK_genrate` invocation:
- **call #1 (external)**: consumes target hidden (`cnets.py:775/777`) — this is
  the only call that sees `h_hat` under rotation;
- **calls #2..#(depth)**: consume recycled `out_hidden` — always original basis.

Verified with truncated trees (`generate_tree_buffers` on `mc_sim_7b_63`
filtered by path length): recycled calls per round = depth−1
(depth 2→1, 3→2, 4→3, 5→4). **depth=1 is not runnable** — EAGLE's own
`utils_c.generate_tree_buffers` crashes on a tree with no level-2 nodes
(IndexError), so the depth sweep uses depths 2–5 and level-1 exactness of B is
established via single-forward identity + level-wise hidden diagnostics instead.

`mc_sim_7b_63`: 25 nodes, depth histogram {1:4, 2:8, 3:8, 4:3, 5:2},
prefix-closed, and remains prefix-closed under every length-≤d truncation
(verified programmatically) — so depth-truncated trees are valid tree_choices.

## 6. Variant insertion points

All installed WITHOUT editing third_party (instance attribute patching):

| variant | mechanism |
|---|---|
| Naive | wrap `ea_layer.topK_genrate`: pass `h_hat` through; substitute original-basis head |
| A (unrotate) | same wrap: `h = (h_hat @ R1.T) * gamma_f` before the original `topK_genrate`; original head |
| A_nogamma | same wrap: `h = h_hat @ R1.T` (ablation) |
| B (fold) | offline: `fc.weight[:, 4096:8192] @= diag(gamma_f) @ R1` (fp64, cast back); wrap only substitutes the head |
| **B2 (two-path fold)** | keep TWO fc hidden-block weights (folded / original). Wrap `topK_genrate` to arm a flag; wrap `ea_layer.forward` (instance) to select the folded weight for the FIRST draft call after arming (external `h_hat`) and the original weight for subsequent calls (recycled `f`). Weight switch is a `.data` pointer swap (no copy). |
| C | load retrained draft state dict; use the A wrap |

The head substitution is uniform: every adapter passes a frozen copy of the
PRE-rotation `lm_head` into `topK_genrate` (stashed before `fuse_layer_norms`).

## 7. Where timing is measured

- **target time**: instance-wrap `base_model.model.forward` (covers prefill,
  tree decoding at `utils.py:308`, and vanilla `naive_generate` since
  `base_model.__call__` routes through it). CUDA-event accumulate.
- **draft time**: outermost wrap of `ea_layer.topK_genrate` (adapter inside).
- **verify time**: patch `eagle.model.ea_model.evaluate_posterior` — legal
  because `ea_model.py` does `from .utils import *`, so the name is resolved
  from `ea_model`'s module globals at call time.
- **unrotation time**: CUDA events inside the A/A_nogamma adapter around the
  unrotation GEMM only.
- Whole-generation wall clock: CUDA events around the `ea_generate` /
  `naive_generate` drive loop (as in `eagle_bridge.generate_and_measure`).

## 8. Rotation-side facts the adapters rely on

(from `third_party/SpinQuant`, verified in docs/appendix/A2)
- `fuse_layer_norms` (utils/fuse_norm_utils.py) folds every norm scale into the
  adjacent linears, folds `gamma_f` into `lm_head`, zero-centers embedding rows.
  **`gamma_f` must be stashed BEFORE this call.**
- `rotate_model` (eval_utils/rotation_utils.py:122-147): R1 fused into
  embed/qkv/up/gate (input side), o/down (output side), lm_head; per-layer R2
  fused v_proj-out / o_proj-in; R4 inverse folded into down_proj + online
  Hadamard at runtime; R3 = online post-RoPE Hadamard on Q/K, added only when
  K-cache quantization is on (monkeypatched around `apply_rotary_pos_emb`).
- Consequence: in exact arithmetic only **R1 and gamma_f** are visible at the
  drafter interface; R2/R3/R4 cancel inside their blocks (H4 to be validated
  empirically via the component ablation).
- Component-ablation caveat: SpinQuant's `rotate_mlp_output` couples the R1
  output-side rotation with the R4 fold in one function; the study's
  `rotate_components()` re-implements it split so `r1`-only configs do not fold
  R4 into `down_proj` without also enabling the online Hadamard (which would
  change the function).
