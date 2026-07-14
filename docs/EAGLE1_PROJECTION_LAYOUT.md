# EAGLE-1 PROJECTION LAYOUT (code-verified, EAGLE v1 @ 4a9cf3a)

## Concat order

`third_party/EAGLE/eagle/model/cnets.py:593` (class `Model.forward`):

    inputs_embeds = inputs_embeds.to(hidden_states.dtype)
    hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

→ order is **[token_embedding | feature]**:

| slice of `fc.weight` [4096, 8192] | multiplies | semantic name |
|---|---|---|
| `[:, 0:4096]` | token embedding `e` | `W_embedding` |
| `[:, 4096:8192]` | target/draft feature | `W_feature` |

`fc = nn.Linear(2*hidden, hidden, bias=True)` (`cnets.py:492`); the
`yuhuili/EAGLE-llama2-chat-7B` checkpoint ships `fc.bias`.

Runtime assertion used by the test suite: zeroing `fc.weight[:, 4096:]` must
kill the model's sensitivity to the `hidden_states` argument but not to
`input_ids`, and vice versa (see `tests/test_b2_concat_order.py`).

## Call sequence inside `topK_genrate` (cnets.py:762-845)

    call #0: self(hidden_states=TARGET post-norm features, input_ids=accepted ids)
             rows: 1..L_accepted, ALL target-originated  → projection_first
    tree loop (i = 0..len(tree_indices)-1):
        hidden_states = repeat_hidden(out_hidden ...)     draft's own features
        call #i+1: ALL draft-originated                   → projection_recurrent

`stable_kv` caches draft-layer KV across verification cycles; each cycle's
call #0 processes only the newly accepted suffix — still 100% target-originated
rows. **No call ever mixes the two origins.**

## Interface tensor

`ea_model.py:124-145`: target runs `base_model.model(...)`; `outputs[0]` is
post-final-norm (`modeling_llama_kv.py:1074`); passed to `topK_genrate` (prompt
phase) and, per cycle, `utils.py:466` passes accepted-position hiddens.

- stock target → `h_t = RMSNorm_gamma(x)` (gamma INCLUDED, original basis)
- SpinQuant-fused target → `a_t = n(x)·R1` (gamma EXCLUDED → folded into
  `lm_head`; rotated basis). This asymmetry vs the recurrent draft feature is
  what forces the B2 split.

## Head & embedding ownership

- draft `embed_tokens`: separate frozen copy (own storage).
- draft head: stock EAGLE passes `base_model.lm_head` INTO `topK_genrate`
  (alias!). All adapters in this repo substitute an isolated `adapter.head`
  inside the wrapped `topK_genrate` → draft-head precision is controlled
  independently of target verification (verified again by
  `tests/test_embedding_lm_head_aliasing`-equivalent checks in the B2 suite).
