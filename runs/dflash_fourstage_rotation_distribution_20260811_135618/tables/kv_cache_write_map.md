# K/V cache write map (audited)
Single write point per layer: dflash/model.py:238 `past_key_values.update(...)`.
| tensor | source | norm/RoPE state | dtype | persistent? |
|---|---|---|---|---|
| context K | k_proj(H_t[@R_C]) rows [:prefix_len] | post k_norm, post RoPE | bf16 | YES (grows monotonically) |
| context V | v_proj(H_t[@R_C]) rows [:prefix_len] | raw projection (no norm/RoPE) | bf16 | YES |
| draft K | k_proj(input_ln(h)) rows [prefix_len:] | post k_norm, post RoPE | bf16 | NO — crop(start) discards every cycle (model.py:120) |
| draft V | v_proj(input_ln(h)) rows [prefix_len:] | raw | bf16 | NO — transient |
Capture taps: RotQuantDraft S4_k_stored_l{i}/S4_v_stored_l{i} (regression-gated
bitwise-neutral; exactly the tensors passed to the cache write in the stock
path). Flattening for plots: channel = head_index*128 + head_dim.
