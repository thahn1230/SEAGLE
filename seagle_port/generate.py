"""dflash_generate with interface hooks. Logic is a faithful copy of
dflash/model.py::dflash_generate (commit 94e4abc); hooks default to upstream
behaviour so Gate H can assert bit-identical outputs on stock arms.

Hooks:
  ctx_transform(H)  : [B,S,m*D] concat target hidden -> same shape.
                      Interface arms (explicit unrotation, MP3 activation
                      scaling) live here. Default identity.
  embed_fn(ids)     : noise embedding for the draft. Default
                      target.model.embed_tokens.
  head_fn(h)        : draft logits head. Default target.lm_head.
Per-cycle records: acceptance_length (=tau, accepted+1), per-position accept
prefix, block ids, for survival curves + RCAL replay.
"""
import torch
from types import SimpleNamespace

from dflash.model import extract_context_feature, sample, _cuda_time


@torch.inference_mode()
def dflash_generate_hooked(
    model,
    target,
    input_ids,
    max_new_tokens,
    stop_token_ids,
    temperature,
    block_size=None,
    mask_token_id=None,
    ctx_transform=None,
    embed_fn=None,
    head_fn=None,
    record_cycles=False,
    return_stats=True,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    embed_fn = embed_fn or target.model.embed_tokens
    head_fn = head_fn or target.lm_head

    from transformers import DynamicCache
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long,
        device=target.device)
    position_ids = torch.arange(output_ids.shape[1],
                                device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids)
        if ctx_transform is not None:
            target_hidden = ctx_transform(target_hidden)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    cycles = []
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]
        if block_size > 1:
            noise_embedding = embed_fn(block_output_ids)
            draft_logits = head_fn(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length():start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        posterior = sample(output.logits, temperature)
        match = (block_output_ids[:, 1:] == posterior[:, :-1])
        acceptance_length = match.cumprod(dim=1).sum(dim=1)[0].item()
        if record_cycles:
            cycles.append({
                "prefix_len": int(start),
                "block": block_output_ids[0].tolist(),
                "posterior": posterior[0].tolist(),
                "accept_prefix": match[0].int().tolist(),
                "tau": int(acceptance_length + 1),
            })
        output_ids[:, start:start + acceptance_length + 1] = \
            block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = \
            posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, model.target_layer_ids
            )[:, :acceptance_length + 1, :]
            if ctx_transform is not None:
                target_hidden = ctx_transform(target_hidden)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop = torch.tensor(stop_token_ids, device=output_ids.device)
        idx = torch.isin(output_ids[0][num_input_tokens:],
                         stop).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            output_ids = output_ids[:, :num_input_tokens + idx[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        cycles=cycles,
    )
