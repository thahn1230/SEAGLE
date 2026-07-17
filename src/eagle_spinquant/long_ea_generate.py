"""Long-context replica of EAGLE-1 EaModel.ea_generate.

The stock generator hard-codes `if input_ids.shape[1] > 1960: break`
(ea_model.py:356) and `new_token > 1024`, which silently truncates any
run whose PROMPT already exceeds 1960 tokens — every cycle after the
first is dropped, so context-length studies above L~2000 are invalid.
third_party is read-only, so this module replicates the generator with
the caps replaced by the true KV-buffer capacity
(max_position_embeddings) minus the tree transient.

Behavior is otherwise identical (same eagle.model.utils calls); for
prompts under 1960 tokens it produces the same cycles as the stock
path.
"""
import torch

from eagle.model.utils import (initialize_tree, reset_tree_mode,
                               generate_candidates, tree_decoding,
                               evaluate_posterior,
                               update_inference_inputs)
from eagle.model.ea_model import generate_tree_buffers
from eagle.model.kv_cache import initialize_past_key_values


@torch.no_grad()
def long_ea_generate(model, input_ids, max_steps=512,
                     tree_choices=None, max_new_tokens=None):
    logits_processor = None
    assert input_ids.shape[0] == 1
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices,
            device=model.base_model.model.layers[-1]
            .self_attn.q_proj.weight.device)
        tree_buffers["retrieve_indices_head"] = \
            tree_buffers["retrieve_indices"].to(
                model.base_model.lm_head.weight.device)
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = \
            initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    tree_len = tree_buffers["tree_attn_mask"].shape[-1]
    kv_cap = model.base_model.config.max_position_embeddings
    hard_cap = kv_cap - tree_len - 8

    input_len = input_ids.shape[1]
    assert input_len < hard_cap, \
        f"prompt {input_len} exceeds KV capacity budget {hard_cap}"
    reset_tree_mode(model)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["tree_attn_mask"],
        past_key_values, logits_processor)
    new_token = 0

    for _ in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = \
            generate_candidates(tree_logits, tree_buffers["tree_indices"],
                                tree_buffers["retrieve_indices"],
                                sample_token, logits_processor)
        logits, hidden_state_new, _ = tree_decoding(
            model, tree_candidates, past_key_values,
            tree_buffers["tree_position_ids"], input_ids,
            tree_buffers["retrieve_indices_head"])
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor, cart_candidates_prob,
            tree_logits[2], tree_buffers["p_indices"], tree_candidates,
            tree_buffers["b_indices"])
        input_ids, tree_logits, new_token, hidden_state, sample_token = \
            update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length,
                tree_buffers["retrieve_indices"], logits_processor,
                logits, tree_logits, new_token, past_key_values_data,
                current_length_data, model, hidden_state,
                hidden_state_new, sample_p)
        yield input_ids
        if model.tokenizer.eos_token_id in \
                input_ids[0, input_len:].tolist():
            break
        if max_new_tokens is not None and new_token >= max_new_tokens:
            break
        if input_ids.shape[1] > hard_cap:
            break
