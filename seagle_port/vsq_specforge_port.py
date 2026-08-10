"""SpecForge-semantics DFlash training forward, ported for the Llama-3.1
DFlash pair (§4/§6 of the update).

Ported (adapted, provenance: refs/SpecForge @ 87e8cf4b, specforge/
algorithms/common/dflash_family_model.py + dflash_family_data semantics):
  - anchor sampling (two consecutive supervised tokens, sorted, keep mask)
  - noise construction (anchor token + mask tokens, shared embed)
  - absolute block position ids
  - create_dflash_sdpa_mask (ctx strictly < anchor; intra-block full
    bidirectional for our non-sliding draft; cross-block isolation)
  - loss_type="dflash": CE weighted by exp(-(pos-1)/gamma), gamma=5 (B=10)

NOT upstream: W4A4 fake-quant/STE — upstream SpecForge has no QAT; our
extension is labeled "SpecForge-semantics DFlash W4A4 QAT" (§5).

The draft forward reuses OUR validated z-lab Llama-3.1 DFlash modules
(dflash/model.py DFlashDraftModel), which are architecture-identical to
SpecForge's Qwen3 DFlash draft (model_type qwen3), with the attention mask
passed through to sdpa.
"""
import torch
import torch.nn.functional as F


def sample_anchor_positions(seq_len, loss_mask, num_anchors, device,
                            generator=None):
    num_candidates = max(seq_len - 1, 0)
    valid = (loss_mask[:, :num_candidates] > 0.5) & (
        loss_mask[:, 1:num_candidates + 1] > 0.5)
    valid_counts = valid.sum(dim=1)
    width = min(num_anchors, int(valid_counts.max().item()))
    if width == 0:
        raise ValueError("need two consecutive supervised tokens")
    random_values = torch.rand(valid.shape, device=device,
                               generator=generator)
    random_values.masked_fill_(~valid, 2.0)
    candidates = random_values.argsort(dim=1)[:, :width]
    keep = torch.arange(width, device=device).unsqueeze(0) < \
        valid_counts.clamp(max=width).unsqueeze(1)
    sentinel = valid.shape[1]
    anchors = torch.where(keep, candidates,
                          torch.full_like(candidates, sentinel))
    anchors = anchors.sort(dim=1).values
    keep = anchors < sentinel
    return torch.where(keep, anchors, torch.zeros_like(anchors)), keep


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S,
                            block_size, device, sliding_window=None):
    """SpecForge-exact 4D boolean mask [B, 1, Q_LEN, KV_LEN]."""
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size
    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2)
    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        lb = anchor_expanded + q_block_offsets - (sliding_window - 1)
        mask_context = mask_context & (kv_indices >= lb)
    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    if sliding_window is not None:
        kv_block_offsets = (kv_indices - S) % block_size
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)
    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def create_noise_ids(input_ids, anchor_positions, block_keep_mask,
                     block_size, mask_token_id):
    bsz, seq_len = input_ids.shape
    n = anchor_positions.shape[1]
    noise_ids = torch.full((bsz, n * block_size), mask_token_id,
                           dtype=torch.long, device=input_ids.device)
    block_starts = (torch.arange(n, device=input_ids.device) *
                    block_size).unsqueeze(0).expand(bsz, -1)
    anchor_tok = torch.gather(input_ids, 1,
                              anchor_positions.clamp(0, seq_len - 1))
    bidx = torch.arange(bsz, device=input_ids.device).unsqueeze(1).expand(
        bsz, n)
    noise_ids[bidx, block_starts] = torch.where(
        block_keep_mask, anchor_tok,
        torch.tensor(mask_token_id, device=input_ids.device))
    return noise_ids


def block_position_ids(anchor_positions, block_size):
    bsz, n = anchor_positions.shape
    off = torch.arange(block_size,
                       device=anchor_positions.device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + off).view(bsz, -1)


def gather_block_targets(input_ids, anchor_positions, block_keep_mask,
                         block_size, loss_mask=None):
    """Ground-truth labels (SpecForge exact): position k predicts the token
    AT anchor+k; k=0 (anchor token itself) excluded; bounds-checked; gated
    by the loss_mask at the label position."""
    bsz, seq_len = input_ids.shape
    n = anchor_positions.shape[1]
    off = torch.arange(block_size, device=input_ids.device).view(1, 1, -1)
    tgt_pos = anchor_positions.unsqueeze(-1) + off
    safe = tgt_pos.clamp(max=seq_len - 1)
    valid = (tgt_pos < seq_len) & block_keep_mask.unsqueeze(-1) & (off > 0)
    labels = torch.gather(input_ids, 1, safe.view(bsz, -1)
                          ).view(bsz, n, block_size)
    if loss_mask is not None:
        lm = torch.gather(loss_mask, 1, safe.view(bsz, -1)
                          ).view(bsz, n, block_size)
        valid = valid & (lm > 0.5)
    return labels, valid


def dflash_loss(logits, labels, valid, block_size, gamma=5.0):
    """loss_type='dflash': position-decay-weighted CE (SpecForge exact)."""
    neg_log_q = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none").reshape_as(labels).float()
    w = valid.float()
    if gamma and gamma > 0:
        pos = torch.arange(block_size, device=logits.device).view(1, 1, -1)
        w = w * torch.exp(-(pos - 1).clamp(min=0).float() / gamma)
    num = (neg_log_q * w).sum()
    den = w.sum().clamp(min=1.0)
    with torch.no_grad():
        top1 = (logits.argmax(-1) == labels) & valid
        acc = top1.float().sum() / valid.float().sum().clamp(min=1.0)
    return num / den, acc


def specforge_block_forward(draft, embed_tokens, lm_head, input_ids,
                            target_hidden_cat, loss_mask, num_anchors=64,
                            block_size=10, mask_token_id=128002,
                            gamma=5.0, generator=None):
    """One SpecForge-semantics training forward on our draft modules.
    target_hidden_cat: [B, S, m*D] concat of selected target hiddens
    (basis/quantization policy chosen by caller — H0/H1 contract §19)."""
    dev = input_ids.device
    bsz, S = input_ids.shape
    anchors, keep = sample_anchor_positions(S, loss_mask, num_anchors, dev,
                                            generator)
    noise_ids = create_noise_ids(input_ids, anchors, keep, block_size,
                                 mask_token_id)
    noise_embedding = embed_tokens(noise_ids)
    ctx_pos = torch.arange(S, device=dev).unsqueeze(0).expand(bsz, -1)
    pos = torch.cat([ctx_pos, block_position_ids(anchors, block_size)],
                    dim=1)
    mask = create_dflash_sdpa_mask(anchors, keep, S, block_size, dev)
    out = draft(position_ids=pos, noise_embedding=noise_embedding,
                target_hidden=target_hidden_cat,
                attention_mask=mask,                # 4D [B,1,Q,KV] for sdpa
                use_cache=False, is_causal=False)
    n = anchors.shape[1]
    hidden = out.view(bsz, n, block_size, -1)
    logits = lm_head(hidden)
    labels, valid = gather_block_targets(input_ids, anchors, keep,
                                         block_size)
    loss, acc = dflash_loss(logits, labels, valid, block_size, gamma)
    return loss, acc, {"anchors": anchors, "keep": keep, "mask": mask,
                       "labels": labels, "valid": valid, "logits": logits}
