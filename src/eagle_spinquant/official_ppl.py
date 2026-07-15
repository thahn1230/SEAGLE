"""Official-contract WikiText-2 evaluator, callable in-process.

Reproduces third_party/SpinQuant utils.eval_utils.evaluator EXACTLY:
  - tokens: LlamaTokenizerFast(add_bos_token=False, add_eos_token=False)
            over "\n\n".join(wikitext-2-raw-v1 test["text"]) — full set
  - windows: non-overlapping seqlen=2048, tail truncated
  - per window: mean CE over the 2047 shifted positions (fp32)
  - PPL = exp(mean over windows of per-window mean NLL)
but runs a normal full forward per window on an already-placed model instead
of the layer-offload loop (mathematically identical logits path for a given
model implementation; use parity tests to verify).

Used to replace the old prefix-32768/overlap-window wikitext_ce for all
target-quality claims (GATE B corrected evaluator).
"""
import torch
import torch.nn.functional as F


def get_official_test_tokens(tokenizer_path):
    from datasets import load_dataset
    from transformers import LlamaTokenizerFast
    tok = LlamaTokenizerFast.from_pretrained(
        tokenizer_path, add_bos_token=False, add_eos_token=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt")
    return enc.input_ids  # (1, n_tokens)


@torch.no_grad()
def official_ppl(forward_logits, input_ids, dev, seqlen=2048):
    """forward_logits: callable(ids_batch)->logits (B, T, V).
    input_ids: (1, n_tokens) full-corpus tensor.
    Returns (ppl, ce_mean_of_window_means, window_rows)."""
    n = input_ids.numel() // seqlen
    ids = input_ids[0, : n * seqlen].view(n, seqlen)
    rows = []
    nlls = []
    for i in range(n):
        w = ids[i][None].to(dev)
        lg = forward_logits(w)
        shift_logits = lg[:, :-1, :].float()
        shift_labels = w[:, 1:]
        loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels,
                               reduction="none")
        m = loss.mean(dim=1)          # per-window mean over 2047 positions
        nlls.append(m.cpu())
        rows.append(dict(window=i, start=i * seqlen, end=(i + 1) * seqlen,
                         n_pred=seqlen - 1,
                         sum_nll=float(loss.sum()),
                         mean_nll=float(m)))
    ce = float(torch.cat(nlls).mean())
    return float(torch.exp(torch.tensor(ce))), ce, rows


@torch.no_grad()
def hf_forward_logits(model):
    def f(w):
        out = model(w, use_cache=False)
        return out.logits if hasattr(out, "logits") else out[0]
    return f


@torch.no_grad()
def eagle_forward_logits(ea_model):
    bm = ea_model.base_model
    bm.model.tree_mask = None
    def f(w):
        return bm(w).logits
    return f
