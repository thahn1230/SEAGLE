#!/usr/bin/env python
"""Phase 3 / GATE B: official vs project evaluator parity.

For base FP16 (stock HF) and chat FP16 (stock HF — same class the official
path uses), on ONE shared cached token tensor per model:

  (1) official contract in-process (official_ppl module): non-overlap 2048,
      per-window mean-of-means
  (2) OLD project contract (prefix 32768, overlap-1 windows, per-token mean),
      computed on the SAME shared tensor (control: no tokenizer diff)
  (3) OLD project contract on its OWN tokenization (add_bos=True chat
      tokenizer) — reproduces the original numbers' tokens

Additionally re-aggregates (1)'s per-position losses under (2)'s aggregation
to isolate pure aggregation-policy differences from token differences.

Outputs artifacts/spinquant_ppl_reproduction_fix/evaluator_parity/*.
"""
import argparse, csv, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from eagle_spinquant.official_ppl import (get_official_test_tokens,
                                          official_ppl, hf_forward_logits)

OUT = os.path.join(PROJECT_ROOT, "artifacts", "spinquant_ppl_reproduction_fix",
                   "evaluator_parity")


@torch.no_grad()
def project_ce_on_tensor(fwd, ids_1d, dev, n_tokens=32768):
    """OLD wikitext_ce aggregation, verbatim (prefix n_tokens, stride 2048,
    window 2049, per-token global mean)."""
    ids = ids_1d[:n_tokens].to(dev)
    losses, denom, rows = 0.0, 0, []
    for i in range(0, ids.shape[0] - 1, 2048):
        chunk = ids[i:i + 2049][None]
        if chunk.shape[1] < 2:
            break
        lg = fwd(chunk)[0, :-1].float()
        tgt = chunk[0, 1:]
        s = F.cross_entropy(lg, tgt, reduction="sum").item()
        losses += s; denom += tgt.numel()
        rows.append(dict(window=len(rows), start=int(i),
                         end=int(i + chunk.shape[1]),
                         n_pred=int(tgt.numel()), sum_nll=round(s, 6),
                         mean_nll=round(s / tgt.numel(), 6)))
    return losses / denom, denom, rows


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--chat")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "6,7")
    os.makedirs(OUT, exist_ok=True)
    dev = args.device
    summary = {}
    for name, path in (("base", args.base), ("chat", args.chat)):
        if not path:
            continue
        print(f"[parity] {name}: loading {path}", flush=True)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16).to(dev).eval()
        fwd = hf_forward_logits(model)

        # shared official token tensor
        toks = get_official_test_tokens(path)
        torch.save(toks, os.path.join(OUT, f"shared_tokens_{name}.pt"))

        ppl_off, ce_off, rows_off = official_ppl(fwd, toks, dev)
        write_rows(os.path.join(OUT, f"{name}_fp16_official.csv"), rows_off)
        print(f"[parity] {name} official: CE={ce_off:.5f} PPL={ppl_off:.4f} "
              f"({len(rows_off)} windows)", flush=True)

        # project aggregation on the SHARED tensor (control)
        ce_projS, denS, rows_projS = project_ce_on_tensor(fwd, toks[0], dev)
        write_rows(os.path.join(OUT, f"{name}_fp16_custom.csv"), rows_projS)
        print(f"[parity] {name} project-agg on shared tokens: "
              f"CE={ce_projS:.5f} PPL={float(torch.exp(torch.tensor(ce_projS))):.4f} "
              f"n={denS}", flush=True)

        # project aggregation on its OWN tokenization (original contract)
        tok_own = AutoTokenizer.from_pretrained(path)
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        own = tok_own("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        ce_projO, denO, _ = project_ce_on_tensor(fwd, own, dev)
        print(f"[parity] {name} project-own-tokens: CE={ce_projO:.5f} "
              f"PPL={float(torch.exp(torch.tensor(ce_projO))):.4f} n={denO}",
              flush=True)

        # official aggregation restricted to the same 32k prefix (isolate
        # subsample-vs-fullset)
        pre = toks[:, :32768]
        ppl_offp, ce_offp, _ = official_ppl(fwd, pre, dev)
        print(f"[parity] {name} official-agg on 32k prefix: CE={ce_offp:.5f} "
              f"PPL={ppl_offp:.4f}", flush=True)

        summary[name] = dict(
            official=dict(ce=round(ce_off, 5), ppl=round(ppl_off, 4),
                          n_windows=len(rows_off),
                          n_tokens=int(toks.numel())),
            project_agg_shared_tokens=dict(ce=round(ce_projS, 5), n=denS),
            project_own_tokens=dict(
                ce=round(ce_projO, 5), n=denO,
                add_bos=bool(getattr(tok_own, 'add_bos_token', None)),
                first_id=int(own[0])),
            official_agg_32k_prefix=dict(ce=round(ce_offp, 5)),
            ce_diff_shared=round(abs(ce_projS - ce_off), 6),
            token_sha=hashlib.sha256(
                toks.numpy().tobytes()).hexdigest()[:16])
        del fwd, model    # fwd closure retains the model -> OOM on next load
        torch.cuda.empty_cache()
        # write incrementally so a later-model crash keeps earlier results
        sp = os.path.join(OUT, "evaluator_parity_summary.json")
        merged = json.load(open(sp)) if os.path.exists(sp) else {}
        merged.update(summary)
        with open(sp, "w") as f:
            json.dump(merged, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("[parity] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
