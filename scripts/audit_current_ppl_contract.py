#!/usr/bin/env python
"""Phase 1: reproduce and fully document the execution contract that produced
the suspicious chat W4A4 CE=2.33469 / PPL=10.3263 / n_tokens=32767 (and the
FP16 1.92539 / 6.8578 baseline) from the bitwidth-AL study grader.

Rebuilds the exact grader-path targets via study.build_study_target and reruns
the exact wikitext_ce logic (copied verbatim from run_fixed_tree_target_grader
at commit b44152d), instrumented to dump:

  audit/current_ppl_execution_contract.json
  audit/current_quantized_module_inventory.csv
  audit/current_rotation_inventory.csv
  audit/current_tokenization_dump.json
  audit/current_per_chunk_losses.csv
"""
import argparse, csv, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant import fake_w4a4_draft as fq

ART = os.path.join(PROJECT_ROOT, "artifacts", "spinquant_ppl_reproduction_fix",
                   "audit")


def sha(t):
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def wikitext_ce_instrumented(model, tok, dev, n_tokens=32768):
    """VERBATIM logic of run_fixed_tree_target_grader.wikitext_ce, plus
    per-chunk logging."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids[0][:n_tokens].to(dev)
    model.base_model.model.tree_mask = None
    losses, denom, chunks = 0.0, 0, []
    for i in range(0, ids.shape[0] - 1, 2048):
        chunk = ids[i:i + 2049][None]
        if chunk.shape[1] < 2:
            break
        lg = model.base_model(chunk).logits[0, :-1].float()
        tgt = chunk[0, 1:]
        s = F.cross_entropy(lg, tgt, reduction="sum").item()
        losses += s
        denom += tgt.numel()
        chunks.append(dict(chunk_idx=len(chunks), start=i,
                           end=int(i + chunk.shape[1]),
                           n_pred=int(tgt.numel()),
                           sum_nll=round(s, 6),
                           mean_nll=round(s / tgt.numel(), 6)))
    return losses / denom, denom, chunks, ids, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cvd in ("6", "6,7"), cvd
    os.makedirs(ART, exist_ok=True)
    dev = args.device

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")

    contract = dict(
        source_commit="b44152d",
        source_function="scripts/run_fixed_tree_target_grader.py::wikitext_ce",
        checkpoint=paths["target_path"], model_id=paths["target_id"],
        model_impl="eagle.model.modeling_llama_kv.KVLlamaForCausalLM "
                   "(EAGLE-vendored 4.31-era, eager attention)",
        dataset="wikitext-2-raw-v1 split=test via datasets.load_dataset",
        text_join="\\n\\n".join(["<doc>"]) and "\\n\\n",
        n_tokens_policy="first 32768 tokens of the concatenated test set "
                        "(PREFIX SUBSAMPLE, not the full ~288k-token set)",
        window_policy="stride 2048, window 2049 (overlap 1): every token in "
                      "[1, 32767] predicted exactly once; global per-token "
                      "mean (denominator 32767)",
        boundary="cross-window boundary token used as context via 1-token "
                 "overlap; no window-mean reweighting",
        use_cache="model config default (not disabled); no past passed",
        dtype="float16 weights, fp32 loss accumulation",
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                  cudnn=torch.backends.cudnn.allow_tf32),
        kv_quant="fp16 (KV16)",
        gpu_deviation="physical GPU 7 absent; CVD=" + cvd,
    )

    results = {}
    for tag, rot, q in (("fp16", "none", "none"),
                        ("rh0_w4a4", "full", "w4a4")):
        print(f"[audit] building {tag} target ...", flush=True)
        model, stash, meta = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, "random_hadamard", q, 0, device=dev, rotations_root=rr)
        tok = eagle_bridge.get_tokenizer(model)
        ce, denom, chunks, ids, enc = wikitext_ce_instrumented(
            model, tok, dev)
        ppl = float(torch.exp(torch.tensor(ce)))
        print(f"[audit] {tag}: CE={ce:.5f} PPL={ppl:.4f} n={denom}",
              flush=True)
        results[tag] = dict(ce=round(ce, 5), ppl=round(ppl, 4),
                            n_tokens=denom)
        with open(os.path.join(ART, f"current_per_chunk_losses__{tag}.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chunks[0]))
            w.writeheader(); w.writerows(chunks)

        if tag == "fp16":
            tok_dump = dict(
                tokenizer_class=type(tok).__name__,
                name_or_path=tok.name_or_path,
                add_bos_token=getattr(tok, "add_bos_token", None),
                add_eos_token=getattr(tok, "add_eos_token", None),
                bos_id=tok.bos_token_id, eos_id=tok.eos_token_id,
                total_test_tokens=int(enc.input_ids.shape[1]),
                used_tokens=int(ids.shape[0]),
                first_16_ids=ids[:16].tolist(),
                ids_sha=sha(ids),
                bos_prepended=bool(ids[0].item() == tok.bos_token_id))
            with open(os.path.join(ART, "current_tokenization_dump.json"),
                      "w") as f:
                json.dump(tok_dump, f, indent=2)
        else:
            # quantized-module inventory
            rows = []
            for nm, m in model.base_model.named_modules():
                if isinstance(m, fq.FakeW4A4Linear) or \
                        type(m).__name__ in ("QuantizeLinear",):
                    rows.append(dict(
                        module=nm, cls=type(m).__name__,
                        w_bits=getattr(m, "w_bits", None),
                        a_bits=getattr(m, "a_bits", None),
                        n_forward=getattr(m, "n_forward", None)))
            # SpinQuant-style quant wrappers from study build
            for nm, m in model.base_model.named_modules():
                aq = getattr(m, "quantizer", None) or \
                    getattr(m, "act_quantizer", None)
                if aq is not None and not any(r["module"] == nm
                                              for r in rows):
                    rows.append(dict(module=nm, cls=type(m).__name__,
                                     w_bits=getattr(aq, "bits", None),
                                     a_bits=None, n_forward=None))
            with open(os.path.join(
                    ART, "current_quantized_module_inventory.csv"),
                    "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
            rot_rows = [dict(name=k, sha=sha(v), shape=list(v.shape))
                        for k, v in stash.items()
                        if torch.is_tensor(v) and v.dim() >= 1]
            R = torch.load(study.r_bin_path("random_hadamard", 0,
                                            paths["target_path"], rr),
                           map_location="cpu", weights_only=False)
            for k, v in R.items():
                if torch.is_tensor(v):
                    rot_rows.append(dict(name=f"R.bin:{k}", sha=sha(v),
                                         shape=list(v.shape)))
            with open(os.path.join(ART, "current_rotation_inventory.csv"),
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "sha", "shape"])
                w.writeheader(); w.writerows(rot_rows)
            contract["quant_recipe"] = dict(
                weights="per-output-channel symmetric RTN + MSE clip "
                        "(SpinQuant WeightQuantizer, w_clip)",
                acts="per-token asymmetric, reduction over final dim only",
                rotation="random Hadamard seed 0 (R1 + per-layer R2 folded; "
                         "R4 online where configured)",
                quant_cfg=str(study.QUANT_CFGS.get("w4a4")))
        del model
        torch.cuda.empty_cache()

    contract["reproduction"] = results
    contract["original_values"] = dict(
        fp16=dict(ce=1.92539, ppl=6.8578),
        rh0_w4a4=dict(ce=2.33469, ppl=10.3263), n_tokens=32767)
    with open(os.path.join(ART, "current_ppl_execution_contract.json"),
              "w") as f:
        json.dump(contract, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print("[audit] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
