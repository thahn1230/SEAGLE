#!/usr/bin/env python
"""Phase 7 (GATE D) part 3: CPU comparison of official vs inprocess dumps.

Outputs into pipeline_parity/:
  module_inventory.csv        matched/unmatched module names
  transformed_weight_diff.csv per-module hash match (weights are fake-quant
                              OUTPUTS in both pipelines)
  quantizer_config_diff.csv
  layer_output_diff.csv       per-layer rel-L2 / max-abs on the fixed batch
  first_divergence.json
  pipeline_parity_summary.md
"""
import csv, json, os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "artifacts", "spinquant_ppl_reproduction_fix",
                   "pipeline_parity")

# in-process module names live under EaModel.base_model; strip nothing —
# both dumps used the LlamaForCausalLM root, so names should align directly.
ALIAS = {}


def main():
    off = torch.load(os.path.join(OUT, "official_dump.pt"),
                     map_location="cpu", weights_only=False)
    inp = torch.load(os.path.join(OUT, "inprocess_dump.pt"),
                     map_location="cpu", weights_only=False)
    wo, wi = off["weights"], inp["weights"]
    names = sorted(set(wo) | set(ALIAS.get(n, n) for n in wi))
    inv, wrows = [], []
    n_match = n_diff = 0
    for n in names:
        ho, hi = wo.get(n), wi.get(n)
        inv.append(dict(module=n, in_official=ho is not None,
                        in_inprocess=hi is not None))
        if ho and hi:
            same = ho == hi
            n_match += same; n_diff += (not same)
            wrows.append(dict(module=n, official_sha=ho, inprocess_sha=hi,
                              identical=same))
    _w("module_inventory.csv", inv)
    _w("transformed_weight_diff.csv", wrows)

    qrows = []
    qo, qi = off["quantcfg"], inp["quantcfg"]
    for n in sorted(set(qo) | set(qi)):
        if qo.get(n) != qi.get(n):
            qrows.append(dict(name=n, official=qo.get(n, "-"),
                              inprocess=qi.get(n, "-")))
    _w("quantizer_config_diff.csv", qrows)

    ao, ai = off["acts"], inp["acts"]
    lrows, first_div = [], None
    for n in sorted(set(ao) & set(ai)):
        a, b = ao[n], ai[n]
        if a.shape != b.shape:
            lrows.append(dict(hook=n, rel_l2="shape-mismatch",
                              max_abs=str(a.shape) + str(b.shape)))
            continue
        rel = float((a - b).norm() / (a.norm() + 1e-9))
        mx = float((a - b).abs().max())
        lrows.append(dict(hook=n, rel_l2=round(rel, 6),
                          max_abs=round(mx, 5)))
        if first_div is None and rel > 5e-3:
            first_div = dict(hook=n, rel_l2=rel, max_abs=mx)
    _w("layer_output_diff.csv", lrows)
    with open(os.path.join(OUT, "first_divergence.json"), "w") as f:
        json.dump(first_div or {"none_above_5e-3": True}, f, indent=2)

    lg = next((r for r in lrows if r["hook"] == "logits"), None)
    md = [
        "# Pipeline parity: official ptq_model vs in-process build",
        f"- weight hashes: {n_match} identical / {n_diff} different "
        f"(of {n_match + n_diff} matched modules)",
        f"- quantizer-config diffs: {len(qrows)}",
        f"- first activation divergence >5e-3 rel-L2: "
        f"{json.dumps(first_div) if first_div else 'none'}",
        f"- logits diff: {lg}",
        "",
        "Same chat checkpoint, same seed-0 R.bin, W4A4KV16 RTN+clip, "
        "identical fixed 4x256 wikitext batch. Weight hashes compare the "
        "POST-fake-quant fp16 tensors; bitwise equality is expected only if "
        "both pipelines apply identical rotation folding order and RTN "
        "clip search; small fp differences otherwise show up in the "
        "activation table instead.",
    ]
    open(os.path.join(OUT, "pipeline_parity_summary.md"), "w").write(
        "\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def _w(fn, rows):
    p = os.path.join(OUT, fn)
    if not rows:
        open(p, "w").close(); return
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
