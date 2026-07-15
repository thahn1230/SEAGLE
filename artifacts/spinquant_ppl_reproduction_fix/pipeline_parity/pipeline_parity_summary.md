# Pipeline parity: official ptq_model vs in-process build
- weight hashes: 451 identical / 0 different (of 451 matched modules)
- quantizer-config diffs: 0
- first activation divergence >5e-3 rel-L2: {"hook": "final_norm", "rel_l2": 0.479629784822464, "max_abs": 3.44580078125}
- logits diff: {'hook': 'logits', 'rel_l2': 0.467488, 'max_abs': 11.24707}

Same chat checkpoint, same seed-0 R.bin, W4A4KV16 RTN+clip, identical fixed 4x256 wikitext batch. Weight hashes compare the POST-fake-quant fp16 tensors; bitwise equality is expected only if both pipelines apply identical rotation folding order and RTN clip search; small fp differences otherwise show up in the activation table instead.
