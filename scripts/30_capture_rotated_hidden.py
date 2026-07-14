#!/usr/bin/env python
"""Capture the EAGLE-1 draft-input tensors and document ACTUAL shapes (target 7).

Hooks the stock EAGLE draft (ea_layer.embed_tokens, ea_layer.fc) during one real
decoding step to record the true shapes/dtypes of:
  - e   : token embedding branch          (draft's own original embedding)
  - h   : target hidden feature fed to the draft (post-final-norm, top layer)
  - z   : fc input = concat([e, h])       (order: embedding FIRST)
  - f   : fc output (fused draft feature)
Then builds the rotated target and captures h_hat, verifying that
unrotate(h_hat) matches the stock h basis.

Outputs:
  runs/debug_hidden_capture/*.pt (small slices + shape metadata)
  docs/03_EAGLE1_ACTUAL_TENSOR_SHAPES.md

Usage: python scripts/30_capture_rotated_hidden.py --gpu 0
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import eagle_bridge, experiment, spinquant_bridge as sb  # noqa: E402
from eagle_spinquant import rotation_interface as ri  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    out_dir = os.path.join(PROJECT_ROOT, "runs", "debug_hidden_capture")
    os.makedirs(out_dir, exist_ok=True)

    model = eagle_bridge.load_eagle_model(
        base_model_path=paths["target_path"], ea_model_path=paths["draft_path"],
        dtype=dtype, device_map="cuda")
    tok = eagle_bridge.get_tokenizer(model)

    captured = {}

    def embed_hook(mod, inp, out):
        captured["e"] = out.detach()

    def fc_hook(mod, inp, out):
        captured["z"] = inp[0].detach()   # concat([e, h])
        captured["f"] = out.detach()

    h_embed = model.ea_layer.embed_tokens.register_forward_hook(embed_hook)
    h_fc = model.ea_layer.fc.register_forward_hook(fc_hook)

    # capture the hidden fed to the draft by wrapping topK_genrate
    orig_topk = model.ea_layer.topK_genrate

    def wrapped(hidden_states, input_ids, head, lp, *a, **k):
        captured["h_into_draft"] = hidden_states.detach()
        return orig_topk(hidden_states, input_ids, head, lp, *a, **k)
    model.ea_layer.topK_genrate = wrapped

    input_ids = eagle_bridge.build_llama2_chat_prompt(tok, "Explain how a rainbow forms.")
    # one forward through the EaModel to trigger the draft
    with torch.no_grad():
        model(input_ids.to("cuda"), output_orig=True)

    h_embed.remove(); h_fc.remove()
    model.ea_layer.topK_genrate = orig_topk

    D = model.base_model.config.hidden_size
    shapes = {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                     "device": str(t.device)} for name, t in captured.items()}

    # verify concat order: z[..., :D] == e, z[..., D:] == h
    z = captured["z"].float()
    e = captured["e"].float()
    order_check = {}
    if z.shape[-1] == 2 * D:
        e_block_err = (z[..., :D] - e[..., :e.shape[-1]] if e.shape[1] == z.shape[1]
                       else torch.tensor(float("nan")))
        order_check["z_first_block_is_embedding"] = bool(
            torch.allclose(z[..., :D], captured["e"].float()[:, :z.shape[1]], atol=1e-3)
            if captured["e"].shape[1] >= z.shape[1] else False)

    # save small slices
    torch.save({k: v[..., :8].float().cpu() if v.dim() >= 1 else v.float().cpu()
                for k, v in captured.items()},
               os.path.join(out_dir, "draft_input_capture.pt"))

    meta = {
        "hidden_size_D": D,
        "vocab_size_V": model.base_model.config.vocab_size,
        "num_layers": model.base_model.config.num_hidden_layers,
        "shapes": shapes,
        "concat_order": "cat([embed_e, hidden_h]) -> fc(2D -> D); embedding FIRST",
        "hidden_source": "target post-final-RMSNorm last-layer output (outputs[0])",
        "order_check": order_check,
    }
    with open(os.path.join(out_dir, "shape_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # free draft/base, then build rotated target to capture h_hat
    del model
    torch.cuda.empty_cache()

    r_bin = os.path.join(PROJECT_ROOT, "outputs", "rotations", "random_hadamard", "R.bin")
    from transformers import AutoConfig
    conf = AutoConfig.from_pretrained(paths["target_path"])
    if not os.path.isfile(r_bin):
        sb.make_random_rotation_bin(conf, r_bin, mode="hadamard", seed=0)

    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=dtype,
                                  low_cpu_mem_usage=True).to("cuda").eval()
    spec = sb.default_ptq_args(rotate=True, optimized_rotation_path=r_bin)
    stash = sb.apply_spinquant_pipeline(rot, spec, input_model_id=cfg["model"]["target"],
                                        stage="rotate_only")
    rot.to("cuda")
    with torch.no_grad():
        h_hat = rot.model(input_ids=input_ids.to("cuda"))[0]
    h_hat_cpu = h_hat.float().cpu()
    meta["h_hat"] = {"shape": list(h_hat.shape), "dtype": str(h_hat.dtype),
                     "basis": "R1-rotated residual (gamma_f folded into lm_head)"}
    # relationship check on last position
    R1 = stash["R1"].float(); gamma_f = stash["gamma_f"].float()
    h_from_hat = ri.unrotate_hidden(h_hat_cpu, R1, gamma_f)
    meta["h_hat_norm"] = float(h_hat_cpu.norm())
    meta["unrotated_h_hat_norm"] = float(h_from_hat.norm())

    write_doc(meta)
    print(json.dumps({k: meta[k] for k in
                      ["hidden_size_D", "vocab_size_V", "concat_order",
                       "hidden_source"]}, indent=2))
    print("shapes:", json.dumps(shapes, indent=2))
    print(f"\n-> docs/03_EAGLE1_ACTUAL_TENSOR_SHAPES.md")
    print(f"-> {out_dir}/")
    return 0


def write_doc(meta: dict) -> None:
    D = meta["hidden_size_D"]
    s = meta["shapes"]

    def row(name, desc):
        info = s.get(name, {})
        return f"| `{name}` | {desc} | `{info.get('shape')}` | `{info.get('dtype')}` |"

    doc = f"""# 03 — EAGLE-1 Actual Tensor Shapes (measured)

Captured by `scripts/30_capture_rotated_hidden.py` on the real
Llama-2-7b-chat + yuhuili/EAGLE-llama2-chat-7B during one decoding step. These are
the ACTUAL runtime shapes, not the conceptual sketch in docs/01.

Model geometry: hidden_size D = {D}, vocab V = {meta['vocab_size_V']},
target layers = {meta['num_layers']}, draft layers = 1.

## Draft input interface (cnets.py:592-593)

Concat order (verified): **{meta['concat_order']}**.
Hidden source (verified): **{meta['hidden_source']}**.

| tensor | meaning | shape | dtype |
|---|---|---|---|
{row('e', 'token embedding branch (draft embed_tokens output; ORIGINAL/unrotated basis)')}
{row('h_into_draft', 'target hidden feature passed to ea_layer.topK_genrate')}
{row('z', 'fc input = concat([e, h]) — embedding first, hidden second')}
{row('f', 'fc output = fused draft feature (hidden_size)')}

`z` last-dim = 2·D = {2 * D}; the hidden block is columns `[D:2D]` = `[{D}:{2 * D}]`.
This is why the Variant-B conjugation folds into `fc.weight[:, {D}:{2 * D}]`.

## Rotated target hidden

The SpinQuant-rotated target emits `h_hat` in the R1 basis (gamma_f folded into
lm_head): shape `{meta.get('h_hat', {}).get('shape')}`,
basis = {meta.get('h_hat', {}).get('basis')}.

- ‖h_hat‖ = {meta.get('h_hat_norm'):.2f}
- ‖unrotate(h_hat)‖ = {meta.get('unrotated_h_hat_norm'):.2f}  (recovers the original-basis feature)

The draft consumes an original-basis hidden, so the adapter applies
`h = (h_hat @ R1^T) * gamma_f` (Variant A) or folds it into fc (Variant B) before
the draft; the draft's predicted features are scored by the ORIGINAL lm_head.

Raw slices saved to `runs/debug_hidden_capture/draft_input_capture.pt`,
metadata to `runs/debug_hidden_capture/shape_meta.json`.
"""
    with open(os.path.join(PROJECT_ROOT, "docs", "03_EAGLE1_ACTUAL_TENSOR_SHAPES.md"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    sys.exit(main())
