#!/usr/bin/env python
"""Phase 7 (GATE D) part 1+2: dump transformed-weight hashes, quantizer
configs, and layerwise activations for EITHER pipeline:

  --pipeline official   : vendored SpinQuant eval_utils.main.ptq_model
                          (run under the pinned venv, transformers 4.44.2)
  --pipeline inprocess  : project study.build_study_target (system python)

Both use the SAME chat checkpoint, SAME R.bin (project random-Hadamard
seed 0), W4A4KV16, RTN + w_clip. A fixed 4x256-token wikitext batch (saved
by the first caller) drives layer-output capture via forward hooks on every
decoder layer plus final norm + lm_head logits.

Saves <out>/{tag}_dump.pt with:
  weights: {module_name: sha16 of fp32 weight bytes}
  quantcfg: {module_name: str(quantizer config)}
  acts: {hook_name: fp32 tensor (first batch element, first 64 positions)}
  logits: fp32 (1, 64, vocab)
"""
import argparse, hashlib, os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "artifacts", "spinquant_ppl_reproduction_fix",
                   "pipeline_parity")
CHAT = "/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"


def sha(t):
    return hashlib.sha256(
        t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def fixed_batch(tok_path, dev):
    p = os.path.join(OUT, "fixed_batch.pt")
    if os.path.exists(p):
        return torch.load(p, map_location=dev, weights_only=True)
    from datasets import load_dataset
    from transformers import LlamaTokenizerFast
    tok = LlamaTokenizerFast.from_pretrained(
        tok_path, add_bos_token=False, add_eos_token=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    b = ids[0, : 4 * 256].view(4, 256)
    torch.save(b, p)
    return b.to(dev)


def hook_capture(model_root, layers, final_norm, lm_head, batch, dev):
    acts = {}
    hooks = []
    def mk(name):
        def h(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            acts[name] = t[0, :64].detach().float().cpu()
        return h
    for i, ly in enumerate(layers):
        hooks.append(ly.register_forward_hook(mk(f"layer{i:02d}")))
    if final_norm is not None:
        hooks.append(final_norm.register_forward_hook(mk("final_norm")))
    with torch.no_grad():
        out = model_root(batch.to(dev))
        logits = (out.logits if hasattr(out, "logits") else out[0])
    acts["logits"] = logits[0, :64].detach().float().cpu()
    for h in hooks:
        h.remove()
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True,
                    choices=["official", "inprocess"])
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    dev = args.device

    if args.pipeline == "official":
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party",
                                        "SpinQuant"))
        from types import SimpleNamespace
        import transformers
        from eval_utils.main import ptq_model
        cfg = transformers.AutoConfig.from_pretrained(CHAT)
        model = transformers.LlamaForCausalLM.from_pretrained(
            CHAT, config=cfg, torch_dtype=torch.float16)
        model.cuda()
        pa = SimpleNamespace(
            rotate=True, rotate_mode="hadamard",
            optimized_rotation_path=args.rbin,
            w_bits=4, a_bits=4, k_bits=16, v_bits=16,
            w_clip=True, w_rtn=True, w_asym=False, w_groupsize=-1,
            a_asym=True, a_groupsize=-1, a_clip_ratio=1.0,
            k_asym=True, v_asym=True, k_groupsize=128, v_groupsize=128,
            k_clip_ratio=1.0, v_clip_ratio=1.0,
            int8_down_proj=False, nsamples=128, percdamp=0.01,
            act_order=False, seed=0,
            capture_layer_io=False, layer_idx=0, load_qmodel_path=None,
            save_qmodel_path=None, export_to_et=False)
        ma = SimpleNamespace(input_model=CHAT, access_token=None)
        model = ptq_model(pa, model, ma)
        model = model.to(dev)
        root, layers = model, model.model.layers
        fn, head = model.model.norm, model.lm_head
        tag = "official"
    else:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
        from eagle_spinquant import experiment, study
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        rr = cfg.get("paths", {}).get("rotations_root")
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "full", "random_hadamard", "w4a4", 0, device=dev,
            rotations_root=rr)
        bm = model.base_model
        bm.model.tree_mask = None
        root, layers = bm, bm.model.layers
        fn, head = bm.model.norm, bm.lm_head
        tag = "inprocess"

    weights, quantcfg = {}, {}
    for nm, m in root.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and w.dim() == 2:
            weights[nm] = sha(w)
        for qn in ("quantizer", "act_quantizer", "weight_quantizer",
                   "_weight_fake_quant"):
            q = getattr(m, qn, None)
            if q is not None:
                quantcfg[f"{nm}.{qn}"] = repr(
                    {k: v for k, v in vars(q).items()
                     if isinstance(v, (int, float, bool, str))})
    batch = fixed_batch(CHAT, dev)
    acts = hook_capture(root, layers, fn, head, batch, dev)
    torch.save(dict(weights=weights, quantcfg=quantcfg, acts=acts),
               os.path.join(OUT, f"{tag}_dump.pt"))
    print(f"[dump] {tag}: {len(weights)} weights, {len(quantcfg)} qcfgs, "
          f"{len(acts)} acts saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
