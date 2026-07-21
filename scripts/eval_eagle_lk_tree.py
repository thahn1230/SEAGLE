#!/usr/bin/env python
"""M3: production EAGLE tree greedy micro-AL for an LK candidate.

Same runtime as the validated eval_eagle_draft_rotation.py (concat-selective
adapter, folded bridge, P3 alpha, target KV4 via persistent past), extended
for LK checkpoints:

  --lk-ckpt FILE.pt  loads {R_D, alpha, core_weights?}. R_D becomes the
  draft gauge (stash R1), first_fold_R keeps R_T. If core_weights are
  present (draft-core QAT candidates I/J), the trained quantized weights
  are INJECTED into the installed runtime modules' w_fake buffers via
  ExactQATRotatedDraft.quantized_weights() — the exact fold-and-quantize
  deployment of the trained draft (Gate D proves this equals the runtime
  construction).

Writes shards/lktree__<tag>__<target>__<ds>.csv (same schema as the
previous study, micro-AL only).
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.kv4_cache import install_kv4_on_past, DraftKV4Patch
from eagle_spinquant.eval_datasets import load_eval_prompts
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
TARGETS = {"t8": ("w8a8", 16), "t4": ("w4a4", 16), "t4kv4": ("w4a4", 4)}
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True)


def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        final = out
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= mx:
            break
    return deltas


def inject_core(ad, model, ck, sd0, R_T, gamma, W_lm, dev, kv_bits):
    """Overwrite installed runtime quantized weights with the trained
    draft-core fold-and-quantize (candidates I/J)."""
    from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
    from eagle_spinquant.residual_rotation import FullRotation
    rot = FullRotation(ck["R_D"].float())
    eq = ExactQATRotatedDraft(
        sd0, R_T, gamma, W_lm, rot.to(dev),
        alpha_init=float(ck.get("alpha", 32.0)), w_bits=4, a_bits=4,
        draft_kv_bits=kv_bits, device=dev, first_fold_R=R_T)
    with torch.no_grad():
        for n in ck["core_weights"]:
            getattr(eq, n).copy_(ck["core_weights"][n].to(dev))
    qw, _ = eq.quantized_weights()
    ea = model.ea_layer
    tgt = dict(W_first=ad.split.projection_first_preR,
               W_rec=ad.split.projection_recurrent_preR,
               q=ea.layers[0].self_attn.q_proj,
               k=ea.layers[0].self_attn.k_proj,
               v=ea.layers[0].self_attn.v_proj,
               o=ea.layers[0].self_attn.o_proj,
               gate=ea.layers[0].mlp.gate_proj,
               up=ea.layers[0].mlp.up_proj,
               down=ea.layers[0].mlp.down_proj)
    with torch.no_grad():
        for n, mod in tgt.items():
            mod.w_fake.copy_(qw[n].to(mod.w_fake.dtype))
    del eq
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lk-ckpt", default="shared",
                    help="'shared' or checkpoint .pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", default="t4kv4", choices=list(TARGETS))
    ap.add_argument("--datasets",
                    default="mtbench,sharegpt,c4,gsm8k,humaneval")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--alpha-override", type=float, default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    quant, t_kv = TARGETS[args.target]
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, quant, 0, device=dev, rotations_root=rr)
    if t_kv < 16:
        past, pkv_data, cur_len = initialize_past_key_values(
            model.base_model)
        model.past_key_values = past
        model.past_key_values_data = pkv_data
        model.current_length_data = cur_len
        install_kv4_on_past(past, bits=t_kv)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]

    R_T = stash["R1"].clone()
    sd0 = {k: v.detach().cpu().clone()
           for k, v in model.ea_layer.state_dict().items()}
    kw = dict(D4P3)
    ck = None
    if args.lk_ckpt == "shared":
        kw["embed_scale_alpha"] = 32.0
    else:
        ck = torch.load(args.lk_ckpt, map_location="cpu",
                        weights_only=False)
        stash = dict(stash)
        stash["R1"] = ck["R_D"].double()
        kw["first_fold_R"] = R_T
        kw["embed_scale_alpha"] = float(ck.get("alpha", 32.0))
    if args.alpha_override is not None:
        kw["embed_scale_alpha"] = args.alpha_override

    for ds_name in args.datasets.split(","):
        out_csv = os.path.join(
            args.run_dir, "shards",
            f"lktree__{args.tag}__{args.target}__{ds_name}.csv")
        if os.path.exists(out_csv):
            print(f"[lktree] {args.tag}/{ds_name}: exists, skip",
                  flush=True)
            continue
        prompts, _man = load_eval_prompts(ds_name, args.n_prompts, "eval")
        ids_list = [build_prompt(tok, p["text"])[:, :1024].to(dev)
                    for p in prompts]
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, **kw)
        ad.install()
        if ck is not None and "core_weights" in ck:
            inject_core(ad, model, ck, sd0, R_T, stash["gamma_f"],
                        stash["lm_head_weight"].float(), dev, 4)
        dpatch = DraftKV4Patch(model.ea_layer, bits=4).install()
        rows = []
        for pi, ids in enumerate(ids_list):
            if hasattr(ad, "set_context"):
                ad.set_context(prompts[pi]["row_id"])
            deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            rows.append(dict(tag=args.tag, target=args.target,
                             dataset=ds_name,
                             prompt_id=prompts[pi]["row_id"],
                             acceptance_list=json.dumps(deltas),
                             n_cycles=len(deltas)))
        dpatch.uninstall()
        ad.uninstall()
        taus = [t for r in rows for t in json.loads(r["acceptance_list"])]
        mal = sum(taus) / max(len(taus), 1)
        logging_utils.write_csv(out_csv, rows)
        print(f"[lktree] {args.tag}/{args.target}/{ds_name}: "
              f"micro-AL={mal:.4f} ({len(taus)} cycles)", flush=True)
    print(f"[lktree] {args.tag} DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
