#!/usr/bin/env python
"""Quantization-grid forensics: R5 PTQ (anchor master) vs R5 QAT
(post-QAT master), identical GS + R5 + quantizer.

Per quantized module reports:
  FP/master folded weight: RMS, absmax, excess kurtosis, relative
    weight movement ||W_post - W_pre||_F / ||W_pre||_F
  W4 grid: per-channel scale stats, W4 code-flip fraction (elements
    whose 4-bit code differs after QAT, computed on the shared
    per-channel symmetric grid of each arm), W4 reconstruction NMSE
  Activations (8 mtbench prompts, deployed W4A4 forward): input RMS /
    absmax / excess kurtosis per site.

Writes <run>/geometry/quant_grid_forensics.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eagle_spinquant import experiment, study, eagle_bridge
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts
from eval_eagle_acceptance_length import run_gen  # noqa: E402
from _gsr5_proxy_diag import ActStat  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_G = D ** 0.42
D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G)
SITES = dict(W_first="split.projection_first_preR",
             W_rec="split.projection_recurrent_preR",
             q="ea.q_proj", k="ea.k_proj", v="ea.v_proj", o="ea.o_proj",
             gate="ea.gate_proj", up="ea.up_proj", down="ea.down_proj")


def get_mod(ea, ad, name):
    if name.startswith("split."):
        return getattr(ad.split, name.split(".", 1)[1])
    attn = ea.layers[0].self_attn
    mlp = ea.layers[0].mlp
    short = name.split(".", 1)[1]
    return getattr(attn, short, None) or getattr(mlp, short)


def codes_sym4(w):
    """per-output-channel symmetric 4-bit integer codes + scales from a
    fake-quantized weight (values lie on a per-row uniform grid)."""
    scale = w.abs().amax(dim=1, keepdim=True) / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.round(w / scale).to(torch.int8), scale.squeeze(1)


def stats(w):
    x = w.float().flatten()
    mu = x.mean()
    c = x - mu
    s2 = (c * c).mean()
    return dict(rms=round(float(x.pow(2).mean().sqrt()), 6),
                absmax=round(float(x.abs().max()), 4),
                excess_kurtosis=round(
                    float((c ** 4).mean() / (s2 * s2) - 3.0), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--r5-ckpt", required=True)
    ap.add_argument("--qat-sd", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    ea_mod = model.ea_layer
    R5 = torch.load(args.r5_ckpt, map_location="cpu",
                    weights_only=False)["R_D"].double()
    R_T = stash["R1"].clone()
    anchor = torch.load(
        "checkpoints/eagle1_fresh_fp16_anchor/anchor.pt",
        map_location="cpu", weights_only=False)
    anchor = anchor.get("draft_state_dict", anchor.get("model", anchor))
    qsd = torch.load(args.qat_sd, map_location="cpu", weights_only=False)
    qsd = qsd.get("draft_state_dict", qsd.get("model", qsd))

    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")

    st = dict(stash)
    st["R1"] = R5
    legs = {}
    for leg, sd_over in (("pre_qat", anchor), ("post_qat", qsd)):
        base = {k: v.detach().cpu().clone()
                for k, v in ea_mod.state_dict().items()}
        for k in list(base.keys()):
            if k in sd_over:
                base[k] = sd_over[k].to(base[k].dtype)
        ea_mod.load_state_dict(base)

        # pre-quant folded weights
        orig = fq._weight_fake_quant
        fq._weight_fake_quant = lambda w, b: w.clone()
        ad_fp = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
            **D4GS)
        ad_fp.install()
        Wfp = {}

        class EA:
            pass
        for name in SITES:
            m = get_mod(ea_mod, ad_fp, SITES[name])
            Wfp[name] = m.w_fake.detach().float().cpu().clone()
        ad_fp.uninstall()
        fq._weight_fake_quant = orig
        torch.cuda.empty_cache()

        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
            **D4GS)
        ad.install()
        rec, acts, hooks = {}, {}, []
        for name in SITES:
            m = get_mod(ea_mod, ad, SITES[name])
            qw = m.w_fake.detach().float().cpu()
            cd, sc = codes_sym4(qw)
            nmse = float(((qw - Wfp[name]) ** 2).sum()
                         / (Wfp[name] ** 2).sum())
            rec[name] = dict(fp=stats(Wfp[name]),
                             w4_nmse=round(nmse, 6),
                             scale_mean=round(float(sc.mean()), 6),
                             scale_max=round(float(sc.max()), 6),
                             sat_frac=round(float(
                                 (cd.abs() == 7).float().mean()), 5))
            rec[name]["_codes"] = cd
            rec[name]["_wfp"] = Wfp[name]
            acts[name] = ActStat()

            def mk(nm):
                def h(mod, inp):
                    acts[nm].update(inp[0])
                return h
            hooks.append(m.register_forward_pre_hook(mk(name)))
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=72,
                tree_choices=tree), ids.shape[1], 64)
        for h in hooks:
            h.remove()
        for name in SITES:
            rec[name]["act"] = acts[name].result()
        ad.uninstall()
        torch.cuda.empty_cache()
        legs[leg] = rec

    out = {}
    for name in SITES:
        a, b = legs["pre_qat"][name], legs["post_qat"][name]
        wa, wb = a.pop("_wfp"), b.pop("_wfp")
        ca, cb = a.pop("_codes"), b.pop("_codes")
        move = float((wb - wa).norm() / wa.norm())
        flips = float((ca != cb).float().mean())
        out[name] = dict(pre=a, post=b,
                         rel_weight_movement=round(move, 6),
                         w4_code_flip_frac=round(flips, 5))
    dst = os.path.join(args.run_dir, "geometry",
                       "quant_grid_forensics.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_G, n_prompts=args.n_prompts,
                   note="codes on each arm's own per-channel symmetric "
                        "grid; flip fraction counts differing 4-bit "
                        "codes elementwise", sites=out),
              open(dst, "w"), indent=1)
    print("[forensics] ->", dst)
    for name, r in out.items():
        print(f"  {name}: move={r['rel_weight_movement']:.4f} "
              f"flips={r['w4_code_flip_frac']:.4f} "
              f"nmse {r['pre']['w4_nmse']:.5f}->{r['post']['w4_nmse']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
