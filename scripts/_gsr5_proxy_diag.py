#!/usr/bin/env python
"""Quantization-proxy diagnostics under GS (secondary, §19.8).

For GS + reused R_T vs GS + learned R5 (same GS m, same quantizer,
same fold path — only the draft residual basis differs):

  weight-side per QSITE: W4 NMSE ||Q(W)-W||^2/||W||^2, absmax,
    excess kurtosis of the pre-quant folded weight;
  activation-side per QSITE: input absmax + excess kurtosis captured by
    forward-pre-hooks on the deployed quantized model over 8 mtbench
    prompts (greedy, 64 new tokens, official tree).

Writes <run>/geometry/gs_proxy_diag.json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_eagle_acceptance_length import run_gen  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_G = D ** 0.42
D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G)


def grab(ea, ad):
    return dict(
        W_first=ad.split.projection_first_preR.w_fake,
        W_rec=ad.split.projection_recurrent_preR.w_fake,
        q=ea.layers[0].self_attn.q_proj.w_fake,
        k=ea.layers[0].self_attn.k_proj.w_fake,
        v=ea.layers[0].self_attn.v_proj.w_fake,
        o=ea.layers[0].self_attn.o_proj.w_fake,
        gate=ea.layers[0].mlp.gate_proj.w_fake,
        up=ea.layers[0].mlp.up_proj.w_fake,
        down=ea.layers[0].mlp.down_proj.w_fake)


def wstats(w):
    x = w.detach().float().flatten()
    mu = x.mean()
    c = x - mu
    s2 = (c * c).mean()
    k = float((c ** 4).mean() / (s2 * s2) - 3.0)
    return dict(absmax=float(x.abs().max()), excess_kurtosis=k)


class ActStat:
    def __init__(self):
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.absmax = 0.0

    def update(self, x):
        x = x.detach().reshape(-1).double()
        self.n += x.numel()
        self.s1 += float(x.sum())
        self.s2 += float((x ** 2).sum())
        self.s3 += float((x ** 3).sum())
        self.s4 += float((x ** 4).sum())
        self.absmax = max(self.absmax, float(x.abs().max()))

    def result(self):
        if self.n == 0:
            return None
        m = self.s1 / self.n
        m2 = self.s2 / self.n - m * m
        m3 = self.s3 / self.n - 3 * m * self.s2 / self.n + 2 * m ** 3
        m4 = (self.s4 / self.n - 4 * m * self.s3 / self.n
              + 6 * m * m * self.s2 / self.n - 3 * m ** 4)
        return dict(absmax=round(self.absmax, 4),
                    excess_kurtosis=round(m4 / (m2 * m2) - 3.0, 4),
                    n=self.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rd-ckpt", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    ea = model.ea_layer
    R_T = stash["R1"].clone()
    R5 = torch.load(args.rd_ckpt, map_location="cpu",
                    weights_only=False)["R_D"].double()
    from eagle_spinquant import eagle_bridge
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")

    out = {}
    for tag, R in (("GS_RT", R_T.double()), ("GS_R5", R5)):
        st = dict(stash)
        st["R1"] = R

        # pre-quant folded weights: identity-patch the weight quantizer
        orig = fq._weight_fake_quant
        fq._weight_fake_quant = lambda w, b: w.clone()
        ad_fp = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=R_T, **D4GS)
        ad_fp.install()
        W_fp = {k: v.detach().float().cpu().clone()
                for k, v in grab(ea, ad_fp).items()}
        ad_fp.uninstall()
        fq._weight_fake_quant = orig
        torch.cuda.empty_cache()

        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=R_T, **D4GS)
        ad.install()
        W_q = grab(ea, ad)

        sites = {}
        for name in W_fp:
            w, qw = W_fp[name], W_q[name].detach().float().cpu()
            nmse = float(((qw - w) ** 2).sum() / (w ** 2).sum())
            sites[name] = dict(w4_nmse=round(nmse, 6), **{
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in wstats(w).items()})

        # activation stats via pre-hooks on the deployed modules
        stats, hooks = {}, []
        mods = dict(W_first=ad.split.projection_first_preR,
                    W_rec=ad.split.projection_recurrent_preR,
                    q=ea.layers[0].self_attn.q_proj,
                    k=ea.layers[0].self_attn.k_proj,
                    v=ea.layers[0].self_attn.v_proj,
                    o=ea.layers[0].self_attn.o_proj,
                    gate=ea.layers[0].mlp.gate_proj,
                    up=ea.layers[0].mlp.up_proj,
                    down=ea.layers[0].mlp.down_proj)
        for name, m in mods.items():
            stats[name] = ActStat()

            def mk(nm):
                def h(mod, inp):
                    stats[nm].update(inp[0])
                return h
            hooks.append(m.register_forward_pre_hook(mk(name)))
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            run_gen(model.ea_generate(
                ids, temperature=0.0,
                max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
        for h in hooks:
            h.remove()
        for name in sites:
            if name in stats:
                sites[name]["act"] = stats[name].result()
        ad.uninstall()
        torch.cuda.empty_cache()
        out[tag] = sites
        print(f"[proxy] {tag} done", flush=True)

    dst = os.path.join(args.run_dir, "geometry", "gs_proxy_diag.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_G, rd_ckpt=args.rd_ckpt,
                   n_prompts=args.n_prompts,
                   max_new_tokens=args.max_new_tokens,
                   note="weight NMSE/kurtosis/absmax on pre-quant "
                        "folded weights; act stats pre-act-quant on "
                        "deployed W4A4 model; GS both legs",
                   legs=out), open(dst, "w"), indent=1)
    print(f"[proxy] -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
