#!/usr/bin/env python
"""Phase 10 distribution capture for the bitwidth-AL component study.

--weights (CPU): build the concat first projection in BOTH first-hidden
  modes and record per-input-channel absmax over the concat input dim:
    identity : [W_e | W_h]            (T16_Dq contract, h_t input)
    gamma_R1 : [W_e | W_h·D_γ·R1]     (Tq_Dq contract, a_t input)
  plus the recurrent block [W_e | W_h·R1]. Answers Q15 (gamma folding vs
  first-weight outliers) and e-vs-h block dominance.

--acts (GPU): run N prompts through (a) the stock target capturing h_t
  (draft first-projection hidden input, identity mode) and e (embedding
  output), and (b) the fused target capturing a_t. Records per-channel
  absmax + per-token max/kurtosis of each tensor. Answers Q13/H7 (is the
  identity-mode damage activation-outlier-driven?).

Writes artifacts/bitwidth_al_component_causality/distributions/*.npz + a
summary json.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch

ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality",
                   "distributions")


def chan_stats(x):
    """x: (n_tokens, feat). Returns per-channel absmax and global stats."""
    ax = x.abs()
    per_chan = ax.max(dim=0).values.float().cpu().numpy()
    flat = x.float().flatten()
    mu, sd = flat.mean(), flat.std()
    kurt = float((((flat - mu) / (sd + 1e-12)) ** 4).mean())
    return per_chan, dict(absmax=float(ax.max()),
                          mean_abs=float(ax.mean()),
                          kurtosis=round(kurt, 2),
                          p999=float(ax.flatten().kthvalue(
                              max(1, int(0.999 * ax.numel()))).values))


def weights_capture(summary):
    from eagle_spinquant import experiment, study
    from eagle_spinquant.concat_selective_projection import (
        build_concat_selective_weights)
    from safetensors.torch import load_file
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path("random_hadamard", 0,
                                    paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R1 = R["R1"].float()
    # draft state dict (fc weight) — single-file checkpoint
    dp = paths["draft_path"]
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(dp, fn)
        if os.path.exists(p):
            sd = load_file(p) if fn.endswith("safetensors") else \
                torch.load(p, map_location="cpu", weights_only=True)
            break
    assert sd is not None, dp
    # gamma from the stock target final norm
    import glob
    from safetensors.torch import safe_open
    gamma = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys():
                gamma = f.get_tensor("model.norm.weight").float()
                break
    assert gamma is not None
    out = {}
    for mode_name in ("identity", "gamma_R1"):
        W_first, W_rec, b = build_concat_selective_weights(
            sd, R1, gamma, first_hidden_mode=mode_name)
        for nm, W in ((f"first_{mode_name}", W_first),) + (
                (("recurrent", W_rec),) if mode_name == "gamma_R1" else ()):
            per_in = W.float().abs().max(dim=0).values.cpu().numpy()
            h = W.shape[1] // 2
            out[f"{nm}__per_in_absmax"] = per_in
            summary[f"weights.{nm}"] = dict(
                e_block_absmax=float(per_in[:h].max()),
                h_block_absmax=float(per_in[h:].max()),
                e_block_mean=float(per_in[:h].mean()),
                h_block_mean=float(per_in[h:].mean()),
                e_over_h_absmax_ratio=round(
                    float(per_in[:h].max() / (per_in[h:].max() + 1e-12)), 3))
    np.savez_compressed(os.path.join(ART, "weight_channel_absmax.npz"), **out)
    print("[dist] weights captured", flush=True)


@torch.no_grad()
def acts_capture(summary, num_prompts, dev):
    from eagle_spinquant import eagle_bridge, experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(num_prompts)
    out = {}
    for tag, rot, q, hs_name in (("stock_h_t", "none", "none", "h"),
                                 ("fused_a_t", "full", "w4a4", "a")):
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, "random_hadamard", q, 0, device=dev, rotations_root=rr)
        tok = eagle_bridge.get_tokenizer(model)
        hs_all, emb_all = [], []
        for p in prompts:
            ids = build_prompt(tok, p["text"]).to(dev)
            model.base_model.model.tree_mask = None
            o = model.base_model.model(ids, output_hidden_states=False)
            hs_all.append(o[0][0].float().cpu())          # last hidden (pre-head)
            emb_all.append(model.ea_layer.embed_tokens(ids)[0].float().cpu())
        hs = torch.cat(hs_all)
        em = torch.cat(emb_all)
        pc, st = chan_stats(hs)
        out[f"{tag}__hidden_per_chan_absmax"] = pc
        summary[f"acts.{tag}.hidden"] = st
        pc, st = chan_stats(em)
        out[f"{tag}__embed_per_chan_absmax"] = pc
        summary[f"acts.{tag}.embed"] = st
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(ART, "act_channel_absmax.npz"), **out)
    print("[dist] activations captured", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--acts", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    sp = os.path.join(ART, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    if args.weights:
        weights_capture(summary)
    if args.acts:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
        torch.set_grad_enabled(False)
        acts_capture(summary, args.num_prompts, args.device)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
