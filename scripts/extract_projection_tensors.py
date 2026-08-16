#!/usr/bin/env python
"""Projection-layer tensor extraction for the SEAGLE visualization
package (PTQ / QAT / RT canonical configs).

For each method, replays real MT-Bench generation through the actual
deployed adapter and captures:
  - first / recurrent projection INPUT activations (deployed form)
  - analytic ORIGINAL views (alpha unscaled e-half; h-half restored to
    the stock basis: h_t = (a_t R^T) * gamma_f for first,
    h_orig = h R_used^T for recurrent)
  - projection weights: original (state-dict fc halves) and deployed
    (pre-quant folded W_first/W_rec captured via the fq spy)
plus summary stats and a reproducible window-selection manifest.
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch

from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"
MAX_ROWS = 4096
WINDOW = 128


def run_gen(gen, ilen, mx):
    prev = ilen
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            prev = cur
        if cur - ilen >= mx:
            break


def kurtosis(x):
    x = x.reshape(-1).astype(np.float64)
    m, s = x.mean(), x.std() + 1e-12
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def stats(x):
    a = np.abs(x).astype(np.float64)
    ch = (a ** 2).sum(0)
    k = max(1, int(round(len(ch) * 0.001)))
    top = np.sort(ch)[-k:].sum() / max(ch.sum(), 1e-12)
    return dict(rms=float(np.sqrt((x.astype(np.float64) ** 2).mean())),
                absmax=float(a.max()), kurtosis=kurtosis(x),
                top01pct_channel_energy=float(top),
                shape=list(x.shape))


def median_energy_window(x, w=WINDOW):
    """Contiguous w-row window whose total energy is the median over
    all window starts (selection rule saved in the manifest)."""
    if x.shape[0] <= w:
        return 0, x.shape[0]
    e = (x.astype(np.float64) ** 2).sum(1)
    c = np.concatenate([[0.0], np.cumsum(e)])
    tot = c[w:] - c[:-w]
    start = int(np.argsort(tot)[len(tot) // 2])
    return start, start + w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", required=True,
                    choices=["SEAGLE_PTQ", "SEAGLE_QAT", "SEAGLE_RT"])
    ap.add_argument("--rd-ckpt", default=None)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", KIND, "w4a4", 0, device=dev,
        rotations_root=rr)
    ea = model.ea_layer
    D = ea.fc.weight.shape[0]

    if args.draft_sd:
        sd_new = torch.load(args.draft_sd, map_location="cpu",
                            weights_only=False)
        sd_new = sd_new.get("draft_state_dict",
                            sd_new.get("model", sd_new))
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                            for k, v in sd_new.items()}, strict=True)
        ea.to(dev)

    # original (pre-treatment) projection weight from the live sd
    W = ea.fc.weight.detach().float().cpu().numpy()
    w_first_orig = W.copy()          # [W_e | W_h], stock basis
    w_rec_orig = W.copy()

    st = dict(stash)
    R_T = stash["R1"].clone()
    kw = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True,
              embed_scale_alpha=args.alpha)
    R_used = R_T
    if args.rd_ckpt:
        ck = torch.load(args.rd_ckpt, map_location="cpu",
                        weights_only=False)
        st["R1"] = ck["R_D"].double()
        kw["first_fold_R"] = R_T
        R_used = ck["R_D"]

    captured_w = {}
    orig_wq = fq._weight_fake_quant

    def spy(w, bits=4, name=None):
        if name in ("projection_first_preR", "projection_recurrent_preR"):
            captured_w[name] = w.detach().float().cpu().numpy()
        return orig_wq(w, bits, name=name)

    fq._weight_fake_quant = spy
    try:
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, **kw)
        ad.install()
    finally:
        fq._weight_fake_quant = orig_wq

    acts = {"first": [], "recurrent": []}

    def mk_hook(key):
        def hook(mod, inputs):
            if sum(a.shape[0] for a in acts[key]) < MAX_ROWS:
                z = inputs[0]
                acts[key].append(
                    z.detach().float().reshape(-1, z.shape[-1]).cpu()
                    .numpy())
        return hook

    hs = [ad.split.projection_first_preR.register_forward_pre_hook(
              mk_hook("first")),
          ad.split.projection_recurrent_preR.register_forward_pre_hook(
              mk_hook("recurrent"))]

    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")
    used = []
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
        run_gen(model.ea_generate(ids, temperature=0.0,
                                  max_steps=args.max_new + 8,
                                  tree_choices=tree),
                ids.shape[1], args.max_new)
        used.append(p["row_id"])
        if all(sum(a.shape[0] for a in acts[k]) >= MAX_ROWS
               for k in acts):
            break
    for h in hs:
        h.remove()
    ad.uninstall()

    g = stash["gamma_f"].double().numpy()
    Rt = R_T.double().numpy()
    Ru = R_used.double().numpy()
    out = {}
    man = dict(method=args.method, alpha=args.alpha,
               rd_ckpt=args.rd_ckpt, draft_sd=args.draft_sd,
               prompts_dataset="mtbench:eval-pool",
               prompt_ids=used, max_new=args.max_new,
               window_rule=f"median-energy contiguous {WINDOW}-row "
                           "window over pooled captured rows",
               boundary_channel=D)
    for key in ("first", "recurrent"):
        z = np.concatenate(acts[key], 0)[:MAX_ROWS]
        lo, hi = median_energy_window(z)
        zw = z[lo:hi]
        # deployed ("after") view
        out[f"act_{key}_after"] = zw
        # analytic original view
        e_orig = zw[:, :D] / args.alpha
        if key == "first":
            h_orig = (zw[:, D:].astype(np.float64) @ Rt.T) * g[None, :]
        else:
            h_orig = zw[:, D:].astype(np.float64) @ Ru.T
        out[f"act_{key}_original"] = np.concatenate(
            [e_orig, h_orig.astype(np.float32)], 1)
        man[f"{key}_window"] = [int(lo), int(hi)]
        man[f"{key}_rows_captured"] = int(z.shape[0])
    out["w_first_original"] = w_first_orig
    out["w_first_after"] = captured_w["projection_first_preR"]
    out["w_rec_original"] = w_rec_orig
    out["w_rec_after"] = captured_w["projection_recurrent_preR"]

    allstats = {k: stats(v) for k, v in out.items()}
    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.method}_projection.npz"),
        **out)
    with open(os.path.join(args.out_dir,
                           f"{args.method}_manifest.json"), "w") as fh:
        json.dump(dict(manifest=man, stats=allstats), fh, indent=1)
    for k, s in allstats.items():
        print(f"[ext] {args.method} {k}: rms={s['rms']:.4f} "
              f"absmax={s['absmax']:.2f} kurt={s['kurtosis']:.1f} "
              f"top0.1%={s['top01pct_channel_energy']:.3f} "
              f"{s['shape']}", flush=True)
    print(f"[ext] {args.method} done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
