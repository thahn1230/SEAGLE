#!/usr/bin/env python
"""Study C alpha calibration (per target interface mode).

Calibration corpus: wikitext-2 TRAIN paragraphs (chat-templated), disjoint
from the MT-bench AL evaluation set. Captures projection inputs z=[e|h]
for first/recurrent paths with the FP16 adapter installed, then sweeps
alpha in {2^k, k=-8..8 step .25}, optimizing projection-OUTPUT NMSE with
BOTH W4 weight quant and shared A4 input quant in the loop.

Writes into --run-dir:
  embedding_alpha_sweep__<target>.csv
  activation_distribution_stats__<target>.csv
  alpha_selected.json   (merged: {target: {alpha, k, objective}})
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.embedding_scale_reparameterization import (
    capture_projection_inputs, sweep_alpha)

KIND = "learned_chat_w4a4kv16"


def calib_prompts(tok, build_prompt, n=8, min_chars=600):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) >= min_chars and not t.startswith("="):
            texts.append("Summarize the following passage:\n" + t[:1200])
        if len(texts) == n:
            break
    return [build_prompt(tok, t) for t in texts]


def dist_stats(Z, path, target):
    D = Z.shape[1] // 2
    rows = []
    for sl, name in ((Z[:, :D], "embedding"), (Z[:, D:], "hidden")):
        a = sl.abs()
        q = torch.tensor([.5, .9, .95, .99, .999])
        pq = torch.quantile(a.flatten().float(), q)
        rows.append(dict(target=target, path=path, slice=name,
                         vmin=float(sl.min()), vmax=float(sl.max()),
                         absmax=float(a.max()), mean=float(sl.mean()),
                         std=float(sl.std()),
                         p50=float(pq[0]), p90=float(pq[1]),
                         p95=float(pq[2]), p99=float(pq[3]),
                         p999=float(pq[4]), n_rows=int(Z.shape[0])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["fp16", "w4a4"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    rot, quant, fhm = (("none", "none", "identity") if args.target == "fp16"
                       else ("full", "w4a4", "gamma_R1"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    if rot == "none":
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight.detach() \
            .float().cpu().clone()
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [p.to(dev) for p in calib_prompts(tok, build_prompt)]
    print(f"[alpha] {args.target}: {len(ids_list)} calibration prompts "
          f"(wikitext-2 train, disjoint from MT-bench)", flush=True)

    ad = ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                     variant="folded",
                                     first_hidden_mode=fhm, trace=False)
    ad.install()
    Z = capture_projection_inputs(ad, model, ids_list, tree, max_rows=8192)
    W_f = ad.split.projection_first_preR.weight.detach().float().cpu()
    W_r = ad.split.projection_recurrent_preR.weight.detach().float().cpu()
    bias = ad.split.projection_first_preR.bias.detach().float().cpu()
    ad.uninstall()
    print(f"[alpha] captured rows: "
          f"{ {k: int(v.shape[0]) for k, v in Z.items()} }", flush=True)

    drows = []
    for k, v in Z.items():
        drows += dist_stats(v, k, args.target)
    logging_utils.write_csv(os.path.join(
        args.run_dir, f"activation_distribution_stats__{args.target}.csv"),
        drows)

    best_alpha, rows = sweep_alpha(W_f, W_r, bias,
                                   Z.get("first"), Z.get("recurrent"))
    logging_utils.write_csv(os.path.join(
        args.run_dir, f"embedding_alpha_sweep__{args.target}.csv"), rows)
    sel_path = os.path.join(args.run_dir, "alpha_selected.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    brow = min(rows, key=lambda r: r["objective_nmse_sum"])
    sel[args.target] = dict(alpha=best_alpha, k=brow["k"],
                            objective_nmse_sum=brow["objective_nmse_sum"])
    json.dump(sel, open(sel_path, "w"), indent=2)
    print(f"[alpha] {args.target}: best alpha = {best_alpha} (2^{brow['k']}),"
          f" objective {brow['objective_nmse_sum']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
