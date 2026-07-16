#!/usr/bin/env python
"""Gate B: stock-FP16 target vs learned-rotation-FP16 target equivalence.

Sequentially builds (A) the stock target and (B) the learned-rotation
target with quantization disabled, and compares:
  1. target logits on fixed prefixes (rel-L2, max|Δ|, top-1 agreement)
  2. target-only greedy 64-token continuations (exact sequence match)
  3. EAGLE acceptance (n=20×64) with the matching draft interface
Writes gate_b_summary.json into the run dir.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.study import UnrotateAdapter

KIND = "learned_chat_w4a4kv16"


def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        final = out
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist(), deltas


@torch.no_grad()
def probe_target(model, ids_list, dev, n_greedy=64):
    bm = model.base_model
    bm.model.tree_mask = None
    logits, greedy = [], []
    for ids in ids_list:
        logits.append(bm(ids.to(dev)).logits[0, -32:, :].float().cpu())
        seq = ids.clone().to(dev)
        for _ in range(n_greedy):
            nxt = bm(seq).logits[0, -1].argmax()
            seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
        greedy.append(seq[0, ids.shape[1]:].cpu().tolist())
    return logits, greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=20)
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
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    results = {}
    for tag, rot in (("stock", "none"), ("learned_rot_fp16", "full")):
        print(f"[gateB] building {tag} ...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rot, KIND, "none", 0, device=dev, rotations_root=rr)
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
        lg, gr = probe_target(model, ids_list[:8], dev)
        # EAGLE AL with matching interface
        if rot == "none":
            ad = None
        else:
            ad = UnrotateAdapter(model, stash, dev, torch.float16,
                                 with_gamma=True)
            ad.install()
        als = []
        for pi, ids in enumerate(ids_list):
            if ad is not None and hasattr(ad, "set_context"):
                ad.set_context(prompts[pi]["question_id"])
            _, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=72, tree_choices=tree),
                ids.shape[1], 64)
            als.append(float(np.mean(deltas)) if deltas else 0.0)
        if ad is not None:
            ad.uninstall()
        results[tag] = dict(logits=lg, greedy=gr, als=als)
        del model
        torch.cuda.empty_cache()

    a, b = results["stock"], results["learned_rot_fp16"]
    rel = [float((x - y).norm() / x.norm()) for x, y in
           zip(a["logits"], b["logits"])]
    mabs = [float((x - y).abs().max()) for x, y in
            zip(a["logits"], b["logits"])]
    top1 = [float((x.argmax(-1) == y.argmax(-1)).float().mean())
            for x, y in zip(a["logits"], b["logits"])]
    greedy_same = [ga == gb for ga, gb in zip(a["greedy"], b["greedy"])]
    d = np.array(b["als"]) - np.array(a["als"])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    summ = dict(
        n_logit_probes=len(rel),
        logit_rel_l2=dict(mean=float(np.mean(rel)), max=float(np.max(rel))),
        logit_max_abs=float(np.max(mabs)),
        prefill_top1_agreement=float(np.mean(top1)),
        greedy_identical=int(sum(greedy_same)), greedy_total=len(greedy_same),
        al_stock=float(np.mean(a["als"])),
        al_learned_rot_fp16=float(np.mean(b["als"])),
        al_paired_delta=float(d.mean()),
        al_delta_ci95=[float(np.quantile(boots, .025)),
                       float(np.quantile(boots, .975))],
        verdict=("PASS" if sum(greedy_same) >= len(greedy_same) - 1
                 and np.quantile(boots, .025) < 0 < np.quantile(boots, .975)
                 or abs(d.mean()) < 0.05 else "REVIEW"))
    with open(os.path.join(args.run_dir, "gate_b_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("[gateB]", json.dumps(summ, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
