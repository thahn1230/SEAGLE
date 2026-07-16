#!/usr/bin/env python
"""Fixed-prefix / fixed-tree target-grader experiment (§10).

Stage 1 (STOCK fp16 target + fp16 draft): drive the EAGLE loop manually
(initialize_tree → generate_candidates → tree_decoding → evaluate_posterior →
update_inference_inputs) and CACHE, per verification round:
    prefix token ids, candidates, tree_candidates, retrieve_indices,
    tree_position_ids, root-children token set, fp16 target root logits,
    fp16 accepted depth.

Stage 2 (per target ∈ {fp16(recheck), W8A8, W4A4}): re-verify the IDENTICAL
cached trees: prefill the prefix (init=False), set the tree mask, tree_decode
the cached candidates, evaluate_posterior (greedy) → accepted depth. Also
record the root next-token distribution (entropy, top1, margin, KL/TV vs fp16,
tree mass, top1-in-tree) and classify each round per
configs/target_grader_thresholds.yaml. WikiText-2 token-level CE computed for
each target (no hard-coded PPL).

Writes artifacts/bitwidth_al_component_causality/fixed_tree/
  {rounds.csv, categories_summary.json, target_quality.csv}

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_fixed_tree_target_grader.py --device cuda:0
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality",
                   "fixed_tree")


@torch.no_grad()
def wikitext_ce(model, tok, dev, n_tokens=32768):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][:n_tokens].to(dev)
    model.base_model.model.tree_mask = None
    losses, denom = 0.0, 0
    for i in range(0, ids.shape[0] - 1, 2048):
        chunk = ids[i:i + 2049][None]
        if chunk.shape[1] < 2:
            break
        lg = model.base_model(chunk).logits[0, :-1].float()
        tgt = chunk[0, 1:]
        losses += F.cross_entropy(lg, tgt, reduction="sum").item()
        denom += tgt.numel()
    return losses / denom, denom


@torch.no_grad()
def stage1_cache(model, ids_list, tree, dev, max_rounds=16, max_new=64):
    """Run the EAGLE loop on the stock model, caching per-round tree data."""
    from eagle.model.utils import (initialize_tree, generate_candidates,
                                   tree_decoding, evaluate_posterior,
                                   update_inference_inputs, reset_tree_mode)
    from eagle.model.kv_cache import initialize_past_key_values
    cached = []
    for pi, ids in enumerate(ids_list):
        model.ea_layer.reset_kv()
        reset_tree_mode(model)
        past, pkv_data, cur_len = initialize_past_key_values(model.base_model)
        input_ids = ids.clone()
        tb = model.tree_buffers
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, model, tb["tree_attn_mask"], past, None)
        new_token = 0
        for rnd in range(max_rounds):
            candidates, cart_candidates_prob, tree_candidates = \
                generate_candidates(tree_logits, tb["tree_indices"],
                                    tb["retrieve_indices"], sample_token, None)
            prefix_ids = input_ids[0].tolist()
            lgts, hidden_state_new, outputs = tree_decoding(
                model, tree_candidates, past, tb["tree_position_ids"],
                input_ids, tb["retrieve_indices"])
            # v1 signature: 8 required positionals; the trailing five are
            # unused on the greedy (logits_processor=None) branch
            best_candidate, accept_length, sample_p = evaluate_posterior(
                lgts, candidates, None, cart_candidates_prob, None, None,
                tree_candidates, None)
            root_logits = lgts[0, 0].float().cpu()   # dist at the root position
            root_children = sorted(set(
                candidates[:, 1].tolist()) - {-1})
            cached.append(dict(
                prompt_idx=pi, round=rnd, prefix=prefix_ids,
                candidates=candidates.cpu(),
                tree_candidates=tree_candidates.cpu(),
                fp_accept_length=int(accept_length),
                fp_root_logits=root_logits,
                root_children=root_children))
            # v1 return order (utils.py:472 / ea_model.py:229):
            # (input_ids, tree_logits, new_token, hidden_state, sample_token);
            # 10th arg is the raw KV tensor list, last arg is sample_p
            input_ids, tree_logits, new_token, hidden_state, sample_token = \
                update_inference_inputs(
                    input_ids, candidates, best_candidate, accept_length,
                    tb["retrieve_indices"], None, logits,
                    tree_logits, new_token, pkv_data, cur_len,
                    model, hidden_state, hidden_state_new, sample_p)
            if new_token > max_new or input_ids.shape[1] > ids.shape[1] + max_new:
                break
    return cached


@torch.no_grad()
def stage2_verify(model, cached, tree_buffers, dev):
    """Re-verify cached trees under THIS target. Returns per-round dicts."""
    from eagle.model.utils import tree_decoding, evaluate_posterior
    from eagle.model.kv_cache import initialize_past_key_values
    out = []
    for rec in cached:
        model.base_model.model.tree_mask = None
        past, _pd, _cl = initialize_past_key_values(model.base_model)
        prefix = torch.tensor([rec["prefix"]], device=dev)
        _ = model.base_model(prefix, past_key_values=past, use_cache=True)
        model.base_model.model.tree_mask = tree_buffers["tree_attn_mask"]
        lgts, _hs, _o = tree_decoding(
            model, rec["tree_candidates"].to(dev), past,
            tree_buffers["tree_position_ids"], prefix,
            tree_buffers["retrieve_indices"])
        best, acc = evaluate_posterior(
            lgts, rec["candidates"].to(dev), None, None, None, None,
            rec["tree_candidates"].to(dev), None)[:2]
        out.append(dict(accept_length=int(acc),
                        root_logits=lgts[0, 0].float().cpu()))
        del past, _pd, _cl
        torch.cuda.empty_cache()
    return out


def dist_metrics(z):
    p = F.softmax(z, -1)
    lp = F.log_softmax(z, -1)
    top2 = z.topk(2)
    return dict(entropy=float(-(p * lp).sum()),
                top1=int(top2.indices[0]),
                top1_prob=float(p[top2.indices[0]]),
                margin=float(top2.values[0] - top2.values[1]))


def classify(row, thr):
    d = row["accept_delta"]
    if abs(d) < thr["depth_change_min"]:
        return "UNCHANGED"
    # fp16 recheck (same weights, stage-2 path) already disagrees with
    # stage 1 on this round -> the flip is attributable to the execution
    # path, not to quantization (consumes path_artifact_top1_tol).
    if abs(row["recheck_accept_delta"]) >= thr["path_artifact_top1_tol"] or \
            not row["recheck_top1_same_fp"]:
        return "EXECUTION_PATH_ARTIFACT"
    if d > 0:
        if row["fp_q_top1_same"] and row["kl_fp_q"] <= thr["kl_small"] and \
                row["ce_delta"] <= thr["quality_tolerance_ce"]:
            return "BENIGN_GENEROSITY"
        if not row["fp_q_top1_same"] and row["q_top1_in_tree"]:
            return "ACCIDENTAL_RANK_FLIP"
        if row["entropy_delta"] > thr["entropy_increase"] and \
                row["margin_delta"] < -thr["margin_decrease"]:
            return "FLATTENING_GENEROSITY"
        return "DEGRADATION_INDUCED_GENEROSITY"
    else:
        if not row["q_top1_in_tree"] or row["tree_mass_delta"] < \
                -thr["tree_mass_increase"]:
            return "STRICTER_ALIGNMENT_LOSS"
        return "DEGRADED_FALSE_REJECTION"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--rotation-kind", default="random_hadamard")
    ap.add_argument("--out-suffix", default="",
                    help="artifact subdir suffix, e.g. '_learned'")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        ("6,7", "0", "1", "2", "3", "4", "5", "6")
    assert torch.cuda.device_count() in (1, 2)
    dev = "cuda:0"
    global ART
    ART = ART + args.out_suffix
    os.makedirs(ART, exist_ok=True)
    thr = yaml.safe_load(open(os.path.join(
        PROJECT_ROOT, "configs", "target_grader_thresholds.yaml")))

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    # ---------- stage 1: stock target + fp16 draft, cache trees ------------
    print("[grader] stage 1: stock target, caching trees ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "none", args.rotation_kind, "none", 0, device=dev,
        rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    # EaModel keeps tree buffers on the model for the utils path.
    # MUST be the verifier-side builder (eagle.model.utils) — the cnets
    # star-import resolves to utils_c's DRAFT builder, which lacks
    # retrieve_indices/tree_attn_mask/tree_position_ids.
    from eagle.model.utils import generate_tree_buffers
    model.tree_buffers = generate_tree_buffers(tree, dev)
    model.tree_buffers["retrieve_indices"] = \
        model.tree_buffers["retrieve_indices"].to(dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
    cached = stage1_cache(model, ids_list, tree, dev)
    print(f"[grader] cached {len(cached)} rounds", flush=True)
    # fp16(recheck) control: SAME weights through the stage-2 execution path
    # (fresh prefill instead of incrementally compacted KV). Rounds where this
    # alone flips accept/top-1 are execution-path artifacts, not quantization.
    recheck = stage2_verify(model, cached, model.tree_buffers, dev)
    n_flip = sum(r["accept_length"] != rec["fp_accept_length"]
                 for rec, r in zip(cached, recheck))
    print(f"[grader] fp16 recheck: {n_flip}/{len(cached)} rounds flip accept "
          f"under the stage-2 path", flush=True)
    ce_fp, denom = wikitext_ce(model, tok, dev)
    print(f"[grader] fp16 wikitext CE={ce_fp:.4f} ({denom} tok)", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---------- stage 2: each target grades the SAME trees -----------------
    rows = []
    quality = [dict(target="fp16", wikitext_ce=round(ce_fp, 5),
                    wikitext_ppl=round(float(np.exp(ce_fp)), 4),
                    n_tokens=denom)]
    for tprec, quant in (("W8A8", "w8a8"), ("W4A4", "w4a4")):
        print(f"[grader] stage 2: target {tprec} ...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "full", args.rotation_kind, quant, 0, device=dev,
            rotations_root=rr)
        study.set_draft_tree(model, tree, dev)
        model.tree_buffers = generate_tree_buffers(tree, dev)
        model.tree_buffers["retrieve_indices"] = \
            model.tree_buffers["retrieve_indices"].to(dev)
        res = stage2_verify(model, cached, model.tree_buffers, dev)
        ce_q, dn = wikitext_ce(model, tok, dev)
        quality.append(dict(target=tprec, wikitext_ce=round(ce_q, 5),
                            wikitext_ppl=round(float(np.exp(ce_q)), 4),
                            n_tokens=dn))
        for rec, r, rc in zip(cached, res, recheck):
            zf, zq = rec["fp_root_logits"], r["root_logits"]
            mf, mq = dist_metrics(zf), dist_metrics(zq)
            mrc = dist_metrics(rc["root_logits"])
            pf = F.softmax(zf, -1); pq = F.softmax(zq, -1)
            kl = float(F.kl_div(F.log_softmax(zq, -1), pf,
                                reduction="sum"))
            tv = float(0.5 * (pf - pq).abs().sum())
            ch = rec["root_children"]
            mass_f = float(pf[ch].sum()); mass_q = float(pq[ch].sum())
            row = dict(
                target=tprec, prompt_idx=rec["prompt_idx"],
                round=rec["round"],
                fp_accept=rec["fp_accept_length"],
                q_accept=r["accept_length"],
                accept_delta=r["accept_length"] - rec["fp_accept_length"],
                recheck_accept=rc["accept_length"],
                recheck_accept_delta=rc["accept_length"]
                - rec["fp_accept_length"],
                recheck_top1_same_fp=bool(mrc["top1"] == mf["top1"]),
                accept_delta_vs_recheck=r["accept_length"]
                - rc["accept_length"],
                fp_q_top1_same=bool(mf["top1"] == mq["top1"]),
                q_top1_in_tree=bool(mq["top1"] in ch),
                fp_top1_in_tree=bool(mf["top1"] in ch),
                entropy_delta=round(mq["entropy"] - mf["entropy"], 4),
                margin_delta=round(mq["margin"] - mf["margin"], 4),
                top1_prob_delta=round(mq["top1_prob"] - mf["top1_prob"], 4),
                kl_fp_q=round(kl, 4), tv=round(tv, 4),
                tree_mass_fp=round(mass_f, 4), tree_mass_q=round(mass_q, 4),
                tree_mass_delta=round(mass_q - mass_f, 4),
                ce_delta=round(ce_q - ce_fp, 4))
            row["category"] = classify(row, thr)
            rows.append(row)
        del model
        torch.cuda.empty_cache()

    logging_utils.write_csv(os.path.join(ART, "rounds.csv"), rows)
    logging_utils.write_csv(os.path.join(ART, "target_quality.csv"), quality)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = {}
    for tprec, g in df.groupby("target"):
        inc = g[g.accept_delta >= 1]
        dec = g[g.accept_delta <= -1]
        summ[tprec] = dict(
            n_rounds=len(g),
            frac_increase=round(len(inc) / len(g), 4),
            frac_decrease=round(len(dec) / len(g), 4),
            frac_unchanged=round((g.category == "UNCHANGED").mean(), 4),
            mean_accept_delta=round(float(g.accept_delta.mean()), 4),
            categories={str(k): int(v)
                        for k, v in g.category.value_counts().items()},
            increase_categories={str(k): int(v)
                                 for k, v in inc.category.value_counts().items()},
            decrease_categories={str(k): int(v)
                                 for k, v in dec.category.value_counts().items()})
    recheck_summ = dict(
        n_rounds=len(cached),
        n_accept_flips=int(sum(
            r["accept_length"] != rec["fp_accept_length"]
            for rec, r in zip(cached, recheck))),
        flip_fraction=round(sum(
            r["accept_length"] != rec["fp_accept_length"]
            for rec, r in zip(cached, recheck)) / max(len(cached), 1), 4))
    with open(os.path.join(ART, "categories_summary.json"), "w") as f:
        json.dump(dict(thresholds=thr, quality=quality,
                       fp16_recheck=recheck_summ, per_target=summ),
                  f, indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str), flush=True)
    print("[grader] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
