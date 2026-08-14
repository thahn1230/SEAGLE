#!/usr/bin/env python
"""Execution-shape audit for the AA-QAT teacher (study section 4).

Runs the DEPLOYED speculative pipeline (W4A4 tree-verified target + method
draft) and, in lockstep on a second GPU, the SAME W4A4 target executed the
way the LK-corpus teacher executes it (batched prompt prefill + strictly
one-token incremental decode, tree_mask=None). Along each cycle's ACCEPTED
path, records distribution-level divergence between the tree-executed
verifier distribution p_tree and the sequential teacher distribution p_seq
at identical positions:

  KL(p_tree||p_seq), KL(p_seq||p_tree), TV, argmax agreement,
  top-2 logit gap of p_tree at disagreeing positions,
  R_seq = acceptance length the sequential lane would have granted the
          same proposals (vs deployed R_q).

Since |min(a,q)-min(b,q)| <= |a-b| elementwise, sum_v TV bounds the alpha
error: |alpha(p_tree,q) - alpha(p_seq,q)| <= TV(p_tree,p_seq) for ANY
draft q — TV is the decisive design quantity, no draft distributions
needed. Decision rule (section 4): if median TV is an order of magnitude
below the per-depth acceptance gap being optimized (1-alpha ~ 0.3-0.5),
the sequential corpus teacher is deployment-matched to first order.

Writes tables/tree_vs_seq_teacher__<tag>__<ds>.json (+ per-cycle jsonl).
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-cfg", default="d4p3",
                    choices=["d4p3", "rot"])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--alpha-rec", type=float, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="mtbench")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--pool", default="eval")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ref-device", default="cuda:1")
    args = ap.parse_args()
    dev, rdev = "cuda:0", args.ref_device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")

    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    fhm = "gamma_R1"
    D4 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True)
    if args.draft_cfg == "d4p3":
        kw = dict(embed_scale_alpha=args.alpha, **D4)
        if args.alpha_rec is not None:
            kw["embed_scale_alpha_rec"] = args.alpha_rec
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, **kw)
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        st = dict(stash)
        R_T = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, first_fold_R=R_T,
            embed_scale_alpha=args.alpha, **D4)
    ad.install()

    # sequential lane: the SAME quantized target build, second GPU,
    # executed exactly like build_target_generated_lk_corpus's teacher
    ref_model, _st2, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=rdev, rotations_root=rr)
    ref = ref_model.base_model
    ref.model.tree_mask = None

    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    from eagle.model.utils import (initialize_tree, generate_candidates,
                                   tree_decoding, evaluate_posterior,
                                   update_inference_inputs,
                                   reset_tree_mode)
    from eagle.model.ea_model import generate_tree_buffers
    from eagle.model.kv_cache import initialize_past_key_values
    tb = generate_tree_buffers(tree, device=dev)
    tb["retrieve_indices_head"] = tb["retrieve_indices"].to(dev)
    model.tree_buffers = tb
    model.tree_choices = tree
    ri = tb["retrieve_indices"]
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "cycles"), exist_ok=True)

    def metrics(zt, zs):
        """zt, zs: (V,) fp32 logits -> dict of divergence metrics."""
        pt = F.softmax(zt, -1)
        ps = F.softmax(zs, -1)
        lpt = F.log_softmax(zt, -1)
        lps = F.log_softmax(zs, -1)
        tv = float(0.5 * (pt - ps).abs().sum())
        kl_ts = float((pt * (lpt - lps)).sum())
        kl_st = float((ps * (lps - lpt)).sum())
        agree = int(int(zt.argmax()) == int(zs.argmax()))
        top2 = torch.topk(zt, 2).values
        return tv, kl_ts, kl_st, agree, float(top2[0] - top2[1])

    for ds_name in args.datasets.split(","):
        rows_path = os.path.join(
            args.run_dir, "cycles",
            f"tvs__{args.tag}__{ds_name}.jsonl")
        prompts, _ = load_eval_prompts(ds_name, args.n_prompts, args.pool)
        f = open(rows_path, "w")
        agg = dict(n_pos=0, n_cyc=0, tv=[], kl_ts=[], kl_st=[],
                   agree=0, disagree_gap=[], by_depth={},
                   r_q_sum=0, r_seq_sum=0)
        for p in prompts:
            input_ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            input_len = input_ids.shape[1]
            model.ea_layer.reset_kv()
            reset_tree_mode(model)
            if hasattr(model, "past_key_values"):
                past_key_values = model.past_key_values
                past_key_values_data = model.past_key_values_data
                current_length_data = model.current_length_data
                current_length_data.zero_()
            else:
                (past_key_values, past_key_values_data,
                 current_length_data) = initialize_past_key_values(
                    model.base_model)
                model.past_key_values = past_key_values
                model.past_key_values_data = past_key_values_data
                model.current_length_data = current_length_data
            ref.model.tree_mask = None
            ref_past, ref_past_data, ref_cur = \
                initialize_past_key_values(ref)
            # corpus-style prefill: batched prompt forward, then
            # one-token incremental — identical to the corpus builder
            h = ref.model(input_ids=input_ids.to(rdev),
                          past_key_values=ref_past, use_cache=True)[0]
            del h

            def seq_feed(t):
                step = torch.tensor([[int(t)]], device=rdev)
                hh = ref.model(input_ids=step, past_key_values=ref_past,
                               use_cache=True)[0]
                return ref.lm_head(hh)[0, -1].float()

            tree_logits, logits, hidden_state, sample_token = \
                initialize_tree(input_ids, model, tb["tree_attn_mask"],
                                past_key_values, None)
            last_seq = seq_feed(int(sample_token))    # after root token
            ids = input_ids
            n_cyc_prompt = 0
            for _ in range(args.max_new_tokens + 8):
                (candidates, cart_p, tree_candidates) = \
                    generate_candidates(tree_logits, tb["tree_indices"],
                                        tb["retrieve_indices"],
                                        sample_token, None)
                prefix_len = ids.shape[1]
                logits, hs_new, _o = tree_decoding(
                    model, tree_candidates, past_key_values,
                    tb["tree_position_ids"], ids,
                    tb["retrieve_indices_head"])
                best_q, len_q, sample_p = evaluate_posterior(
                    logits, candidates, None, cart_p, tree_logits[2],
                    tb["p_indices"], tree_candidates, tb["b_indices"])
                len_q = int(len_q)
                # accepted-path comparison, positions j=0..len_q
                seq_lgs = [last_seq]
                acc_toks = candidates[best_q, 1:1 + len_q].tolist()
                for t in acc_toks:
                    seq_lgs.append(seq_feed(t))
                rec = dict(prompt_id=p["row_id"], cycle=agg["n_cyc"],
                           prefix_len=prefix_len, R_q=len_q, pos=[])
                # sequential re-verification of the same proposal path
                r_seq = 0
                for j, t in enumerate(acc_toks):
                    if int(seq_lgs[j].argmax()) == int(t):
                        r_seq += 1
                    else:
                        break
                for j in range(len_q + 1):
                    zt = logits[best_q, j].float().to(rdev)
                    tv, k1, k2, agr, gap = metrics(zt, seq_lgs[j])
                    agg["tv"].append(tv)
                    agg["kl_ts"].append(k1)
                    agg["kl_st"].append(k2)
                    agg["agree"] += agr
                    if not agr:
                        agg["disagree_gap"].append(gap)
                    d = agg["by_depth"].setdefault(
                        j, dict(n=0, tv=0.0, kl=0.0, agree=0))
                    d["n"] += 1
                    d["tv"] += tv
                    d["kl"] += k1
                    d["agree"] += agr
                    agg["n_pos"] += 1
                    rec["pos"].append(dict(j=j, tv=round(tv, 5),
                                           kl=round(k1, 5), agree=agr))
                rec["R_seq"] = r_seq
                agg["r_q_sum"] += len_q
                agg["r_seq_sum"] += r_seq
                f.write(json.dumps(rec) + "\n")
                agg["n_cyc"] += 1
                n_cyc_prompt += 1
                # advance the seq lane over the correction token
                corr = int(torch.argmax(logits[best_q, len_q]).item())
                last_seq = seq_feed(corr)
                ids, tree_logits, new_token, hidden_state, \
                    sample_token = update_inference_inputs(
                        ids, candidates, best_q,
                        torch.tensor(len_q, device=dev), ri, None,
                        logits, tree_logits, 0,
                        past_key_values_data, current_length_data,
                        model, hidden_state, hs_new, sample_p)
                if tok.eos_token_id in ids[0, input_len:].tolist():
                    break
                if ids.shape[1] - input_len >= args.max_new_tokens:
                    break
            del ref_past, ref_past_data, ref_cur
            torch.cuda.empty_cache()
        f.close()
        tvs = torch.tensor(agg["tv"])
        kls = torch.tensor(agg["kl_ts"])
        summary = dict(
            tag=args.tag, dataset=ds_name, n_prompts=len(prompts),
            n_cycles=agg["n_cyc"], n_positions=agg["n_pos"],
            tv_mean=float(tvs.mean()), tv_median=float(tvs.median()),
            tv_p90=float(tvs.quantile(0.9)),
            tv_max=float(tvs.max()),
            kl_tree_seq_mean=float(kls.mean()),
            kl_tree_seq_median=float(kls.median()),
            kl_seq_tree_mean=float(torch.tensor(agg["kl_st"]).mean()),
            argmax_agreement=agg["agree"] / max(agg["n_pos"], 1),
            disagree_top2gap_median=(
                float(torch.tensor(agg["disagree_gap"]).median())
                if agg["disagree_gap"] else None),
            r_q_mean=agg["r_q_sum"] / max(agg["n_cyc"], 1),
            r_seq_mean=agg["r_seq_sum"] / max(agg["n_cyc"], 1),
            by_depth={j: dict(n=d["n"], tv=d["tv"] / d["n"],
                              kl=d["kl"] / d["n"],
                              agree=d["agree"] / d["n"])
                      for j, d in sorted(agg["by_depth"].items())},
            alpha_bound_note=("|alpha(p_tree,q)-alpha(p_seq,q)| <= "
                              "TV(p_tree,p_seq) for any q"))
        out = os.path.join(args.run_dir, "tables",
                           f"tree_vs_seq_teacher__{args.tag}__"
                           f"{ds_name}.json")
        json.dump(summary, open(out, "w"), indent=1)
        print(f"[tvs] {ds_name}: TV median {summary['tv_median']:.4f} "
              f"p90 {summary['tv_p90']:.4f} agree "
              f"{summary['argmax_agreement']:.4f} R_q "
              f"{summary['r_q_mean']:.3f} R_seq "
              f"{summary['r_seq_mean']:.3f} -> {out}", flush=True)
    ad.uninstall()
    return 0


if __name__ == "__main__":
    sys.exit(main())
