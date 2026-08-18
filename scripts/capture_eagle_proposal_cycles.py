#!/usr/bin/env python
"""Deployed-proposal capture + FP16-reference lockstep replay (study
§14-§15). Greedy only (temperature 0).

Per speculative cycle of the DEPLOYED config (Tq target + method draft):
capture the pre-cycle prefix length, the full candidates tensor, the
Tq-accepted (best_candidate, accept_length); verify the SAME proposal
tree from the SAME prefix with the FP16 reference target T0 held in
lockstep on a second GPU (its KV always compressed along the DEPLOYED
trajectory, so both verifiers see identical prefixes). Records
proposal-only accepted sequences S_q, S_0, their longest common prefix
R_RC, branch-divergence info, and correction tokens.

Writes cycles/<tag>__<ds>.jsonl (one record per cycle) + a summary json.
The replay never regenerates proposals and never lets trajectories mix.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["fp16", "int4", "w8a8"])
    ap.add_argument("--draft-cfg", required=True,
                    choices=["stock", "naive_w4a4", "d4p3", "d4p3_deploy",
                             "rot", "fp16_deploy", "rot_ep3p",
                             "native_raw"])
    ap.add_argument("--nr-quant", default="fp16",
                    help="native_raw: QUANT_BITS mode for all sites")
    ap.add_argument("--nr-rot", default="none",
                    help="native_raw: 'had' or R_D ckpt path")
    ap.add_argument("--nr-alpha", type=float, default=None,
                    help="native_raw: exact embed/W_e balancing alpha")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--proj-rot-first", default=None,
                    help="R-EP3-P rotation spec JSON (first path)")
    ap.add_argument("--proj-rot-rec", default=None,
                    help="R-EP3-P rotation spec JSON (recurrent path)")
    ap.add_argument("--alpha-rec", type=float, default=None,
                    help="pathwise EP3-P recurrent factor")
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default="mtbench")
    ap.add_argument("--n-prompts", type=int, default=80)
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

    # deployed system (Tq + method draft) on dev
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w8a8") if args.target == "w8a8"
                  else ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    if stash.get("R1") is None:
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                        rr), map_location="cpu",
                       weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight \
            .detach().float().cpu().clone()
    ea = model.ea_layer
    if args.draft_sd:
        sd = torch.load(args.draft_sd, map_location="cpu",
                        weights_only=False)
        sd = sd.get("draft_state_dict", sd.get("model", sd))
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                            for k, v in sd.items()}, strict=True)
        ea.to(dev)
    fhm = "identity" if args.target == "fp16" else "gamma_R1"
    D4 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True)
    ad = None
    if args.draft_cfg == "native_raw":
        # strict SEAGLE-RT arms: no adapter/restore; optional raw
        # quant / interface rotation / alpha (mirrors the AL evaluator)
        assert args.target != "fp16"
        if args.nr_alpha is not None:
            from eagle_spinquant.native_raw_draft import (
                apply_embed_alpha)
            apply_embed_alpha(model, args.nr_alpha)
        if args.nr_rot != "none":
            from eagle_spinquant.native_raw_draft import (
                apply_interface_rotation)
            R = None
            if args.nr_rot != "had":
                R = torch.load(args.nr_rot, map_location="cpu",
                               weights_only=False)["R_D"].double()
            apply_interface_rotation(model, args.nr_quant, R=R)
        elif args.nr_quant != "fp16":
            from eagle_spinquant.native_raw_draft import (
                apply_native_raw_quant)
            apply_native_raw_quant(model, args.nr_quant)
    elif args.draft_cfg in ("d4p3", "d4p3_deploy"):
        kw = dict(embed_scale_alpha=args.alpha, **D4)
        if args.alpha_rec is not None:
            kw["embed_scale_alpha_rec"] = args.alpha_rec
        if args.proj_rot_first:
            kw["proj_rot_first"] = json.loads(args.proj_rot_first)
        if args.proj_rot_rec:
            kw["proj_rot_rec"] = json.loads(args.proj_rot_rec)
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, **kw)
    elif args.draft_cfg == "naive_w4a4":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, **D4)
    elif args.draft_cfg == "rot":
        ck = torch.load(args.ckpt, map_location="cpu",
                        weights_only=False)
        st = dict(stash)
        R_T = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
        def_alpha = 45.254834 if args.target == "fp16" else 32.0
        r2o = ck.get("R2_D")   # GS/R2 study: learned draft-aware R2_D
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, first_fold_R=R_T,
            embed_scale_alpha=(args.alpha if args.alpha is not None
                               else float(ck.get("alpha", def_alpha))),
            ar_r2_override=(r2o.double() if r2o is not None else None),
            **D4)
    elif args.draft_cfg == "rot_ep3p":
        # R_D gauge + EP3-P pathwise scales (T->D bridge pinned at R_T)
        ck = torch.load(args.ckpt, map_location="cpu",
                        weights_only=False)
        st = dict(stash)
        R_T = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
        a_rec = (args.alpha_rec if args.alpha_rec is not None
                 else ck.get("alpha_rec"))
        kw = dict(embed_scale_alpha=(args.alpha
                                     if args.alpha is not None
                                     else float(ck.get("alpha", 32.0))),
                  **D4)
        if a_rec is not None:
            kw["embed_scale_alpha_rec"] = float(a_rec)
        r2o = ck.get("R2_D")   # GS/R2 study: learned draft-aware R2_D
        if r2o is not None:
            kw["ar_r2_override"] = r2o.double()
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode=fhm, trace=False, first_fold_R=R_T, **kw)
    elif args.draft_cfg in ("stock", "fp16_deploy") \
            and args.target != "fp16":
        from eagle_spinquant.causal_interface import (
            RestoredInterfaceCSAdapter)
        ad = RestoredInterfaceCSAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode="identity", trace=False)
    if ad is not None:
        ad.install()

    # FP16 reference target T0 on the second GPU
    # vendored KV llama: supports tree_mask + managed KV cache
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    ref = KVLlama.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(rdev).eval()

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

    os.makedirs(os.path.join(args.run_dir, "cycles"), exist_ok=True)

    def ref_tree_logits(ref_past, prefix_len, tree_cands):
        ref.model.tree_mask = tb["tree_attn_mask"].to(rdev)
        pos = (tb["tree_position_ids"].to(rdev) + prefix_len)[None]
        out = ref(tree_cands.to(rdev), past_key_values=ref_past,
                  position_ids=pos, use_cache=True)
        lg = out.logits[0, ri.to(rdev)]
        return lg

    for ds_name in args.datasets.split(","):
        out_path = os.path.join(args.run_dir, "cycles",
                                f"cyc__{args.tag}__{ds_name}.jsonl")
        if os.path.exists(out_path):
            print(f"[cap] {args.tag}/{ds_name} exists, skip", flush=True)
            continue
        prompts, _ = load_eval_prompts(ds_name, args.n_prompts,
                                       args.pool)
        f = open(out_path, "w")
        n_cyc = 0
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
            # fresh reference KV per prompt (managed cache)
            ref.model.tree_mask = None
            ref_past, ref_past_data, ref_cur_len = \
                initialize_past_key_values(ref)
            # prefill reference on the SAME prefix
            ref(input_ids.to(rdev), past_key_values=ref_past,
                use_cache=True)

            tree_logits, logits, hidden_state, sample_token = \
                initialize_tree(input_ids, model, tb["tree_attn_mask"],
                                past_key_values, None)
            new_token = 0
            ids = input_ids
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
                # reference verification of the SAME tree/prefix
                lg0 = ref_tree_logits(ref_past, prefix_len,
                                      tree_candidates)
                pm0 = (candidates[:, 1:].to(rdev)
                       == torch.argmax(lg0[:, :-1], dim=-1)).int()
                cal0 = torch.cumprod(pm0, dim=1).sum(dim=1)
                len_0 = int(cal0.max())
                best_0 = (int(torch.argmax(cal0)) if len_0 > 0 else 0)
                S_q = candidates[best_q, 1:1 + len_q].tolist()
                S_0 = candidates[best_0, 1:1 + len_0].tolist()
                r_rc = lcp(S_q, S_0)
                corr_q = int(torch.argmax(
                    logits[best_q, len_q]).item())
                corr_0 = int(torch.argmax(lg0[best_0, len_0]).item())
                same_branch = (r_rc == min(len_q, len_0))
                # divergence depth: first index where accepted paths
                # differ (1-based depth; None if same-branch)
                div_depth = (None if same_branch else r_rc + 1)
                f.write(json.dumps(dict(
                    prompt_id=p["row_id"], cycle=n_cyc,
                    prefix_len=prefix_len, R_q=len_q, R_0=len_0,
                    R_RC=r_rc, same_branch=bool(same_branch),
                    div_depth=div_depth, S_q=S_q, S_0=S_0,
                    best_q=int(best_q), best_0=best_0,
                    corr_q=corr_q, corr_0=corr_0,
                    tree_tokens=tree_candidates[0].tolist())) + "\n")
                n_cyc += 1
                # advance BOTH KVs along the DEPLOYED trajectory
                sel0 = (ri[best_q, :len_q + 1].to(rdev) + prefix_len)
                for rp in ref_past_data:
                    tgt = rp[..., sel0.to(rp.device), :]
                    dst = rp[..., prefix_len:prefix_len + tgt.shape[-2],
                             :]
                    dst.copy_(tgt, non_blocking=True)
                ref_cur_len.fill_(prefix_len + len_q + 1)
                ids, tree_logits, new_token, hidden_state, \
                    sample_token = update_inference_inputs(
                        ids, candidates, best_q,
                        torch.tensor(len_q, device=dev), ri, None,
                        logits, tree_logits, new_token,
                        past_key_values_data, current_length_data,
                        model, hidden_state, hs_new, sample_p)
                if tok.eos_token_id in ids[0, input_len:].tolist():
                    break
                if ids.shape[1] - input_len >= args.max_new_tokens:
                    break
            del ref_past, ref_past_data, ref_cur_len
            torch.cuda.empty_cache()
        f.close()
        print(f"[cap] {args.tag}/{ds_name}: {n_cyc} cycles -> "
              f"{out_path}", flush=True)
    if ad is not None:
        ad.uninstall()
    return 0


if __name__ == "__main__":
    sys.exit(main())
