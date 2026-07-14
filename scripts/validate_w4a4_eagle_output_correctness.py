#!/usr/bin/env python
"""Implementation-correctness pass for mode
w4a4_unfused_target_fused_draft_head.

Tests (spec S8):
  A target tail correctness    (fused vs unfused logits on same x_R)
  B projection branch trace     ([R1.T,I] first, [R1.T,R1.T] recurrent)
  C draft lm_head trace         (fused head, R1.T+gamma_f absorbed)
  D AR target-only == target+EAGLE greedy tokens   (THE key test)
  E verifier logits == AR logits at every step

Phased by --quant: none(fp16) | w4a4(fake). Real-W4A4 is a separate backend
(documented if not run). Greedy only (temperature 0). Does not optimize
anything; reports mismatches loudly.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/validate_w4a4_eagle_output_correctness.py \
      --run-dir runs/w4a4_impl_fix_<ts> --quant none --num-prompts 2 --max-new-tokens 16
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study, w4a4_impl_fix as wf)

DEV = "cuda:0"


def logit_metrics(a, b):
    a = a.float(); b = b.float()
    d = a - b
    k = min(5, a.shape[-1])
    a5 = a.topk(k, -1).indices; b5 = b.topk(k, -1).indices
    ov = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / k
                       for x, y in zip(a5.reshape(-1, k), b5.reshape(-1, k))]).mean().item()
    pa = F.log_softmax(a, -1); pb = F.log_softmax(b, -1)
    return {"max_abs_logit_error": d.abs().max().item(),
            "mean_abs_logit_error": d.abs().mean().item(),
            "rel_l2_logit_error": (d.norm() / (b.norm() + 1e-9)).item(),
            "top1_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "top5_overlap": ov,
            "kl_divergence": F.kl_div(pb, pa, log_target=True, reduction="batchmean").item()}


@torch.no_grad()
def run_gen(gen, input_len, max_new):
    final, deltas, prev = None, [], input_len
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev)
            prev = cur
        final = out
        if cur - input_len >= max_new:
            break
    toks = final[0, input_len:input_len + max_new].tolist()
    return toks, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chat-template", default=None, choices=["llama2", "vicuna"])
    ap.add_argument("--quant", default="none", choices=["none", "w4a4", "w4a4kv4"])
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--phase", default="")
    ap.add_argument("--ablation", default="main",
                    choices=["main", "ident_control", "abl1_first_II",
                             "abl2_first_R1TR1T", "abl3_unfused_head",
                             "abl4_expose_h_hat"])
    args = ap.parse_args()

    # ablation -> (draft-adapter kwargs, expose_h_hat)
    ABL = {
        "main":            dict(first_embed="R1T", first_hidden="I",
                                rec_embed="R1T", rec_hidden="R1T", head_mode="fused"),
        "ident_control":   dict(first_embed="I", first_hidden="I",
                                rec_embed="I", rec_hidden="I", head_mode="unfused"),
        "abl1_first_II":   dict(first_embed="I", first_hidden="I",
                                rec_embed="R1T", rec_hidden="R1T", head_mode="fused"),
        "abl2_first_R1TR1T": dict(first_embed="R1T", first_hidden="R1T",
                                  rec_embed="R1T", rec_hidden="R1T", head_mode="fused"),
        "abl3_unfused_head": dict(first_embed="R1T", first_hidden="I",
                                  rec_embed="R1T", rec_hidden="R1T", head_mode="unfused"),
        "abl4_expose_h_hat": dict(first_embed="R1T", first_hidden="I",
                                  rec_embed="R1T", rec_hidden="R1T", head_mode="fused"),
    }
    dkw = ABL[args.ablation]
    expose_h_hat = (args.ablation == "abl4_expose_h_hat")
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"
    device = "cuda:0"
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    if not os.path.isfile(os.path.join(run_dir, "environment.txt")):
        with open(os.path.join(run_dir, "environment.txt"), "w") as f:
            f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")
    ph = args.phase or f"quant_{args.quant}"

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    tmpl = args.chat_template or cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)

    print(f"[fix] building target quant={args.quant}...", flush=True)
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", args.quant, 0, device=device,
        rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)

    tail = wf.UnfusedTailAdapter(model, stash, expose_h_hat=expose_h_hat).install()
    draft = wf.DraftProjTransformAdapter(model, stash, device, torch.float16,
                                         trace=True, **dkw).install()

    # ---------- Test A: target tail correctness ----------
    print("[fix] Test A: target tail...", flush=True)
    a_rows = []
    for p in prompts:
        ids = build_prompt(tok, p["text"]).to(DEV)
        # capture x_R via a temporary hook on the (patched) norm's INPUT
        cap = {}
        h = model.base_model.model.norm.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x_R", i[0].detach()))
        _ = model.base_model.model(input_ids=ids)
        h.remove()
        fused, unfused = tail.logits_from_xR(cap["x_R"].double())
        m = logit_metrics(unfused, fused)
        m["prompt_id"] = p["question_id"]
        a_rows.append(m)
    logging_utils.write_csv(os.path.join(run_dir, f"target_tail_correctness__{ph}.csv"), a_rows)
    A_ok = all(r["top1_agreement"] > 0.999 for r in a_rows)
    with open(os.path.join(run_dir, f"target_tail_correctness__{ph}.json"), "w") as f:
        json.dump({"rows": a_rows, "all_top1_match": A_ok}, f, indent=2)
    print(f"[fix] Test A top1 match all prompts: {A_ok}", flush=True)

    # ---------- Test D + E: AR vs EAGLE greedy ----------
    print("[fix] Test D: AR vs EAGLE greedy...", flush=True)
    d_rows, e_rows, jsonl = [], [], []
    first_mismatch_dump = None
    for p in prompts:
        ids = build_prompt(tok, p["text"]).to(DEV)
        draft.set_context(p["question_id"])
        input_len = ids.shape[1]
        # AR (target-only greedy)
        ar_toks, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                             max_steps=args.max_new_tokens + 4, tree_choices=tree),
                             input_len, args.max_new_tokens)
        # EAGLE (speculative greedy)
        eg_toks, deltas = run_gen(model.ea_generate(ids, temperature=0.0,
                                  max_steps=args.max_new_tokens + 8, tree_choices=tree),
                                  input_len, args.max_new_tokens)
        n = min(len(ar_toks), len(eg_toks))
        mism = next((i for i in range(n) if ar_toks[i] != eg_toks[i]), -1)
        exact = (ar_toks[:n] == eg_toks[:n]) and (len(ar_toks) == len(eg_toks))
        accs = [d for d in deltas]
        d_rows.append({
            "prompt_id": p["question_id"],
            "prompt_text": p["text"][:80].replace("\n", " "),
            "ar_output_text": tok.decode(ar_toks, skip_special_tokens=True)[:120],
            "eagle_output_text": tok.decode(eg_toks, skip_special_tokens=True)[:120],
            "exact_token_match": bool(exact),
            "first_mismatch_position": mism,
            "ar_tokens": str(ar_toks), "eagle_tokens": str(eg_toks),
            "num_new_tokens": len(ar_toks),
            "acceptance_length_mean": (sum(accs) / len(accs)) if accs else 0.0,
            "acceptance_length_list": str(accs),
            "notes": "" if exact else f"MISMATCH at {mism}"})
        jsonl.append(d_rows[-1])

        # Test E: recompute target logits on the AR prefix vs the EAGLE prefix
        base = ids[0].tolist()
        for step in range(min(args.max_new_tokens, n)):
            ar_prefix = torch.tensor([base + ar_toks[:step]], device=DEV)
            eg_prefix = torch.tensor([base + eg_toks[:step]], device=DEV)
            lg_ar = model.base_model(ar_prefix, use_cache=False).logits[0, -1]
            lg_vf = model.base_model(eg_prefix, use_cache=False).logits[0, -1]
            e_rows.append({
                "prompt_id": p["question_id"], "step": step,
                "rel_l2_logit_error": ((lg_ar - lg_vf).norm() / (lg_ar.norm() + 1e-9)).item(),
                "max_abs_logit_error": (lg_ar - lg_vf).abs().max().item(),
                "top1_agreement": int(lg_ar.argmax().item() == lg_vf.argmax().item()),
                "ar_top1": int(lg_ar.argmax().item()),
                "verifier_top1": int(lg_vf.argmax().item()),
                "accepted_token": ar_toks[step] if step < len(ar_toks) else -1,
                "notes": "verifier=target logits on EAGLE prefix (same base_model)"})
        if not exact and first_mismatch_dump is None:
            first_mismatch_dump = {
                "prompt_id": p["question_id"], "first_mismatch_position": mism,
                "ar_tokens": ar_toks, "eagle_tokens": eg_toks,
                "ar_token_at_mismatch": ar_toks[mism] if mism >= 0 else None,
                "eagle_token_at_mismatch": eg_toks[mism] if mism >= 0 else None}

    logging_utils.write_csv(os.path.join(run_dir, f"ar_vs_eagle_outputs__{ph}.csv"), d_rows)
    with open(os.path.join(run_dir, f"ar_vs_eagle_outputs__{ph}.jsonl"), "w") as f:
        for r in jsonl:
            f.write(json.dumps(r) + "\n")
    logging_utils.write_csv(os.path.join(run_dir, f"verifier_vs_ar_logits__{ph}.csv"), e_rows)
    D_ok = all(r["exact_token_match"] for r in d_rows)
    E_ok = all(r["top1_agreement"] == 1 for r in e_rows) and \
        all(r["rel_l2_logit_error"] < 1e-3 for r in e_rows)
    if first_mismatch_dump:
        with open(os.path.join(run_dir, f"first_mismatch_dump__{ph}.json"), "w") as f:
            json.dump(first_mismatch_dump, f, indent=2)

    # ---------- Test B/C traces ----------
    logging_utils.write_csv(os.path.join(run_dir, f"projection_branch_trace__{ph}.csv"),
                            draft.proj_trace)
    logging_utils.write_csv(os.path.join(run_dir, f"draft_lm_head_trace__{ph}.csv"),
                            draft.head_trace)
    # B verdict: first fc call [R1.T,I], later [R1.T,R1.T]
    firsts = [r for r in draft.proj_trace if r["is_first_external_target_forward"]]
    recs = [r for r in draft.proj_trace if r["is_recycled_draft_forward"]]
    B_ok = bool(firsts) and all(r["embedding_transform"] == "R1T" and r["hidden_transform"] == "I" for r in firsts) \
        and bool(recs) and all(r["embedding_transform"] == "R1T" and r["hidden_transform"] == "R1T" for r in recs)
    C_ok = bool(draft.head_trace) and all(r["lm_head_mode"].startswith("fused") and
              r["has_R1T_absorbed"] and r["has_gamma_f_absorbed"] for r in draft.head_trace)

    draft.uninstall(); tail.uninstall()

    verdicts = {"phase": ph, "quant": args.quant, "gpu": gpu,
                "A_target_tail_ok": A_ok, "B_projection_branches_ok": B_ok,
                "C_draft_fused_head_ok": C_ok,
                "D_ar_eq_eagle_greedy": D_ok,
                "D_match_rate": sum(r["exact_token_match"] for r in d_rows) / len(d_rows),
                "E_verifier_eq_ar_logits": E_ok,
                "acceptance_mean": sum(r["acceptance_length_mean"] for r in d_rows) / len(d_rows)}
    with open(os.path.join(run_dir, f"verdicts__{ph}.json"), "w") as f:
        json.dump(verdicts, f, indent=2)
    print(json.dumps(verdicts, indent=2))
    if not D_ok:
        print(f"[fix] !!! MISMATCH — see first_mismatch_dump__{ph}.json", flush=True)
    return 0 if D_ok else 3


if __name__ == "__main__":
    sys.exit(main())
