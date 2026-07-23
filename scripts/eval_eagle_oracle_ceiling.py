#!/usr/bin/env python
"""Oracle-draft AL ceiling + policy-theoretical ceiling (spec section 12).

Oracle draft = perfect draft that proposes the deployed target's SEQUENTIAL
greedy continuation at every depth, under the SAME tree policy
(mc_sim_7b_63), verifier advance rule (accepted + 1 bonus), EOS and
max-new-token budget. With greedy verification every proposed token on the
deepest tree chain is accepted, so per cycle the verifier advances
  min(L_max + 1, remaining)
where L_max = deepest root-to-leaf draft depth of the tree. We MEASURE the
resulting micro-tau on real sequential-greedy continuations (EOS/length
effects included) rather than assuming the bound.

Policy-theoretical ceiling: tau_max = L_max + 1 (recorded).

--self-test runs the Gate J toy validation (no GPU).
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))


def oracle_cycles(n_tokens, l_max):
    """Advance pattern of a perfect draft under the EAGLE verifier:
    each cycle accepts min(l_max, remaining-1) drafts + 1 verifier token."""
    deltas, pos = [], 0
    while pos < n_tokens:
        adv = min(l_max + 1, n_tokens - pos)
        deltas.append(adv)
        pos += adv
    return deltas


def self_test():
    # toy chain: 13 tokens, depth 5 tree -> cycles of 6,6,1
    assert oracle_cycles(13, 5) == [6, 6, 1]
    # exactly one full cycle
    assert oracle_cycles(6, 5) == [6]
    # sub-cycle generation (EOS after 3 tokens)
    assert oracle_cycles(3, 5) == [3]
    # chain tree depth 1 -> pure 2-token cycles
    assert oracle_cycles(8, 1) == [2, 2, 2, 2]
    # zero tokens -> no cycles
    assert oracle_cycles(0, 5) == []
    # tau equals the policy ceiling on long generations
    d = oracle_cycles(600, 5)
    assert abs(sum(d) / len(d) - 6.0) < 0.01
    print("[oracle] self-test PASS (Gate J toy validation)")


@torch.no_grad() if False else (lambda f: f)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--target", choices=["fp16", "int4"])
    ap.add_argument("--datasets", default="mtbench")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    assert args.target and args.run_dir
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))

    import torch
    from eagle_spinquant import eagle_bridge, experiment, logging_utils, \
        study
    from eagle_spinquant.eval_datasets import load_eval_prompts
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    l_max = max(len(p) for p in tree)

    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, "learned_chat_w4a4kv16", quant, 0, device=dev,
        rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    eos = tok.eos_token_id

    from eagle.model.kv_cache import initialize_past_key_values

    def seq_greedy(ids):
        # vendored KV llama needs its managed cache object (validated
        # pattern from check_verifier_correctness.py)
        past, _pd, _cl = initialize_past_key_values(model.base_model)
        cur, out = ids, []
        for _ in range(args.max_new_tokens):
            res = model.base_model(input_ids=cur, past_key_values=past,
                                   use_cache=True)
            nxt = int(res.logits[:, -1].argmax(-1))
            out.append(nxt)
            if nxt == eos:
                break
            cur = torch.tensor([[nxt]], device=dev)
        return out

    for ds_name in args.datasets.split(","):
        out_csv = os.path.join(
            args.run_dir, "shards",
            f"al__ORACLE_{args.target}__{args.target}__{ds_name}.csv")
        if os.path.exists(out_csv):
            print(f"[oracle] {ds_name}: exists, skip", flush=True)
            continue
        prompts, _ = load_eval_prompts(ds_name, args.n_prompts, "eval")
        rows = []
        for p in prompts:
            ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
            seq = seq_greedy(ids)
            deltas = oracle_cycles(len(seq), l_max)
            rows.append(dict(
                tag=f"ORACLE_{args.target}", target=args.target,
                dataset=ds_name, prompt_id=p["row_id"],
                acceptance_list=json.dumps(deltas),
                n_cycles=len(deltas),
                seq_tokens=json.dumps(seq)))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        logging_utils.write_csv(out_csv, rows)
        taus = [t for r in rows for t in json.loads(r["acceptance_list"])]
        tau = sum(taus) / max(len(taus), 1)
        os.makedirs(os.path.join(args.run_dir, "oracle"), exist_ok=True)
        with open(os.path.join(
                args.run_dir, "oracle",
                f"oracle_{args.target}_{ds_name}.json"), "w") as f:
            json.dump(dict(target=args.target, dataset=ds_name,
                           tree="mc_sim_7b_63", l_max=l_max,
                           policy_theoretical_tau=l_max + 1,
                           oracle_tau_measured=tau,
                           n_prompts=len(rows)), f, indent=1)
        print(f"[oracle] {args.target}/{ds_name}: oracle tau={tau:.4f} "
              f"(policy ceiling {l_max + 1})", flush=True)
    print(f"[oracle] {args.target} DONE", flush=True)
    return 0


if __name__ == "__main__":
    import torch  # noqa: F401  (imported late for --self-test without GPU)
    sys.exit(main())
