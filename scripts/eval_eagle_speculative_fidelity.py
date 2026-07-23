#!/usr/bin/env python
"""Speculative-fidelity audit (spec section 16, Gate L).

For the named target configuration, per prompt and greedy decoding:
  (a) sequential autoregressive generation with the bare target
  (b) EAGLE tree generation under the SAME target (stock draft for the
      FP16 target; strict D4P3 gamma_R1 draft for the INT4 target — the
      draft cannot change the emitted tokens if verification is faithful,
      only the speed)
and reports exact-prefix agreement. The dynamic per-token A4 activation
quantizer is execution-shape dependent, so (b) may diverge from (a) for
the INT4 target; if it does, the system must not be called lossless.

Writes shards/fidelity__<target>__mtbench.csv and
tables/speculative_fidelity_<target>.json. The saved sequential token
streams also feed the cross-target (INT4-vs-FP16 sequential) agreement
table computed by analyze_qat_attainment.py.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["fp16", "int4"])
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    eos = tok.eos_token_id

    ad = None
    if args.target == "int4":
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            embed_scale_alpha=32.0, quant_first="fake_w4a4",
            quant_recurrent="fake_w4a4", quant_ar="fake_w4a4",
            ar_r2r4=True)
        ad.install()

    def seq_greedy(ids):
        past, cur, out = None, ids, []
        for _ in range(args.max_new_tokens):
            res = model.base_model(input_ids=cur, past_key_values=past,
                                   use_cache=True)
            past = res.past_key_values
            nxt = int(res.logits[:, -1].argmax(-1))
            out.append(nxt)
            if nxt == eos:
                break
            cur = torch.tensor([[nxt]], device=dev)
        return out

    def eagle_greedy(ids):
        prev, toks = ids.shape[1], None
        for out in model.ea_generate(ids, temperature=0.0,
                                     max_steps=args.max_new_tokens + 8,
                                     tree_choices=tree):
            toks = out
            if out.shape[1] - ids.shape[1] >= args.max_new_tokens:
                break
        seq = toks[0, ids.shape[1]:].tolist() if toks is not None else []
        if eos in seq:
            seq = seq[:seq.index(eos) + 1]
        return seq[:args.max_new_tokens]

    prompts, _ = load_eval_prompts("mtbench", args.n_prompts, "eval")
    rows, n_exact, agree_sum = [], 0, 0.0
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
        s = seq_greedy(ids)
        e = eagle_greedy(ids)
        m = min(len(s), len(e))
        k = 0
        while k < m and s[k] == e[k]:
            k += 1
        exact = (k == m and len(s) == len(e))
        frac = k / max(m, 1)
        n_exact += int(exact)
        agree_sum += frac
        rows.append(dict(target=args.target, prompt_id=p["row_id"],
                         seq_len=len(s), eagle_len=len(e),
                         agree_prefix=k, agree_frac=round(frac, 4),
                         exact=int(exact),
                         seq_tokens=json.dumps(s),
                         eagle_tokens=json.dumps(e)))
    out_csv = os.path.join(args.run_dir, "shards",
                           f"fidelity__{args.target}__mtbench.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    logging_utils.write_csv(out_csv, rows)
    summ = dict(target=args.target, n=len(rows),
                exact_match_rate=n_exact / max(len(rows), 1),
                mean_prefix_agreement=agree_sum / max(len(rows), 1),
                lossless=bool(n_exact == len(rows)))
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables",
                           f"speculative_fidelity_{args.target}.json"),
              "w") as f:
        json.dump(summ, f, indent=1)
    if ad is not None:
        ad.uninstall()
    print(f"[fidelity] {summ}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
