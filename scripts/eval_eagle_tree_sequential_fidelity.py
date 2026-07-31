#!/usr/bin/env python
"""Target-execution-shape controls (study §19 B/C).

From captured cycle records (cycles/cyc__<tag>__<ds>.jsonl):
  B. sequential-to-sequential: replay each cycle's DEPLOYED accepted
     path token-by-token under T0 and Tq sequential execution; count
     agreement of next-token argmax along the path (isolates model
     quantization drift, no tree shape).
  C. Tq self-fidelity: Tq tree-accepted tokens vs Tq sequential argmax
     along the same path (execution-shape drift of dynamic A4).

Writes tables/tree_seq_fidelity_<tag>.json with RCAL_tree (from the
capture), RCAL_sequential (B), and Tq tree-vs-sequential agreement (C).
Sequential replays teacher-force the deployed trajectory, so prefixes
match the capture exactly.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="capture tag")
    ap.add_argument("--target", required=True, choices=["fp16", "int4"],
                    help="Tq of the capture")
    ap.add_argument("--dataset", default="mtbench")
    ap.add_argument("--pool", default="eval")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    cyc_p = os.path.join(rd, "cycles",
                         f"cyc__{args.tag}__{args.dataset}.jsonl")
    recs = [json.loads(x) for x in open(cyc_p)]
    by_prompt = {}
    for r in recs:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    dev = "cuda:0"
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w4a4"))
    # Tq base target (no draft needed for sequential replay)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    tq = model.base_model
    from transformers import AutoModelForCausalLM
    t0 = AutoModelForCausalLM.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(
        "cuda:1").eval()
    tok = eagle_bridge.get_tokenizer(model)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts, _ = load_eval_prompts(args.dataset, args.n_prompts,
                                   args.pool)
    pmap = {p["row_id"]: p for p in prompts}
    from eagle.model.kv_cache import initialize_past_key_values

    def seq_argmax_along(modelx, dvc, ids_prefix, traj):
        """Teacher-forced: feed prefix then each trajectory token;
        return the argmax prediction made BEFORE each token."""
        modelx.model.tree_mask = None
        past, _pd, _cl = initialize_past_key_values(modelx)
        out = modelx(ids_prefix.to(dvc), past_key_values=past,
                     use_cache=True)
        preds = []
        for t in traj:
            preds.append(int(out.logits[:, -1].argmax(-1)))
            out = modelx(torch.tensor([[t]], device=dvc),
                         past_key_values=past, use_cache=True)
        return preds

    n_tok = agree_00 = agree_q = agree_seqpair = 0
    n_prompts_done = 0
    for pid, rows in list(by_prompt.items())[:args.max_prompts]:
        if pid not in pmap:
            continue
        ids = build_prompt(tok, pmap[pid]["text"])[:, :1024]
        # deployed trajectory: accepted tokens + per-cycle correction
        traj = []
        for r in sorted(rows, key=lambda x: x["cycle"]):
            traj += r["S_q"] + [r["corr_q"]]
        if not traj:
            continue
        pq = seq_argmax_along(tq, dev, ids, traj)
        p0 = seq_argmax_along(t0, "cuda:1", ids, traj)
        for i, t in enumerate(traj):
            n_tok += 1
            agree_q += int(pq[i] == t)      # Tq tree-vs-seq (C)
            agree_00 += int(p0[i] == t)     # T0 seq agreement w/ traj
            agree_seqpair += int(pq[i] == p0[i])   # B: seq-vs-seq
        n_prompts_done += 1
    out = dict(tag=args.tag, target=args.target,
               n_prompts=n_prompts_done, n_tokens=n_tok,
               tq_tree_vs_seq_agreement=agree_q / max(n_tok, 1),
               t0_seq_agreement_with_deployed_traj=agree_00
               / max(n_tok, 1),
               tq_seq_vs_t0_seq_agreement=agree_seqpair
               / max(n_tok, 1))
    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(
        rd, "tables", f"tree_seq_fidelity_{args.tag}.json"), "w"),
        indent=1)
    print(f"[treeseq] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
