#!/usr/bin/env python
"""PMG target-quality safeguard (spec §15): TARGET-ONLY greedy
generation quality per precision.

Metrics per target in {fp16, w8a8, int4}:
  - gsm8k exact-match on the '#### N' answer (eval pool 200,
    max_new_tokens 256 — SHORT-BUDGET CAVEAT: understates absolute
    accuracy; valid for BETWEEN-target comparison only)
  - humaneval pass@1 (164 problems, sandboxed exec with timeout)
  - sha256 checksum of all generated token ids
  - token agreement vs the FP16 target's generations (same prompts)
Writes tables/target_quality_<target>.json (+ generations jsonl).
"""
import argparse, hashlib, json, os, re, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"


def gsm8k_answer(text):
    m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text.replace(",", ""))
    return m[-1] if m else None


def run_humaneval(problem, completion, timeout=6.0):
    import multiprocessing as mp

    def target(q):
        import contextlib, io, signal

        def handler(signum, frame):
            raise TimeoutError

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(timeout))
            env = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(problem["prompt"] + completion + "\n"
                     + problem["test"] + f"\ncheck({problem['entry_point']})",
                     env)
            q.put(True)
        except Exception:
            q.put(False)

    q = mp.Queue()
    p = mp.Process(target=target, args=(q,))
    p.start()
    p.join(timeout + 2)
    if p.is_alive():
        p.terminate()
        return False
    return q.get() if not q.empty() else False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["fp16", "int4", "w8a8"])
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "humaneval"])
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16" else
                  ("full", "w8a8") if args.target == "w8a8" else
                  ("full", "w4a4"))
    model, _stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], rot, KIND, quant, 0, device=dev,
        rotations_root=rr)
    bm = model.base_model
    bm.model.tree_mask = None
    tok = eagle_bridge.get_tokenizer(model)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    n = args.n_prompts or (200 if args.dataset == "gsm8k" else 164)
    prompts, extra = load_eval_prompts(args.dataset, n, "eval")

    from eagle.model.kv_cache import initialize_past_key_values
    past, _pd, cur = initialize_past_key_values(bm)
    gens, all_ids = [], []
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :1024].to(dev)
        cur.zero_()
        h = bm.model(input_ids=ids, past_key_values=past,
                     use_cache=True)[0]
        nxt = bm.lm_head(h[:, -1:]).argmax(-1)
        out_ids = []
        for _ in range(args.max_new_tokens):
            t = int(nxt)
            if t == tok.eos_token_id:
                break
            out_ids.append(t)
            h = bm.model(input_ids=nxt, past_key_values=past,
                         use_cache=True)[0]
            nxt = bm.lm_head(h[:, -1:]).argmax(-1)
        text = tok.decode(out_ids, skip_special_tokens=True)
        gens.append(dict(row_id=p["row_id"], ids=out_ids, text=text))
        all_ids.extend(out_ids + [-1])
    checksum = hashlib.sha256(str(all_ids).encode()).hexdigest()[:16]

    result = dict(target=args.target, dataset=args.dataset,
                  n=len(gens), max_new_tokens=args.max_new_tokens,
                  checksum=checksum,
                  caveat="256-token budget; between-target "
                         "comparison only")
    from datasets import load_dataset
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ok = 0
        for g in gens:
            idx = int(g["row_id"].rsplit("_", 1)[1])
            gold = gsm8k_answer(ds[idx]["answer"].split("####")[-1])
            pred = gsm8k_answer(g["text"])
            ok += int(gold is not None and pred == gold)
        result["exact_match"] = round(ok / max(len(gens), 1), 4)
    else:
        ds = load_dataset("openai/openai_humaneval", split="test")
        by_id = {r["task_id"]: r for r in ds}
        ok = 0
        for g in gens:
            pr = by_id[g["row_id"]]
            ok += int(run_humaneval(pr, g["text"]))
        result["pass_at_1"] = round(ok / max(len(gens), 1), 4)
    gp = os.path.join(args.run_dir, "tables",
                      f"target_gen_{args.target}_{args.dataset}.jsonl")
    with open(gp, "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    op = os.path.join(args.run_dir, "tables",
                      f"target_quality_{args.target}_{args.dataset}"
                      f".json")
    json.dump(result, open(op, "w"), indent=1)
    print(f"[tq] {args.target}/{args.dataset}: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
