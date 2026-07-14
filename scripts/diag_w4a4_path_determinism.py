#!/usr/bin/env python
"""Root-cause diagnostic for the Phase-2 mismatch: is fake-W4A4 greedy
path-dependent (incremental use_cache=True vs full use_cache=False vs EAGLE
tree), independent of the unfused-tail interface?

Tests, on the PLAIN W4A4 target (NO tail/draft adapters) and a fixed prompt:
  1. full forward (use_cache=False) argmax + top-2 gap, run twice (self-determinism)
  2. incremental decode (use_cache=True, token-by-token) argmax at each step
  3. do (1) and (2) agree at each step? report the top-2 logit gap at each flip.

If full vs incremental disagree with TINY top-2 gaps, the mismatch is fake-W4A4
argmax non-determinism across execution paths, not an interface bug.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/diag_w4a4_path_determinism.py \
      --out-dir runs/w4a4_impl_fix_<ts> --quant w4a4 --steps 16
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study  # noqa: E402

DEV = "cuda:0"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quant", default="w4a4", choices=["none", "w4a4"])
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", args.quant, 0, device=DEV, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    p = experiment.load_mt_bench_prompts(2)[0]
    ids = eagle_bridge.PROMPT_BUILDERS[tmpl](tok, p["text"]).to(DEV)
    base = model.base_model

    def full_argmax(prefix):
        lg = base(prefix, use_cache=False).logits[0, -1].float()
        top = lg.topk(2)
        return int(top.indices[0]), (top.values[0] - top.values[1]).item()

    # self-determinism of the full forward
    a1, _ = full_argmax(ids)
    a2, _ = full_argmax(ids)
    self_det = (a1 == a2)

    # incremental (use_cache=True) greedy, EXACTLY like naive_generate
    inc = []
    cur = ids.clone()
    from eagle.model.utils import reset_tree_mode
    from eagle.model.kv_cache import initialize_past_key_values
    pkv, pkv_data, cur_len = initialize_past_key_values(base)
    reset_tree_mode(model)
    out = base(cur, past_key_values=pkv, use_cache=True)
    rows = []
    for step in range(args.steps):
        inc_tok = int(out.logits[0, -1].argmax())
        # full-forward argmax on the SAME prefix built from incremental tokens
        full_tok, gap = full_argmax(cur)
        rows.append({"step": step, "incremental_top1": inc_tok,
                     "full_forward_top1": full_tok, "agree": int(inc_tok == full_tok),
                     "full_top2_gap": round(gap, 4)})
        nxt = torch.tensor([[inc_tok]], device=DEV)
        out = base(nxt, past_key_values=pkv, use_cache=True)
        cur = torch.cat([cur, nxt], -1)
        inc.append(inc_tok)

    n_flip = sum(1 for r in rows if not r["agree"])
    gaps_at_flip = [r["full_top2_gap"] for r in rows if not r["agree"]]
    out_json = {
        "gpu": gpu, "quant": args.quant, "prompt_id": p["question_id"],
        "full_forward_self_deterministic": bool(self_det),
        "steps": args.steps, "n_incremental_vs_full_disagreements": n_flip,
        "top2_gap_at_disagreements": gaps_at_flip,
        "mean_top2_gap_at_flip": (sum(gaps_at_flip) / len(gaps_at_flip)) if gaps_at_flip else None,
        "conclusion": ("fake-W4A4 greedy is PATH-DEPENDENT (incremental vs full "
                       "forward disagree at near-ties)" if n_flip else
                       "no path-dependence at this prompt"),
    }
    logging_utils.write_csv(os.path.join(args.out_dir,
                            f"path_determinism_{args.quant}.csv"), rows)
    with open(os.path.join(args.out_dir,
              f"path_determinism_{args.quant}.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(json.dumps(out_json, indent=2))
    print("\nper-step:")
    for r in rows:
        print(f"  step {r['step']:2d}  inc={r['incremental_top1']:5d}  "
              f"full={r['full_forward_top1']:5d}  agree={r['agree']}  "
              f"gap={r['full_top2_gap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
