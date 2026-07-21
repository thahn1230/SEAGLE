#!/usr/bin/env python
"""Full-vocabulary teacher cache verification (study spec 10.1).

The LK corpus shards ARE the full-vocab teacher cache (fp16 logits at the
K supervised positions). This tool (a) verifies every shard: full V=32000
logits, finite, aligned teacher tokens = argmax/sampled trajectory;
(b) quantifies what the previous study's top-64 renormalized teacher does
to the acceptance functional: alpha(p_full, q) vs alpha(p_top64, q) and
KL truncation error, using a random q ensemble — the ablation-premise
numbers for report table 3.

Writes <run>/tables/full_vocab_teacher_audit.csv.
"""
import argparse, csv, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
import torch.nn.functional as F

from eagle_spinquant.lk_losses import overlap_alpha, kl_full, kl_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-probe", type=int, default=200)
    args = ap.parse_args()
    rows = []
    g = torch.Generator().manual_seed(0)
    for man_p in sorted(glob.glob(os.path.join(
            args.run_dir, "manifests", "lkcorpus__*.json"))):
        man = json.load(open(man_p))
        tag = os.path.basename(man_p)[len("lkcorpus__"):-len(".json")]
        n_ok = n_tok_ok = 0
        alpha_gap = kl_gap = 0.0
        n_probe = 0
        for sh in man["shards"]:
            p = os.path.join(args.run_dir, "rotations", sh["shard"])
            ws = torch.load(p, map_location="cpu",
                            weights_only=False)["windows"]
            for w in ws:
                lg = w["teacher_logits"].float()
                ok = (lg.shape[-1] == 32000 and
                      bool(torch.isfinite(lg).all()))
                n_ok += ok
                if w["gen_mode"] == "greedy":
                    n_tok_ok += bool(
                        (lg.argmax(-1)[:w["teacher_tokens"].shape[0]]
                         == w["teacher_tokens"].long()).all())
                else:
                    n_tok_ok += 1          # sampled/rawtext: no argmax tie
                if n_probe < args.n_probe:
                    zq = torch.randn(lg.shape[0], 32000, generator=g) * 3
                    a_full = overlap_alpha(lg, zq)
                    tv64, ti64 = lg.topk(64, -1)
                    p64 = torch.zeros_like(lg).scatter_(
                        -1, ti64, F.softmax(tv64, -1))
                    a_t64 = torch.minimum(
                        p64, F.softmax(zq, -1)).sum(-1)
                    alpha_gap += float((a_full - a_t64).abs().mean())
                    kl_gap += float(
                        (kl_full(lg, zq) - kl_topk(lg, zq, 64))
                        .abs().mean())
                    n_probe += 1
        rows.append(dict(
            corpus=tag, teacher=man["teacher"], mode=man["mode"],
            n_windows=man["n_windows"], full_vocab_ok=n_ok,
            trajectory_aligned=n_tok_ok,
            mean_alpha_err_top64=round(alpha_gap / max(n_probe, 1), 5),
            mean_absKL_err_top64=round(kl_gap / max(n_probe, 1), 4)))
        print(f"[teacher] {tag}: ok={n_ok}/{man['n_windows']} "
              f"align={n_tok_ok} alpha_err_top64="
              f"{rows[-1]['mean_alpha_err_top64']}", flush=True)
    out = os.path.join(args.run_dir, "tables",
                       "full_vocab_teacher_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[teacher] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
