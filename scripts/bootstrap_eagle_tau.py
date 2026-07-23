#!/usr/bin/env python
"""Paired prompt-cluster bootstrap for Average Acceptance Length (tau).

Reuses the repo's official aggregation contract (aggregate_micro_al.py):
micro tau = pooled sum(acceptance)/cycles over prompt clusters; paired
bootstrap resamples prompt clusters aligned by prompt_id with >= 2000
replicates (default 3000).

  python scripts/bootstrap_eagle_tau.py --run-dir RD \
      --pair "C2_fp16_d4p3@fp16:C3_s0_qat@fp16" [--dataset mtbench]

--battery runs every registered study comparison found on disk and writes
stats/bootstrap_pairs.json.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_micro_al import read_shard, pooled_mal, boot_ci, paired_boot

BATTERY = [
    # (name, tagA@target, tagB@target)  delta = tau_B - tau_A
    ("qat_gain_fp16_s0", "C2_fp16_d4p3@fp16", "C3_s0_qat@fp16"),
    ("qat_gain_fp16_s1", "C2_fp16_d4p3@fp16", "C3_s1_qat@fp16"),
    ("qat_gain_fp16_s2", "C2_fp16_d4p3@fp16", "C3_s2_qat@fp16"),
    ("qat_gain_int4_s0", "C5_int4_d4p3@int4", "C7_s0_qat@int4"),
    ("qat_gain_int4_s1", "C5_int4_d4p3@int4", "C7_s1_qat@int4"),
    ("qat_gain_int4_s2", "C5_int4_d4p3@int4", "C7_s2_qat@int4"),
    ("rot_vs_qat_fp16", "C11_fp16_rot@fp16", "C3_s0_qat@fp16"),
    ("rot_vs_qat_int4", "C12_int4_rot@int4", "C7_s0_qat@int4"),
    ("rot_gain_fp16", "C2_fp16_d4p3@fp16", "C11_fp16_rot@fp16"),
    ("rot_gain_int4", "C5_int4_d4p3@int4", "C12_int4_rot@int4"),
    ("retrainptq_vs_qat_fp16", "C9_c8ptq@fp16", "C3_s0_qat@fp16"),
    ("retrainptq_vs_qat_int4", "C10_c6ptq@int4", "C7_s0_qat@int4"),
    ("teacher_mismatch_fp16draft", "C13_c8_at_int4@int4",
     "C6_tgtadapt@int4"),
    ("teacher_mismatch_qat", "C14_c3_at_int4@int4", "C7_s0_qat@int4"),
    ("tgt_adapt_gain", "C4_int4_stockrestored@int4", "C6_tgtadapt@int4"),
    ("fp16_retrain_gain", "C1_fp16_stock@fp16", "C8_fp16retrain@fp16"),
    ("staged_init_c7b", "C7_s0_qat@int4", "C7b_qat@int4"),
    ("p3_recovery_fp16", "N1_fp16_naive@fp16", "C2_fp16_d4p3@fp16"),
    ("p3_recovery_int4", "N2_int4_naive@int4", "C5_int4_d4p3@int4"),
    ("target_ptq_cost", "C1_fp16_stock@fp16", "C4_int4_stockrestored@int4"),
]


def shard_path(rd, spec, dataset):
    tag, _, tgt = spec.partition("@")
    return os.path.join(rd, "shards", f"al__{tag}__{tgt}__{dataset}.csv")


def clusters(path):
    return [(pid, taus) for pid, taus, _ in read_shard(path)]


def run_pair(rd, name, a, b, dataset, reps, rng):
    pa, pb = shard_path(rd, a, dataset), shard_path(rd, b, dataset)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return dict(name=name, a=a, b=b, dataset=dataset, status="missing",
                    missing=[p for p in (pa, pb)
                             if not os.path.exists(p)])
    ca, cb = clusters(pa), clusters(pb)
    ta, tb = pooled_mal(ca), pooled_mal(cb)
    la, ha = boot_ci(ca, rng, reps)
    lb, hb = boot_ci(cb, rng, reps)
    d, lo, hi, p2, n_common = paired_boot(ca, cb, rng, reps)
    # cross-target pairs are paired by prompt identity (same prompts under
    # both systems); prompt ids match across targets by construction
    return dict(name=name, a=a, b=b, dataset=dataset, status="ok",
                tau_a=round(ta, 4), ci_a=[round(la, 4), round(ha, 4)],
                tau_b=round(tb, 4), ci_b=[round(lb, 4), round(hb, 4)],
                delta_b_minus_a=round(d, 4),
                ci_delta=[round(lo, 4), round(hi, 4)],
                p_two_sided=p2, n_common=n_common, reps=reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pair", action="append", default=[],
                    help="name=A@tgt:B@tgt or A@tgt:B@tgt")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--dataset", default="mtbench")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    todo = []
    if args.battery:
        todo += [(n, a, b) for n, a, b in BATTERY]
    for p in args.pair:
        name, _, rest = p.partition("=")
        if not rest:
            name, rest = p.replace("@", "_").replace(":", "__vs__"), p
        a, _, b = rest.partition(":")
        todo.append((name, a, b))
    out = [run_pair(args.run_dir, n, a, b, args.dataset, args.reps, rng)
           for n, a, b in todo]
    os.makedirs(os.path.join(args.run_dir, "stats"), exist_ok=True)
    path = os.path.join(args.run_dir, "stats",
                        f"bootstrap_pairs_{args.dataset}.json")
    json.dump(out, open(path, "w"), indent=1)
    for r in out:
        if r["status"] == "ok":
            print(f"[boot] {r['name']}: {r['tau_a']} -> {r['tau_b']} "
                  f"delta={r['delta_b_minus_a']} CI={r['ci_delta']} "
                  f"p={r['p_two_sided']}")
        else:
            print(f"[boot] {r['name']}: MISSING {r['missing']}")
    print(f"[boot] -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
