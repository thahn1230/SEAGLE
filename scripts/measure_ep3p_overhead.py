#!/usr/bin/env python
"""Pathwise EP3-P runtime/memory overhead vs LP3 (study §10).

Runs the standard evaluator on a small MT-Bench slice for LP3
(single global alpha) and EP3-P (independent first/recurrent factors:
adds one elementwise rescale on the recurrent e-slice), same GPU,
same prompts. Reports wall-clock per generated token from the shard's
gen_seconds and peak GPU memory sampled via nvidia-smi.
Writes tables/ep3p_overhead.json.
"""
import argparse, csv, json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def peak_mem_watcher(stop, out, gpu):
    peak = 0
    while not stop.is_set():
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", str(gpu)],
                capture_output=True, text=True, timeout=10)
            peak = max(peak, int(q.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(1)
    out["peak_mib"] = peak


def shard_stats(rd, tag, tgt):
    p = os.path.join(rd, "shards", f"al__{tag}__{tgt}__mtbench.csv")
    secs = toks = 0.0
    for r in csv.DictReader(open(p)):
        secs += float(r["gen_seconds"])
        toks += sum(json.loads(r["acceptance_list"]))
    return secs, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", default="int4",
                    choices=["fp16", "int4"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=8)
    args = ap.parse_args()
    rd = args.run_dir
    sel = json.load(open(os.path.join(
        rd, "tables", f"ep3_selection_{args.target}.json")))
    pb = sel.get("primary_rule3") or sel.get("primary_by_al")
    legacy = 45.254834 if args.target == "fp16" else 32.0
    runs = dict(
        OVH_LP3=["--draft-cfg", "d4p3", "--alpha", str(legacy)],
        OVH_EP3P=["--draft-cfg", "d4p3_deploy", "--alpha",
                  str(pb["m_first"]), "--alpha-rec",
                  str(pb["m_rec"])])
    out = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for tag, extra in runs.items():
        stop, mem = threading.Event(), {}
        th = threading.Thread(target=peak_mem_watcher,
                              args=(stop, mem, args.gpu))
        th.start()
        subprocess.run(
            [PY, "scripts/eval_eagle_acceptance_length.py",
             "--run-dir", rd, "--target", args.target, "--tag", tag,
             "--datasets", f"mtbench:{args.n_prompts}"] + extra,
            cwd=ROOT, env=env)
        stop.set(); th.join()
        secs, toks = shard_stats(rd, tag, args.target)
        out[tag] = dict(gen_seconds=round(secs, 3),
                        gen_tokens=int(toks),
                        ms_per_token=round(1000 * secs
                                           / max(toks, 1), 3),
                        peak_mem_mib=mem.get("peak_mib"))
    a, b = out["OVH_LP3"], out["OVH_EP3P"]
    out["overhead"] = dict(
        time_ratio=round(b["ms_per_token"] / a["ms_per_token"], 4),
        mem_delta_mib=(b["peak_mem_mib"] - a["peak_mem_mib"])
        if (a.get("peak_mem_mib") and b.get("peak_mem_mib")) else None)
    json.dump(out, open(os.path.join(
        rd, "tables", "ep3p_overhead.json"), "w"), indent=1)
    print(f"[ovh] {out['overhead']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
