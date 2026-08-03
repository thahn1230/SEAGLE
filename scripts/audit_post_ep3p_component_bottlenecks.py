#!/usr/bin/env python
"""Post-EP3-P component bottleneck audit (study §4).

Two families on the W4A4 target (T1), all via the validated evaluator
with per-component quantization flags (EP3-P scaling exact-FP-neutral
when a projection stays fp16):

  quantize-one (A0-A18): FP16 draft baseline + one component W4A4
  restore-one  (B0-B12): full EP3-P W4A4 deploy + one component FP16

Phases: run (mtbench 80, 8-GPU scheduler) -> table (per-arm tau +
paired prompt bootstrap vs the family baseline).
Writes tables/component_audit.json.
"""
import argparse, csv, json, os, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
D = 4096
MF, MR = D ** 0.40, D ** 0.45
ALL_AR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
          "up_proj", "down_proj"]


def arms():
    """name -> extra evaluator args."""
    fp = ["--quant-first", "fp16", "--quant-recurrent", "fp16",
          "--quant-ar", "fp16"]
    ep3p = ["--alpha", str(MF), "--alpha-rec", str(MR)]
    one = ["--alpha", "1.0"]
    a = {}
    a["FPDRAFT"] = fp + one                              # reference
    a["A0_embed"] = fp + one + ["--quant-embed", "fake_w4a4"]
    a["A1_head"] = fp + one + ["--quant-head", "fake_w4a4"]
    a["A2_first"] = one + ["--quant-recurrent", "fp16",
                           "--quant-ar", "fp16"]
    a["A3_rec"] = one + ["--quant-first", "fp16",
                         "--quant-ar", "fp16"]
    a["A4_proj_naive"] = one + ["--quant-ar", "fp16"]
    a["A5_proj_ep3p"] = ep3p + ["--quant-ar", "fp16"]
    for nm, mask in (("A6_q", "q_proj"), ("A7_k", "k_proj"),
                     ("A8_v", "v_proj"), ("A9_o", "o_proj"),
                     ("A10_qkv", "q_proj,k_proj,v_proj"),
                     ("A11_qkvo", "q_proj,k_proj,v_proj,o_proj"),
                     ("A12_gate", "gate_proj"), ("A13_up", "up_proj"),
                     ("A14_down", "down_proj"),
                     ("A15_gateup", "gate_proj,up_proj"),
                     ("A16_mlp", "gate_proj,up_proj,down_proj")):
        a[nm] = (["--quant-first", "fp16", "--quant-recurrent",
                  "fp16"] + one + ["--ar-mask", mask])
    a["A17_ar"] = (["--quant-first", "fp16", "--quant-recurrent",
                    "fp16"] + one)
    a["A18_ep3p_ar"] = ep3p                              # proj EP3P+AR
    # restore family from the full EP3-P deployment
    a["FULL"] = ep3p                                     # baseline: is
    # FULL == A18? A18 quantizes proj+AR (embed/head fp16 = deployed
    # contract) -> FULL is the same config; keep one tag, alias below.
    a["B0_first_fp"] = ep3p + ["--quant-first", "fp16"]
    a["B1_rec_fp"] = ep3p + ["--quant-recurrent", "fp16"]
    a["B2_proj_fp"] = ep3p + ["--quant-first", "fp16",
                              "--quant-recurrent", "fp16"]
    for nm, drop in (("B3_q", "q_proj"), ("B4_k", "k_proj"),
                     ("B5_v", "v_proj"), ("B6_o", "o_proj")):
        keep = ",".join(x for x in ALL_AR if x != drop)
        a[nm] = ep3p + ["--ar-mask", keep]
    a["B7_attn_fp"] = ep3p + ["--ar-mask",
                              "gate_proj,up_proj,down_proj"]
    for nm, drop in (("B8_gate", "gate_proj"), ("B9_up", "up_proj"),
                     ("B10_down", "down_proj")):
        keep = ",".join(x for x in ALL_AR if x != drop)
        a[nm] = ep3p + ["--ar-mask", keep]
    a["B11_mlp_fp"] = ep3p + ["--ar-mask",
                              "q_proj,k_proj,v_proj,o_proj"]
    a["B12_ar_fp"] = ep3p + ["--quant-ar", "fp16"]
    return a


def sched(jobs, gpus, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    free = list(gpus); running = []; queue = list(jobs)
    while queue or running:
        for i in range(len(running) - 1, -1, -1):
            pr, name, asg = running[i]
            if pr.poll() is not None:
                print(f"[sched] {'done' if pr.returncode == 0 else 'FAIL'} {name}", flush=True)
                free.extend(asg); running.pop(i)
        while queue and free:
            name, cmd = queue.pop(0)
            g = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
            lf = open(os.path.join(log_dir, f"{name}.log"), "a")
            running.append((subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                cwd=ROOT), name, g and [g] or [g]))
            print(f"[sched] start {name} on {g}", flush=True)
        time.sleep(10)


def prompt_taus(p):
    out = {}
    for r in csv.DictReader(open(p)):
        ts = json.loads(r["acceptance_list"])
        out[r["prompt_id"]] = (sum(ts), len(ts))
    return out


def paired_ci(a, b, reps=3000):
    ks = sorted(set(a) & set(b))
    rng = np.random.default_rng(20260803)
    def agg(d, kk):
        s = sum(d[k][0] for k in kk); c = sum(d[k][1] for k in kk)
        return s / max(c, 1)
    point = agg(b, ks) - agg(a, ks)
    idx = rng.integers(0, len(ks), size=(reps, len(ks)))
    ds = [agg(b, [ks[j] for j in row]) - agg(a, [ks[j] for j in row])
          for row in idx]
    return point, [float(np.percentile(ds, 2.5)),
                   float(np.percentile(ds, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["run", "table"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    A = arms()
    if args.phase == "run":
        jobs = []
        for name, extra in A.items():
            sh = os.path.join(rd, "shards",
                              f"al__CMP_{name}__int4__mtbench.csv")
            if os.path.exists(sh):
                continue
            jobs.append((name, [
                PY, "scripts/eval_eagle_acceptance_length.py",
                "--run-dir", rd, "--target", "int4", "--draft-cfg",
                "d4p3_deploy", "--tag", f"CMP_{name}",
                "--datasets", "mtbench", "--pool", "eval"] + extra))
        sched(jobs, list(range(1, 8)),   # GPU 0 excluded (shared box)
              os.path.join(rd, "logs"))
        return 0
    # table
    data = {}
    for name in A:
        p = os.path.join(rd, "shards",
                         f"al__CMP_{name}__int4__mtbench.csv")
        if os.path.exists(p):
            data[name] = prompt_taus(p)
    tau = {n: round(sum(v[0] for v in d.values())
                    / max(sum(v[1] for v in d.values()), 1) + 1, 4)
           for n, d in data.items()}
    out = dict(tau=tau, paired={})
    for n in data:
        if n.startswith("A") and n != "A18_ep3p_ar" \
                and "FPDRAFT" in data:
            pt, ci = paired_ci(data[n], data["FPDRAFT"])
            out["paired"][f"{n}_cost_vs_FPDRAFT"] = dict(
                delta=round(-pt, 4) * -1, ci=ci)
        if n.startswith("B") and "FULL" in data:
            pt, ci = paired_ci(data["FULL"], data[n])
            out["paired"][f"{n}_gain_vs_FULL"] = dict(
                delta=round(pt, 4), ci=ci,
                significant=bool(pt >= 0.05 and ci[0] > 0))
    json.dump(out, open(os.path.join(
        rd, "tables", "component_audit.json"), "w"), indent=1)
    for n in sorted(tau):
        print(f"[cmp] {n:16s} tau={tau[n]}")
    for k, v in out["paired"].items():
        if k.endswith("gain_vs_FULL"):
            print(f"[cmp] {k}: +{v['delta']} {v['ci']} "
                  f"{'SIG' if v.get('significant') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
