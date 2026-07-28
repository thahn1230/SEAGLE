#!/usr/bin/env python
"""P3 alpha calibration for the official from-scratch study (spec §14).

Conditions (each calibrated independently, held-out calib pool only):
  A  fp16 target + original-basis fresh draft   (identity D4P3)
  B  w4a4 target + restored original-basis draft (restored has NO alpha:
     the draft is FP16 — recorded as n/a)
  C  w4a4 target + shared rotated-basis draft   (gamma_R1 D4P3)
  D  w4a4 target + local residual R_D           (rot D4P3, optional)

For A/C/D: (1) robust initial estimates from hidden/embedding statistics
(RMS, p99, p99.9 ratios, captured on calib prompts), (2) the
pre-registered 9-point grid evaluated by held-out calib acceptance
length via the universal evaluator (c4 calib pool, offset-500),
(3) selection = argmax calib AL; NMSE/logit-agreement recorded by the
first/recurrent audit (companion capture).

  --stage stats   capture ratio-based estimates (one GPU, ~5 min)
  --stage sweep   launch/collect the 9-point grid for one condition
  --stage select  aggregate all shards -> tables/alpha_calibration.json
"""
import argparse, csv, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

GRID = [8.0, 11.3137085, 16.0, 22.627417, 32.0, 45.254834, 64.0,
        90.509668, 128.0]
COND = {"A": ("fp16", "d4p3_deploy"), "C": ("int4", "d4p3_deploy"),
        "D": ("int4", "rot")}


def stage_stats(args):
    import torch
    from eagle_spinquant import experiment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eagle_spinquant.eval_datasets import load_eval_prompts
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    dev = "cuda:0"
    tgt = AutoModelForCausalLM.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(paths["target_path"],
                                        use_fast=False)
    anchor = torch.load(args.anchor, map_location="cpu",
                        weights_only=False)
    sd = anchor.get("model", anchor.get("draft_state_dict", anchor))
    emb = sd["embed_tokens.weight"].float()
    prompts, _ = load_eval_prompts("c4", 20, "calib")
    hs = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["text"], return_tensors="pt", truncation=True,
                      max_length=1024).input_ids.to(dev)
            hs.append(tgt.model(ids).last_hidden_state[0].float().cpu())
    h = torch.cat(hs)
    e = emb

    def q(t, p):
        x = t.abs().flatten().float()
        if x.numel() > 4_000_000:      # quantile() size limit
            g = torch.Generator().manual_seed(0)
            x = x[torch.randint(0, x.numel(), (4_000_000,), generator=g)]
        return float(x.quantile(p))

    stats = dict(
        hidden=dict(rms=float(h.pow(2).mean().sqrt()),
                    p99=q(h, 0.99), p999=q(h, 0.999),
                    absmax=float(h.abs().max())),
        embedding=dict(rms=float(e.pow(2).mean().sqrt()),
                       p99=q(e, 0.99), p999=q(e, 0.999),
                       absmax=float(e.abs().max())))
    stats["alpha_estimates"] = dict(
        rms_ratio=stats["hidden"]["rms"] / stats["embedding"]["rms"],
        p99_ratio=stats["hidden"]["p99"] / stats["embedding"]["p99"],
        p999_ratio=stats["hidden"]["p999"] / stats["embedding"]["p999"])
    out = os.path.join(args.run_dir, "tables", "alpha_stats.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(stats, open(out, "w"), indent=1)
    print(json.dumps(stats["alpha_estimates"], indent=1))
    return 0


def stage_sweep(args):
    tgt, cfg = COND[args.condition]
    rc = 0
    for a in GRID:
        cmd = [sys.executable,
               os.path.join(ROOT, "scripts", "eval_eagle1_ptq_vs_qat.py"),
               "--cell", "C4" if cfg == "rot" else "C3",
               "--target", tgt, "--run-dir", args.run_dir,
               "--sd", args.anchor, "--alpha", str(a),
               "--datasets", "c4:20", "--pool", "calib",
               "--tag", f"ALCAL_{args.condition}_a{a}"]

        if cfg == "rot":
            cmd += ["--ckpt", args.rd_ckpt]
        rc |= subprocess.run(cmd).returncode
    return rc


def stage_select(args):
    out = {}
    for cond in COND:
        best, rows = None, []
        for p in glob.glob(os.path.join(
                args.run_dir, "shards",
                f"al__ALCAL_{cond}_a*__*__c4__calib.csv")):
            a = float(os.path.basename(p).split("_a")[1].split("__")[0])
            taus = [t for r in csv.DictReader(open(p))
                    for t in json.loads(r["acceptance_list"])]
            tau = sum(taus) / max(len(taus), 1)
            rows.append(dict(alpha=a, tau=round(tau, 4)))
            if best is None or tau > best[1]:
                best = (a, tau)
        if rows:
            out[cond] = dict(grid=sorted(rows, key=lambda r: r["alpha"]),
                             selected_alpha=best[0],
                             calib_tau=round(best[1], 4))
    out["B"] = dict(note="restored interface deploys an FP16 draft; "
                         "no P3 alpha applies")
    path = os.path.join(args.run_dir, "tables",
                        "alpha_calibration.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v.get("selected_alpha") for k, v in out.items()},
                     indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["stats", "sweep", "select"])
    ap.add_argument("--condition", choices=list(COND))
    ap.add_argument("--anchor")
    ap.add_argument("--rd-ckpt", default=None)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    return dict(stats=stage_stats, sweep=stage_sweep,
                select=stage_select)[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
