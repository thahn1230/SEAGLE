#!/usr/bin/env python
"""Studies B/C runner: iterate named draft-adapter configurations against a
fixed target (fp16 stock or learned-rotation W4A4), measuring EAGLE AL with
the exact Study-A evaluator contract (80 MT-bench prompts, greedy, 128
tokens, mc_sim_7b_63) plus mandatory isolation checks (Gate D) per config.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<g> \
    python scripts/run_eagle_component_grid.py --config configs/X.yaml \
      --target {fp16,w4a4} --run-dir runs/<id> [--configs name1,name2]

YAML schema:
  study: <name>
  configs:
    <config_name>:
      kwargs: {quant_first: fake_w4a4, ...}     # ConcatSelectiveDraftAdapter
      branch_act: [4, 4]                        # optional
      embed_scale_alpha: auto|<float>           # optional ('auto' -> alpha.json)
"""
import argparse, hashlib, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import numpy as np
import yaml
from eagle_spinquant import eagle_bridge, experiment, logging_utils, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)

ROTATION_KIND = "learned_chat_w4a4kv16"


def sha16(t):
    return hashlib.sha256(
        t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        final = out
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist(), deltas


@torch.no_grad()
def target_isolation_snapshot(model, probe, dev):
    bm = model.base_model
    bm.model.tree_mask = None
    logits = bm(probe.to(dev)).logits[:, -8:, :].float().cpu()
    return dict(
        embed_ptr=bm.model.embed_tokens.weight.data_ptr(),
        head_ptr=bm.lm_head.weight.data_ptr(),
        embed_sha=sha16(bm.model.embed_tokens.weight),
        head_sha=sha16(bm.lm_head.weight),
        embed_id=id(bm.model.embed_tokens), head_id=id(bm.lm_head),
        logits=logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True, choices=["fp16", "w4a4"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of config names")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--tag-prefix", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cvd in tuple(str(i) for i in range(6)), \
        f"GPUs 0-5 only for this study (CVD={cvd})"
    dev = args.device
    spec = yaml.safe_load(open(args.config))
    names = list(spec["configs"])
    if args.configs:
        names = [n for n in names if n in set(args.configs.split(","))]
    rd = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(os.path.join(rd, "shards"), exist_ok=True)
    with open(os.path.join(rd, "commands.sh"), "a") as f:
        f.write(f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={cvd} "
                + " ".join(sys.argv) + "\n")

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    if args.target == "fp16":
        rot, quant, fhm = "none", "none", "identity"
    else:
        rot, quant, fhm = "full", "w4a4", "gamma_R1"
    print(f"[grid] target={args.target} rot={rot} quant={quant} fhm={fhm}",
          flush=True)
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, ROTATION_KIND, quant, 0, device=dev, rotations_root=rr)
    if rot == "none":
        R = torch.load(study.r_bin_path(ROTATION_KIND, 0,
                                        paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight.detach() \
            .float().cpu().clone()
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]

    # naive target reference (output-preservation + quality diagnostics)
    naive_path = os.path.join(rd, "shards", f"naive__{args.target}.pt")
    if os.path.exists(naive_path):
        naive = torch.load(naive_path, weights_only=True)
        naive = [t for t in naive]
        print(f"[grid] naive reference loaded ({len(naive)})", flush=True)
    else:
        naive = []
        for ids in ids_list:
            t, _ = run_gen(model.naive_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 4,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            naive.append(t)
        torch.save([torch.tensor(t) for t in naive], naive_path)
        print("[grid] naive reference computed", flush=True)
        naive = [torch.tensor(t).tolist() for t in naive]
    naive = [list(map(int, (t.tolist() if torch.is_tensor(t) else t)))
             for t in naive]

    probe = ids_list[0][:, :64]
    alpha_file = os.path.join(rd, "alpha_selected.json")

    for name in names:
        c = spec["configs"][name]
        kwargs = dict(c.get("kwargs") or {})
        if c.get("branch_act"):
            kwargs["branch_act"] = tuple(c["branch_act"])
        if c.get("embed_scale_alpha") is not None:
            a = c["embed_scale_alpha"]
            if a == "auto":
                a = json.load(open(alpha_file))[args.target]["alpha"]
            kwargs["embed_scale_alpha"] = float(a)
        kwargs.setdefault("first_hidden_mode", fhm)
        tag = f"{args.tag_prefix}{args.target}__{name}"
        out_csv = os.path.join(rd, "shards", f"grid__{tag}.csv")
        if os.path.exists(out_csv):
            print(f"[grid] {tag}: exists, skip", flush=True)
            continue
        pre = target_isolation_snapshot(model, probe, dev)
        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            trace=False, **kwargs)
        ad.install()
        post = target_isolation_snapshot(model, probe, dev)
        iso_ok = (pre["embed_ptr"] == post["embed_ptr"]
                  and pre["head_ptr"] == post["head_ptr"]
                  and pre["embed_sha"] == post["embed_sha"]
                  and pre["head_sha"] == post["head_sha"]
                  and pre["embed_id"] == post["embed_id"]
                  and pre["head_id"] == post["head_id"]
                  and torch.equal(pre["logits"], post["logits"]))
        if not iso_ok:
            ad.uninstall()
            raise RuntimeError(f"GATE D FAIL: target mutated by {tag}")
        rows = []
        t0 = time.time()
        for pi, ids in enumerate(ids_list):
            if hasattr(ad, "set_context"):
                ad.set_context(prompts[pi]["question_id"])
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            ar = naive[pi]
            n = min(len(ar), len(eg))
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            rows.append(dict(
                config=name, target=args.target,
                prompt_id=prompts[pi]["question_id"],
                mean_acceptance=round(am, 4),
                acceptance_list=json.dumps(deltas),
                n_new_tokens=len(eg), n_cycles=len(deltas), tree_size=26,
                exact_match=bool(ar[:n] == eg[:n] and len(ar) == len(eg)),
                iso_ok=True))
        ad.uninstall()
        meta = dict(tag=tag, kwargs={k: str(v) for k, v in kwargs.items()},
                    quant_meta=getattr(ad, "meta_quant", {}),
                    isolation=dict(ok=True,
                                   embed_sha=post["embed_sha"],
                                   head_sha=post["head_sha"]),
                    wall_s=round(time.time() - t0, 1))
        with open(os.path.join(rd, "shards", f"grid__{tag}.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logging_utils.write_csv(out_csv, rows)
        al = float(np.mean([r["mean_acceptance"] for r in rows]))
        em = float(np.mean([r["exact_match"] for r in rows]))
        print(f"[grid] {tag}: AL={al:.4f} exact_match={em:.3f} "
              f"({meta['wall_s']}s)", flush=True)
    print(f"[grid] target {args.target} DONE -> {rd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
