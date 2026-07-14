#!/usr/bin/env python
"""Fake W4A4 DRAFT acceptance pass. B4 = target fake W4A4 + draft fake W4A4 +
pure-R1 draft. Measures accepted length; PROVES the draft is fake-W4A4 in the
generation path (coverage + forward-hook trace), fails B4 if not.

Groups (2 target builds; run on separate GPUs):
  --group quant : target full/w4a4 -> B1, B2, B3, B4
  --group fp16  : target none/none -> B0, B5

Greedy. Not exact-token equality — main metric is acceptance length.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_fake_w4a4_draft_acceptance.py \
      --run-dir runs/fake_w4a4_draft_<ts> --group quant --configs B4 \
      --num-prompts 2 --max-new-tokens 32
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils, study,  # noqa: E402
                             w4a4_impl_fix as wf, spinquant_draft as spd,
                             fake_w4a4_draft as fq)
DEV = "cuda:0"

# config -> (group, needs_tail, draft_kind, fq_flags)
# fq_flags = dict(r1_only, quant_weight, quant_act) for pure_r1_w4a4 configs
CONFIGS = {
    "B0": ("fp16",  False, "original", {}),
    "B1": ("quant", True,  "original", {}),
    "B2": ("quant", True,  "prev_failed", {}),
    "B3": ("quant", True,  "pure_r1_fp16", {}),
    "B4": ("quant", True,  "pure_r1_w4a4", dict(r1_only=False, quant_weight=True, quant_act=True)),
    "B5": ("fp16",  False, "pure_r1_w4a4", dict(r1_only=False, quant_weight=True, quant_act=True)),
    # isolation ablations (localize any collapse)
    "B4w":  ("quant", True, "pure_r1_w4a4", dict(r1_only=False, quant_weight=True,  quant_act=False)),
    "B4a":  ("quant", True, "pure_r1_w4a4", dict(r1_only=False, quant_weight=False, quant_act=True)),
    "B4r1": ("quant", True, "pure_r1_w4a4", dict(r1_only=True,  quant_weight=True,  quant_act=True)),
    "Borig": ("fp16", False, "orig_w4a4", dict(quant_weight=True, quant_act=True)),
    "Borigw": ("fp16", False, "orig_w4a4", dict(quant_weight=True, quant_act=False)),
}


@torch.no_grad()
def run_gen(gen, ilen, mx):
    final, deltas, prev = None, [], ilen
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        final = out
        if cur - ilen >= mx:
            break
    return final[0, ilen:ilen + mx].tolist(), deltas


def make_draft(kind, model, stash, fq_flags):
    if kind == "original":
        return None
    if kind == "prev_failed":
        return wf.DraftProjTransformAdapter(model, stash, DEV, torch.float16,
                 first_embed="R1T", first_hidden="I", rec_embed="R1T",
                 rec_hidden="R1T", head_mode="fused")
    if kind == "pure_r1_fp16":
        return spd.SpinquantDraftPureR1Adapter(model, stash, DEV, torch.float16, trace=True)
    if kind == "pure_r1_w4a4":
        a = fq.FakeW4A4DraftAdapter(model, stash, DEV, torch.float16, trace=True)
        return a.configure_fq(**fq_flags) if fq_flags else a
    if kind == "orig_w4a4":
        a = fq.FakeW4A4OrigDraftAdapter(model, stash, DEV, torch.float16)
        return a.configure_fq(**fq_flags) if fq_flags else a
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--group", required=True, choices=["quant", "fp16"])
    ap.add_argument("--configs", required=True)
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + " " + " ".join(sys.argv) + "\n")
    if not os.path.isfile(os.path.join(run_dir, "environment.txt")):
        with open(os.path.join(run_dir, "environment.txt"), "w") as f:
            f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        assert CONFIGS[c][0] == args.group, f"{c} is not in group {args.group}"

    if args.group == "quant":
        print("[fq] building target full/w4a4...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "full", "random_hadamard", "w4a4", 0, device=DEV, rotations_root=rr)
    else:
        print("[fq] building target none/fp16...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "none", "random_hadamard", "none", 0, device=DEV, rotations_root=rr)
        # inject R1 (from r_bin) + gamma_f + lm_head for the pure-R1 draft conjugation
        R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach().float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight.detach().float().cpu().clone()
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, DEV)
    mp = f"{cfg['model']['target']}+{cfg['model']['eagle_draft']}"

    base_rows, prompt_rows, tok_rows, samples = [], [], [], []
    cov_rows, hook_rows, rot_rows, fail = [], [], [], []
    for name in configs:
        _, needs_tail, kind, fq_flags = CONFIGS[name]
        tail = wf.UnfusedTailAdapter(model, stash).install() if needs_tail else None
        draft = make_draft(kind, model, stash, fq_flags)
        if draft:
            draft.install()
        timers = study.PhaseTimers(model).install()
        accs = []
        for p in prompts:
            ids = build_prompt(tok, p["text"]).to(DEV)
            if draft is not None and hasattr(draft, "set_context"):
                draft.set_context(p["question_id"])
            ilen = ids.shape[1]
            ar, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                     max_steps=args.max_new_tokens + 4, tree_choices=tree), ilen, args.max_new_tokens)
            eg, deltas = run_gen(model.ea_generate(ids, temperature=0.0,
                     max_steps=args.max_new_tokens + 8, tree_choices=tree), ilen, args.max_new_tokens)
            n = min(len(ar), len(eg))
            mism = next((i for i in range(n) if ar[i] != eg[i]), -1)
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am)
            prompt_rows.append(dict(config_name=name, prompt_id=p["question_id"],
                mean_acceptance=round(am, 4), acceptance_list=str(deltas),
                num_generated_tokens=len(eg),
                exact_output_match=bool(ar[:n] == eg[:n] and len(ar) == len(eg)),
                first_mismatch_position=mism, notes=kind))
            tok_rows.append(dict(config_name=name, prompt_id=p["question_id"],
                                 ar_tokens=ar, eagle_tokens=eg, acceptance_list=deltas))
            samples.append(dict(config_name=name, prompt_id=p["question_id"],
                eagle_text=tok.decode(eg, skip_special_tokens=True)[:200]))
        # coverage / traces for the fake-w4a4 draft
        cov_pass = True
        if isinstance(draft, (fq.FakeW4A4DraftAdapter, fq.FakeW4A4OrigDraftAdapter)):
            cov = draft.coverage()
            for r in cov:
                r["config_name"] = name
            cov_rows.extend(cov)
            for m in draft.fq_modules.values():
                hook_rows.extend([dict(config_name=name, **t) for t in m.trace])
            cov_pass = draft.all_required_covered()
            for r in cov:
                rot_rows.append(dict(module_name=r["module_name"],
                    r1_status="applied" if r["r1_applied"] else "no",
                    r2_status="applied" if r["r2_applied"] else "n/a",
                    r3_status="N/A_w4a4_k16", r4_status="applied" if r["r4_applied"] else "n/a",
                    generation_path_applied=True, unit_test_only=False,
                    notes="R2/R4 in weights; R4 online Hadamard; R3 N/A (KV fp16)"))
            if not cov_pass:
                fail.append(f"{name}: draft fake-W4A4 coverage FAILED")
        acc_arr = np.array(accs, float)
        base_rows.append(dict(config_name=name,
            quant_target=("w4a4" if args.group == "quant" else "fp16"),
            quant_draft=("fake_w4a4" if kind == "pure_r1_w4a4" else "fp16"),
            draft_basis=("pure_r1" if "pure_r1" in kind else ("prev_failed" if kind == "prev_failed" else "original")),
            n_prompts=len(accs), max_new_tokens=args.max_new_tokens,
            mean_acceptance=round(acc_arr.mean(), 4), median_acceptance=round(float(np.median(acc_arr)), 4),
            p10_acceptance=round(float(np.percentile(acc_arr, 10)), 4),
            p90_acceptance=round(float(np.percentile(acc_arr, 90)), 4),
            exact_output_match_rate=round(np.mean([r["exact_output_match"] for r in prompt_rows if r["config_name"] == name]), 4),
            target_ppl_if_available="", draft_fake_w4a4_coverage_pass=cov_pass, notes=kind))
        print(f"[fq] {name} accept={acc_arr.mean():.3f} cov_pass={cov_pass} ({kind})", flush=True)
        timers.uninstall()
        if draft:
            draft.uninstall()
        if tail:
            tail.uninstall()

    tag = f"{args.group}_{'-'.join(configs)}"
    logging_utils.write_csv(os.path.join(run_dir, "shards", f"baseline__{tag}.csv"), base_rows)
    logging_utils.write_csv(os.path.join(run_dir, "shards", f"prompt__{tag}.csv"), prompt_rows)
    if cov_rows:
        logging_utils.write_csv(os.path.join(run_dir, "draft_fake_w4a4_coverage.csv"), cov_rows)
        with open(os.path.join(run_dir, "draft_fake_w4a4_coverage.json"), "w") as f:
            json.dump(cov_rows, f, indent=2)
    if hook_rows:
        logging_utils.write_csv(os.path.join(run_dir, "draft_fake_w4a4_forward_hook_trace.csv"), hook_rows)
    if rot_rows:
        logging_utils.write_csv(os.path.join(run_dir, "draft_rotation_application_trace.csv"), rot_rows)
    with open(os.path.join(run_dir, "shards", f"tokens__{tag}.jsonl"), "w") as f:
        for r in tok_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "shards", f"samples__{tag}.jsonl"), "w") as f:
        for r in samples:
            f.write(json.dumps(r) + "\n")
    if fail:
        with open(os.path.join(run_dir, "failure_report.md"), "w") as f:
            f.write("# Failure report\n\n" + "\n".join(f"- {x}" for x in fail) + "\n")
        print("[fq] FAILURE:", fail, flush=True)
    print(f"[fq] wrote shards for {tag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
