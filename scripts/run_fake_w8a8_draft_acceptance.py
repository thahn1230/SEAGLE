#!/usr/bin/env python
"""Fake W8A8 DRAFT acceptance pass. C4 = target fake W4A4 + pure-R1 draft fake
W8A8. Same machinery as the W4A4 pass, at 8-bit. Proves the draft is fake-W8A8
in the generation path (coverage + hook trace); fails C4 if not.

Groups (2 target builds; run on separate GPUs):
  --group quant : target full/w4a4 -> C1, C2, C3, C4, C6
  --group fp16  : target none/none -> C0, C5

Greedy. Main metric = acceptance length (NOT exact token equality).

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_fake_w8a8_draft_acceptance.py \
      --run-dir runs/fake_w8a8_draft_<ts> --group quant --configs C4 \
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
                             fake_w4a4_draft as f4, fake_w8a8_draft as f8)
DEV = "cuda:0"

# config -> (group, needs_tail, draft_kind, fq_flags)
CONFIGS = {
    "C0": ("fp16",  False, "original", {}),
    "C1": ("quant", True,  "original", {}),
    "C2": ("quant", True,  "pure_r1_fp16", {}),
    "C3": ("quant", True,  "pure_r1_w4a4", dict(w_bits=4, a_bits=4)),
    "C4": ("quant", True,  "pure_r1_w8a8", dict(w_bits=8, a_bits=8)),
    "C5": ("fp16",  False, "pure_r1_w8a8", dict(w_bits=8, a_bits=8)),
    "C6": ("quant", True,  "orig_w8a8", dict(w_bits=8, a_bits=8)),
}
FQ_KINDS = ("pure_r1_w4a4", "pure_r1_w8a8", "orig_w8a8")


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
    if kind == "pure_r1_fp16":
        return spd.SpinquantDraftPureR1Adapter(model, stash, DEV, torch.float16, trace=True)
    if kind == "pure_r1_w4a4":
        return f4.FakeW4A4DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(**fq_flags)
    if kind == "pure_r1_w8a8":
        return f8.FakeW8A8DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(**fq_flags)
    if kind == "orig_w8a8":
        return f8.FakeW8A8OrigDraftAdapter(model, stash, DEV, torch.float16).configure_fq(**fq_flags)
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
        assert CONFIGS[c][0] == args.group, f"{c} not in group {args.group}"

    if args.group == "quant":
        print("[w8] building target full/w4a4...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "full", "random_hadamard", "w4a4", 0, device=DEV, rotations_root=rr)
    else:
        print("[w8] building target none/fp16...", flush=True)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            "none", "random_hadamard", "none", 0, device=DEV, rotations_root=rr)
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
            tok_rows.append(dict(config_name=name, prompt_id=p["question_id"], ar_tokens=ar, eagle_tokens=eg, acceptance_list=deltas))
            samples.append(dict(config_name=name, prompt_id=p["question_id"], eagle_text=tok.decode(eg, skip_special_tokens=True)[:200]))
        cov_pass = True
        if kind in FQ_KINDS:
            cov = draft.coverage()
            for r in cov:
                r["config_name"] = name
            cov_rows.extend(cov)
            for m in draft.fq_modules.values():
                hook_rows.extend([dict(config_name=name, **t) for t in m.trace])
            bits_ok = all(r["weight_bits"] == fq_flags.get("w_bits") and r["activation_bits"] == fq_flags.get("a_bits") for r in cov)
            cov_pass = draft.all_required_covered() and bits_ok
            for r in cov:
                rot_rows.append(dict(module_name=r["module_name"],
                    r1_status="applied" if r["r1_applied"] else "no(orig-ctrl)",
                    r2_status="applied" if r["r2_applied"] else ("n/a" if r["r1_applied"] else "no(orig-ctrl)"),
                    r3_status="N/A_w8a8_k16", r4_status="applied" if r["r4_applied"] else ("n/a" if r["r1_applied"] else "no(orig-ctrl)"),
                    generation_path_applied=True, unit_test_only=False,
                    notes="R2/R4 in weights; R4 online Hadamard; R3 N/A (KV fp16)"))
            if not cov_pass:
                fail.append(f"{name}: draft fake-W8A8 coverage FAILED (covered={draft.all_required_covered()}, bits_ok={bits_ok})")
        acc_arr = np.array(accs, float)
        base_rows.append(dict(config_name=name,
            quant_target=("w4a4" if args.group == "quant" else "fp16"),
            quant_draft=("fake_w8a8" if "w8a8" in kind else ("fake_w4a4" if "w4a4" in kind else "fp16")),
            draft_basis=("pure_r1" if "pure_r1" in kind else ("original" if kind in ("original", "orig_w8a8") else kind)),
            n_prompts=len(accs), max_new_tokens=args.max_new_tokens,
            mean_acceptance=round(acc_arr.mean(), 4), median_acceptance=round(float(np.median(acc_arr)), 4),
            p10_acceptance=round(float(np.percentile(acc_arr, 10)), 4),
            p90_acceptance=round(float(np.percentile(acc_arr, 90)), 4),
            exact_output_match_rate=round(np.mean([r["exact_output_match"] for r in prompt_rows if r["config_name"] == name]), 4),
            draft_fake_w8a8_coverage_pass=cov_pass, notes=kind))
        print(f"[w8] {name} accept={acc_arr.mean():.3f} cov_pass={cov_pass} ({kind})", flush=True)
        timers.uninstall()
        if draft:
            draft.uninstall()
        if tail:
            tail.uninstall()

    tag = f"{args.group}_{'-'.join(configs)}"
    logging_utils.write_csv(os.path.join(run_dir, "shards", f"baseline__{tag}.csv"), base_rows)
    logging_utils.write_csv(os.path.join(run_dir, "shards", f"prompt__{tag}.csv"), prompt_rows)
    if cov_rows:
        logging_utils.write_csv(os.path.join(run_dir, "draft_fake_w8a8_coverage.csv"), cov_rows)
        with open(os.path.join(run_dir, "draft_fake_w8a8_coverage.json"), "w") as f:
            json.dump(cov_rows, f, indent=2)
    if hook_rows:
        logging_utils.write_csv(os.path.join(run_dir, "draft_fake_w8a8_forward_hook_trace.csv"), hook_rows)
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
        print("[w8] FAILURE:", fail, flush=True)
    print(f"[w8] wrote shards for {tag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
