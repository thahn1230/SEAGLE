#!/usr/bin/env python
"""B2 acceptance matrix (Phase 9-11): target-only / draft-only / both quantized,
plus the first-vs-recurrent projection quantization ablation.

Groups (one 7B target build each; run groups on separate GPUs):
  stock       target FP16 (no rotation)
      Q00_targetFP16_draftFP16_stock
  quant       target fake-W4A4 (SpinQuant full rotation)
      Q10_targetW4A4_draftFP16_archA        (A-explicit runtime bridge)
      Q10s_targetW4A4_draftFP16_B2split     (arch-B split, fp16 draft)
      Q11_targetW4A4_draftW4A4_B2split      (both: proj+AR fake-W4A4, R2/R4)
  quant_w4a16 target fake-W4A16
      TQ1_targetW4A16_draftFP16_archA
  quant_kv4   target fake-W4A4KV4 (real KV4 fake-quant; r3 auto-added)
      TQ4_targetW4A4KV4_draftFP16_archA
  rot         target rotated FP16 (function-preserving)
      FP02_B2split_draftFP16               (draft-quant baseline)
      DQ_first_only_W4A4                   (projection_first fake-W4A4 only)
      DQ_recurrent_only_W4A4               (projection_recurrent fake-W4A4 only)
      DQ_both_proj_W4A4
      DQ_ar_only_W4A4                      (AR head fake-W4A4 + R2/R4)
      Q01_targetFP16rot_draftW4A4_full     (proj both + AR fake-W4A4)
      DQ_first_only_W4A16 / DQ_recurrent_only_W4A16 (weight-only)

Greedy, MT-bench prompts, per-prompt acceptance rows + dispatch/coverage proof.
Draft embed + draft head stay fp16 ISOLATED copies in every draft-quant config
(the target's own verification weights are only quantized in `quant*` groups).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_<ts> \
      --group rot --device cuda:1 --num-prompts 20 --max-new-tokens 64
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study, fake_w4a4_draft as fq)
from eagle_spinquant.b2_projection import B2SplitDraftAdapter  # noqa: E402
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402

GROUP_TARGET = {           # group -> (rotation, quant)
    "stock": ("none", "none"),
    "quant": ("full", "w4a4"),
    "quant_w4a16": ("full", "w4a16"),
    "quant_kv4": ("full", "w4a4kv4"),
    "rot": ("full", "none"),
}

# config -> (group, adapter_kind, kwargs, target_precision, draft_precision)
CONFIGS = {
    "Q00_targetFP16_draftFP16_stock": ("stock", "none", {}, "FP16", "FP16"),
    "Q10_targetW4A4_draftFP16_archA": ("quant", "A", {}, "fake_W4A4", "FP16"),
    "Q10s_targetW4A4_draftFP16_B2split": ("quant", "B2", dict(arch="B"),
                                          "fake_W4A4", "FP16"),
    "Q11_targetW4A4_draftW4A4_B2split": ("quant", "B2", dict(
        arch="B", quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True), "fake_W4A4", "fake_W4A4"),
    "TQ1_targetW4A16_draftFP16_archA": ("quant_w4a16", "A", {}, "fake_W4A16", "FP16"),
    "TQ4_targetW4A4KV4_draftFP16_archA": ("quant_kv4", "A", {}, "fake_W4A4KV4", "FP16"),
    "FP02_B2split_draftFP16": ("rot", "B2", dict(arch="B"), "FP16_rotated", "FP16"),
    "DQ_first_only_W4A4": ("rot", "B2", dict(arch="B", quant_first="fake_w4a4"),
                           "FP16_rotated", "fake_W4A4(first proj only)"),
    "DQ_recurrent_only_W4A4": ("rot", "B2", dict(arch="B",
                               quant_recurrent="fake_w4a4"),
                               "FP16_rotated", "fake_W4A4(recurrent proj only)"),
    "DQ_both_proj_W4A4": ("rot", "B2", dict(arch="B", quant_first="fake_w4a4",
                          quant_recurrent="fake_w4a4"),
                          "FP16_rotated", "fake_W4A4(both projections)"),
    "DQ_ar_only_W4A4": ("rot", "B2", dict(arch="B", quant_ar="fake_w4a4",
                        ar_r2r4=True), "FP16_rotated", "fake_W4A4(AR head only)"),
    "Q01_targetFP16rot_draftW4A4_full": ("rot", "B2", dict(
        arch="B", quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True), "FP16_rotated", "fake_W4A4(full draft)"),
    "DQ_first_only_W4A16": ("rot", "B2", dict(arch="B", quant_first="fake_w4a16"),
                            "FP16_rotated", "fake_W4A16(first proj only)"),
    "DQ_recurrent_only_W4A16": ("rot", "B2", dict(arch="B",
                                quant_recurrent="fake_w4a16"),
                                "FP16_rotated", "fake_W4A16(recurrent proj only)"),
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


def coverage_check(adapter, kwargs):
    """Prove quantized modules actually fake-quantized during generation."""
    rows = []
    if not isinstance(adapter, B2SplitDraftAdapter):
        return rows, True
    ok = True
    for sel, mode in (("projection_first", adapter.quant_first),
                      ("projection_recurrent", adapter.quant_recurrent)):
        m = getattr(adapter.split, sel)
        if mode != "fp16":
            good = isinstance(m, fq.FakeW4A4Linear) and m.n_forward > 0 and \
                m.n_weight_quant > 0 and (m.aq is None or m.n_act_quant > 0)
            ok &= good
            rows.append(dict(module=sel, mode=mode, n_forward=m.n_forward,
                             covered=good))
    for parent, attr, _orig in getattr(adapter, "_replaced_ar", []):
        m = getattr(parent, attr)
        good = m.n_forward > 0
        ok &= good
        rows.append(dict(module=f"ar.{attr}", mode=adapter.quant_ar,
                         n_forward=m.n_forward, covered=good))
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--group", required=True, choices=list(GROUP_TARGET))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = args.device
    rd = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(os.path.join(rd, "shards"), exist_ok=True)
    with open(os.path.join(rd, "commands.sh"), "a") as f:
        f.write("CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 "
                + " ".join(sys.argv) + "\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    rotation, quant = GROUP_TARGET[args.group]
    print(f"[b2mx] building target {rotation}/{quant} on {dev} ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rotation, "random_hadamard", quant, 0, device=dev, rotations_root=rr)
    if args.group == "stock":
        R = torch.load(study.r_bin_path("random_hadamard", 0,
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

    print("[b2mx] naive reference ...", flush=True)
    naive_ref = []
    for ids in ids_list:
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_ref.append(t)

    rows, cov_all, disp_all = [], [], []
    for name, (grp, kind, kwargs, tprec, dprec) in CONFIGS.items():
        if grp != args.group:
            continue
        if kind == "none":
            adapter = None
        elif kind == "A":
            adapter = UnrotateAdapter(model, stash, dev, torch.float16,
                                      with_gamma=True)
        else:
            adapter = B2SplitDraftAdapter(model, stash, dev, torch.float16,
                                          **kwargs)
        if adapter:
            adapter.install()
        accs = []
        for pi, ids in enumerate(ids_list):
            if adapter is not None and hasattr(adapter, "set_context"):
                adapter.set_context(prompts[pi]["question_id"])
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            ar = naive_ref[pi]
            n = min(len(ar), len(eg))
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am)
            rows.append(dict(
                config=name, group=grp, target_precision=tprec,
                draft_precision=dprec,
                prompt_id=prompts[pi]["question_id"],
                mean_acceptance=round(am, 4),
                acceptance_list=json.dumps(deltas),
                exact_match=bool(ar[:n] == eg[:n] and len(ar) == len(eg)),
                n_new_tokens=len(eg), n_cycles=len(deltas)))
        cov, cov_ok = coverage_check(adapter, kwargs) if adapter else ([], True)
        for c in cov:
            cov_all.append(dict(config=name, **c))
        if adapter and isinstance(adapter, B2SplitDraftAdapter):
            disp_all.append(dict(config=name, **{
                k: v for k, v in adapter.dispatch_summary().items()
                if k != "quant"}))
        if adapter:
            adapter.uninstall()
        print(f"[b2mx] {name}: accept={np.mean(accs):.4f} cov_ok={cov_ok}",
              flush=True)
        assert cov_ok, f"{name}: quantized modules not covered!"

    tag = args.group
    logging_utils.write_csv(os.path.join(rd, "shards", f"accept__{tag}.csv"), rows)
    if cov_all:
        logging_utils.write_csv(os.path.join(rd, "shards",
                                             f"coverage__{tag}.csv"), cov_all)
    if disp_all:
        logging_utils.write_csv(os.path.join(rd, "shards",
                                             f"dispatch__{tag}.csv"), disp_all)
    print(f"[b2mx] group {tag} DONE -> {rd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
