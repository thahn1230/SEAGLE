#!/usr/bin/env python
"""Concat-selective acceptance matrix + prev-B2 architecture comparison.

Groups (one target build each):
  stock        Q00_targetFP16_draftFP16_stock
  quant (fake-W4A4 target):
      Q10_targetW4A4_draftFP16_archA          (A-explicit bridge)
      Q11_new_targetW4A4_draftW4A4_concat     (concat-selective full draft W4A4)
      Q11_prevB2_targetW4A4_draftW4A4         (previous rotated-embedding B2)
  rot (rotated fp16 target):
      CS_fp16_baseline                        (concat-selective folded, fp16)
      DQ_first_only_W4A4 / DQ_recurrent_only_W4A4 / DQ_both_proj_W4A4
      DQ_ar_only_W4A4
      Q01_new_draft_full_W4A4                 (proj both + AR fake-W4A4)
      Q01_prevB2_draft_full_W4A4              (previous B2, same quant policy)
      DQ_first_only_W4A16 / DQ_recurrent_only_W4A16
      DQ_embed_only_W4A16                     (embedding table weight-quant ablation)

The primary Q01/Q11 keep the draft embedding FP16 and ORIGINAL-basis.
Coverage counters prove every fake-quant module fired in generation.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_concat_selective_acceptance_matrix.py \
      --run-dir runs/cs_matrix_<ts> --group rot --device cuda:1 \
      --num-prompts 20 --max-new-tokens 64
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
from eagle_spinquant.concat_selective_projection import (  # noqa: E402
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402

GROUP_TARGET = {"stock": ("none", "none"), "quant": ("full", "w4a4"),
                "rot": ("full", "none")}

CS = "concat_selective"
CONFIGS = {
    "Q00_targetFP16_draftFP16_stock": ("stock", "none", {}, "FP16", "FP16"),
    "Q10_targetW4A4_draftFP16_archA": ("quant", "A", {}, "fake_W4A4", "FP16"),
    "Q11_new_targetW4A4_draftW4A4_concat": ("quant", CS, dict(
        quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True), "fake_W4A4", "fake_W4A4(concat-selective)"),
    "Q11_prevB2_targetW4A4_draftW4A4": ("quant", "B2", dict(
        arch="B", quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True), "fake_W4A4", "fake_W4A4(prev B2)"),
    "CS_fp16_baseline": ("rot", CS, {}, "FP16_rotated", "FP16"),
    "DQ_first_only_W4A4": ("rot", CS, dict(quant_first="fake_w4a4"),
                           "FP16_rotated", "fake_W4A4(first preR only)"),
    "DQ_recurrent_only_W4A4": ("rot", CS, dict(quant_recurrent="fake_w4a4"),
                               "FP16_rotated", "fake_W4A4(recurrent preR only)"),
    "DQ_both_proj_W4A4": ("rot", CS, dict(quant_first="fake_w4a4",
                                          quant_recurrent="fake_w4a4"),
                          "FP16_rotated", "fake_W4A4(both projections)"),
    "DQ_ar_only_W4A4": ("rot", CS, dict(quant_ar="fake_w4a4", ar_r2r4=True),
                        "FP16_rotated", "fake_W4A4(AR head only)"),
    "Q01_new_draft_full_W4A4": ("rot", CS, dict(
        quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True),
        "FP16_rotated", "fake_W4A4(full, concat-selective)"),
    "Q01_prevB2_draft_full_W4A4": ("rot", "B2", dict(
        arch="B", quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True),
        "FP16_rotated", "fake_W4A4(full, prev B2)"),
    "DQ_first_only_W4A16": ("rot", CS, dict(quant_first="fake_w4a16"),
                            "FP16_rotated", "fake_W4A16(first preR only)"),
    "DQ_recurrent_only_W4A16": ("rot", CS, dict(quant_recurrent="fake_w4a16"),
                                "FP16_rotated", "fake_W4A16(recurrent preR only)"),
    "DQ_embed_only_W4A16": ("rot", CS, dict(quant_embed="fake_w4a16"),
                            "FP16_rotated", "fake_W4A16(embedding table only)"),
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


def coverage_check(adapter):
    rows, ok = [], True
    split = getattr(adapter, "split", None)
    if split is None:
        return rows, True
    for sel in ("projection_first_preR", "projection_recurrent_preR"):
        m = getattr(split, sel, None)
        if isinstance(m, fq.FakeW4A4Linear):
            good = m.n_forward > 0 and m.n_weight_quant > 0 and \
                (m.aq is None or m.n_act_quant > 0)
            ok &= good
            rows.append(dict(module=sel, n_forward=m.n_forward, covered=good))
    for parent, attr, _o in getattr(adapter, "_replaced_ar", []):
        m = getattr(parent, attr)
        ok &= m.n_forward > 0
        rows.append(dict(module=f"ar.{attr}", n_forward=m.n_forward,
                         covered=m.n_forward > 0))
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--group", required=True, choices=list(GROUP_TARGET))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
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

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    rotation, quant = GROUP_TARGET[args.group]
    print(f"[csmx] target {rotation}/{quant} on {dev} ...", flush=True)
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
    print("[csmx] naive reference ...", flush=True)
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
        elif kind == "B2":
            adapter = B2SplitDraftAdapter(model, stash, dev, torch.float16,
                                          **kwargs)
        else:
            adapter = ConcatSelectiveDraftAdapter(model, stash, dev,
                                                  torch.float16,
                                                  variant="folded", **kwargs)
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
            rows.append(dict(config=name, group=grp, target_precision=tprec,
                             draft_precision=dprec,
                             prompt_id=prompts[pi]["question_id"],
                             mean_acceptance=round(am, 4),
                             acceptance_list=json.dumps(deltas),
                             exact_match=bool(ar[:n] == eg[:n]
                                              and len(ar) == len(eg)),
                             n_new_tokens=len(eg), n_cycles=len(deltas)))
        cov, cov_ok = coverage_check(adapter) if adapter else ([], True)
        for c in cov:
            cov_all.append(dict(config=name, **c))
        if adapter and hasattr(adapter, "dispatch_summary"):
            ds = adapter.dispatch_summary()
            disp_all.append(dict(config=name, **{k: v for k, v in ds.items()
                                                 if k != "quant"}))
        if adapter:
            adapter.uninstall()
        print(f"[csmx] {name}: accept={np.mean(accs):.4f} cov_ok={cov_ok}",
              flush=True)
        assert cov_ok, f"{name}: quantized modules not covered"

    tag = args.group
    logging_utils.write_csv(os.path.join(rd, "shards", f"accept__{tag}.csv"), rows)
    if cov_all:
        logging_utils.write_csv(os.path.join(rd, "shards",
                                             f"coverage__{tag}.csv"), cov_all)
    if disp_all:
        logging_utils.write_csv(os.path.join(rd, "shards",
                                             f"dispatch__{tag}.csv"), disp_all)
    print(f"[csmx] group {tag} DONE -> {rd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
