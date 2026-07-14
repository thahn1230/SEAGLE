#!/usr/bin/env python
"""STOP GATE A: 7B fp16 B2 split-projection equivalence (docs/B2_SPLIT_BASIS_CONTRACT.md).

Configs (all fp16, greedy):
  FP00_stock              stock target + stock EAGLE (reference)
  -- rotated fp16 target (SpinQuant 'full', quant none) --
  FP01_A_explicit         Arch A explicit: h=(a_t@R1ᵀ)*γ runtime, stock draft
  FP01b_A_folded          Arch A folded split (B2SplitDraftAdapter arch=A_folded)
  FP02_B2_split           Arch B split (fully rotated draft, fc_first/fc_recurrent)
  FP02x_explicit_Mgamma   positive control: runtime a_t@M_γ + always-recurrent
  FP03_single_folded      NEGATIVE: projection_first reused for every step
  FP04_gamma_omitted      NEGATIVE: gamma dropped from projection_first
  N1_naive                NEGATIVE: a_t straight into stock draft
  N2_gamma_elementwise    NEGATIVE: a_t*γ in rotated basis
  N4_wrong_orientation    NEGATIVE: h-block folded with R1ᵀ
  N5_fold_both_halves     NEGATIVE: D_γ leaked into embedding block

Gate: correct configs preserve the target's greedy output exactly and match
each other's acceptance; the rotated target's own greedy output equals stock;
negative controls degrade. Writes artifacts/b2_split_study/{fp_equivalence.csv,
fp_equivalence.json, projection_dispatch_trace.jsonl, first_mismatch_dumps/}.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/validate_b2_fp_equivalence.py --device cuda:0 \
      --num-prompts 8 --max-new-tokens 48
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)
from eagle_spinquant.b2_projection import B2SplitDraftAdapter, m_gamma  # noqa: E402
from eagle_spinquant.study import UnrotateAdapter, NaiveAdapter  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")
TOL_ACCEPT = 0.15          # |Δ acceptance| between correct fp16 configs
NEG_DROP = 0.5             # negative controls must lose at least this


def b2_gpu_asserts():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7", \
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} != '6,7'"
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert torch.cuda.device_count() == 2, torch.cuda.device_count()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    b2_gpu_asserts()
    dev = args.device
    os.makedirs(os.path.join(ART, "first_mismatch_dumps"), exist_ok=True)
    with open(os.path.join(ART, "commands.sh"), "a") as f:
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

    rows, dispatch_rows, gate_reasons = [], [], []
    mism_dir = os.path.join(ART, "first_mismatch_dumps")

    def run_config(model, name, adapter, ids_list, naive_ref):
        if adapter:
            adapter.install()
        accs, ok = [], []
        for pi, ids in enumerate(ids_list):
            if adapter is not None and hasattr(adapter, "set_context"):
                adapter.set_context(prompts[pi]["question_id"])
            ilen = ids.shape[1]
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ilen, args.max_new_tokens)
            ar = naive_ref[pi]
            n = min(len(ar), len(eg))
            mism = next((i for i in range(n) if ar[i] != eg[i]), -1)
            exact = bool(ar[:n] == eg[:n] and len(ar) == len(eg))
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am); ok.append(exact)
            rows.append(dict(config=name, prompt_id=prompts[pi]["question_id"],
                             mean_acceptance=round(am, 4), exact_match=exact,
                             first_mismatch=mism, n_new=len(eg)))
            if not exact:
                with open(os.path.join(mism_dir, f"{name}_p{pi}.json"), "w") as f:
                    json.dump(dict(config=name, prompt_id=pi, naive=ar,
                                   eagle=eg, first_mismatch=mism), f)
        if adapter:
            if isinstance(adapter, B2SplitDraftAdapter):
                for t in adapter.trace_rows:
                    dispatch_rows.append(dict(config=name, **t))
                print(f"[gateA] {name} dispatch: {adapter.dispatch_summary()}",
                      flush=True)
            adapter.uninstall()
        res = dict(config=name, mean_acceptance=round(float(np.mean(accs)), 4),
                   exact_match_rate=round(float(np.mean(ok)), 4),
                   n_prompts=len(accs))
        print(f"[gateA] {name}: accept={res['mean_acceptance']} "
              f"exact={res['exact_match_rate']}", flush=True)
        return res

    # ---------------- stage 1: stock target ----------------
    print("[gateA] building STOCK target ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "none", "random_hadamard", "none", 0, device=dev, rotations_root=rr)
    R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    stash["R1"] = R["R1"].clone()
    stash["gamma_f"] = model.base_model.model.norm.weight.detach().float().cpu().clone()
    stash["lm_head_weight"] = model.base_model.lm_head.weight.detach().float().cpu().clone()
    torch.save(stash["gamma_f"], os.path.join(PROJECT_ROOT, "outputs",
                                              "target_final_rms_gamma.pt"))
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]

    naive_stock = []
    for ids in ids_list:
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_stock.append(t)
    agg = [run_config(model, "FP00_stock", None, ids_list, naive_stock)]
    del model; torch.cuda.empty_cache()

    # ---------------- stage 2: rotated fp16 target ----------------
    print("[gateA] building ROTATED fp16 target (full/none) ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "none", 0, device=dev, rotations_root=rr)
    study.set_draft_tree(model, tree, dev)

    naive_rot = []
    rot_match = []
    for pi, ids in enumerate(ids_list):
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_rot.append(t)
        eq = bool(t == naive_stock[pi])
        rot_match.append(eq)
        if not eq:
            with open(os.path.join(mism_dir, f"target_rotation_p{pi}.json"), "w") as f:
                json.dump(dict(prompt=pi, stock=naive_stock[pi], rotated=t), f)
    target_rot_exact = float(np.mean(rot_match))
    print(f"[gateA] rotated-target greedy == stock: {target_rot_exact}", flush=True)

    mk = lambda **kw: B2SplitDraftAdapter(model, stash, dev, torch.float16, **kw)
    configs = [
        ("FP01_A_explicit", UnrotateAdapter(model, stash, dev, torch.float16,
                                            with_gamma=True)),
        ("FP01b_A_folded", mk(arch="A_folded")),
        ("FP02_B2_split", mk(arch="B")),
        ("FP02x_explicit_Mgamma", mk(arch="B", nc="explicit_Mgamma")),
        ("FP03_single_folded", mk(arch="B", nc="single_folded")),
        ("FP04_gamma_omitted", mk(arch="B", nc="gamma_omitted")),
        ("N1_naive", NaiveAdapter(model, stash, dev, torch.float16)),
        ("N2_gamma_elementwise", mk(arch="B", nc="gamma_elementwise")),
        ("N4_wrong_orientation", mk(arch="B", nc="wrong_orientation")),
        ("N5_fold_both_halves", mk(arch="B", nc="fold_both_halves")),
    ]
    for name, adapter in configs:
        agg.append(run_config(model, name, adapter, ids_list, naive_rot))

    # ---------------- gate decision ----------------
    per = {r["config"]: r for r in agg}
    ok_cfgs = ["FP00_stock", "FP01_A_explicit", "FP01b_A_folded",
               "FP02_B2_split", "FP02x_explicit_Mgamma"]
    neg_cfgs = ["FP03_single_folded", "FP04_gamma_omitted", "N1_naive",
                "N2_gamma_elementwise", "N4_wrong_orientation",
                "N5_fold_both_halves"]
    if target_rot_exact < 1.0:
        gate_reasons.append(f"rotated target greedy != stock ({target_rot_exact})")
    for c in ok_cfgs:
        if per[c]["exact_match_rate"] < 1.0:
            gate_reasons.append(f"{c} not output-preserving "
                                f"({per[c]['exact_match_rate']})")
    base = per["FP02_B2_split"]["mean_acceptance"]
    for c in ok_cfgs:
        if abs(per[c]["mean_acceptance"] - base) > TOL_ACCEPT:
            gate_reasons.append(f"{c} acceptance {per[c]['mean_acceptance']} "
                                f"vs B2 {base} (tol {TOL_ACCEPT})")
    # paired per-prompt criterion: degraded iff mean drop > 0.1 AND >=75% prompts
    import pandas as _pd
    _piv = _pd.DataFrame(rows).pivot_table(index="prompt_id", columns="config",
                                           values="mean_acceptance")
    for c in neg_cfgs:
        d = _piv["FP02_B2_split"] - _piv[c]
        if not (float(d.mean()) > 0.1 and float((d > 0).mean()) >= 0.75):
            gate_reasons.append(f"negative control {c} did NOT degrade "
                                f"(paired drop {float(d.mean()):.3f}, "
                                f"{float((d > 0).mean()):.0%} prompts)")
    gate = len(gate_reasons) == 0

    logging_utils.write_csv(os.path.join(ART, "fp_equivalence.csv"), rows)
    with open(os.path.join(ART, "fp_equivalence.json"), "w") as f:
        json.dump(dict(stop_gate_a_pass=gate, gate_reasons=gate_reasons,
                       target_rotation_exact_match=target_rot_exact,
                       tol_accept=TOL_ACCEPT, neg_drop=NEG_DROP,
                       n_prompts=args.num_prompts,
                       max_new_tokens=args.max_new_tokens,
                       configs=agg), f, indent=2)
    with open(os.path.join(ART, "projection_dispatch_trace.jsonl"), "w") as f:
        for r in dispatch_rows:
            f.write(json.dumps(r) + "\n")
    # dispatch summary csv
    from collections import Counter
    cnt = Counter((r["config"], r["projection_selected"]) for r in dispatch_rows)
    logging_utils.write_csv(os.path.join(ART, "projection_dispatch_summary.csv"),
                            [dict(config=c, projection=p, calls=n)
                             for (c, p), n in sorted(cnt.items())])
    print(f"[gateA] STOP GATE A: {'PASS' if gate else 'FAIL'} "
          f"{gate_reasons if gate_reasons else ''}", flush=True)
    print(f"[gateA] DONE -> {ART}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
