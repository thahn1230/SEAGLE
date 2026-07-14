#!/usr/bin/env python
"""STOP GATE A (concat-selective): 7B fp16 equivalence + architecture comparison
+ verifier execution-shape consistency (§17).

Configs (fp16, greedy):
  F0_stock                          stock target + stock EAGLE
  -- rotated fp16 target --
  F1_prev_B2_rotated_embedding      previous B2 (rotated embedding, conjugated fc)
  F2_concat_selective_explicit      new architecture, explicit slice ops
  F3_concat_selective_folded        new architecture, folded pre-R + output R1
  F4_embedding_rotated (NC)         e·R1 fed to untouched W_e
  F5_orig_PL_recurrent (NC)         recurrent hidden not absorbed (W_h unrotated)
  N_no_output_R (NC)                post-projection R1 omitted
  N_R_before_PL (NC)                R1 applied to the 2D input instead
  N_first_for_recurrent (NC)        first projection reused recurrently
  N_recurrent_for_first (NC)        recurrent projection used on first forward

Verifier-consistency stage: target logits on identical token sequences via
(a) full-sequence forward, (b) two-chunk KV forward, (c) token-by-token
incremental KV — for the fp16 AND fake-W4A4 targets; per-position rel-L2,
top-1 agreement, margin certificate ||Δ||∞ < margin/2.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/validate_concat_selective_fp.py --device cuda:0 \
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
from eagle_spinquant.b2_projection import B2SplitDraftAdapter  # noqa: E402
from eagle_spinquant.concat_selective_projection import (  # noqa: E402
    ConcatSelectiveDraftAdapter)

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")
TOL_ACCEPT = 0.15


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


@torch.no_grad()
def verifier_consistency(model, seqs, dev, tag, rows):
    """Full-sequence vs chunked-KV vs incremental-KV target logits on the SAME
    token sequences (generated region positions)."""
    from eagle.model.kv_cache import initialize_past_key_values
    bm = model.base_model
    bm.model.tree_mask = None          # clear residual tree mask from ea_generate
    for pi, s in enumerate(seqs):
        ids = torch.tensor([s], device=dev)
        T = ids.shape[1]
        cut = max(1, T - 32)
        lg_full = bm(ids).logits[0].float()

        # EAGLE's own preallocated KVCache machinery — the exact cache
        # implementation the verifier uses during tree decoding
        past, _pd, _cl = initialize_past_key_values(bm)
        o1 = bm(ids[:, :cut], past_key_values=past, use_cache=True)
        o2 = bm(ids[:, cut:], past_key_values=past, use_cache=True)
        lg_chunk = torch.cat([o1.logits[0].float(), o2.logits[0].float()], 0)
        del past, _pd, _cl, o1, o2                 # ~2GiB preallocated cache
        torch.cuda.empty_cache()

        past2, _pd2, _cl2 = initialize_past_key_values(bm)
        out = bm(ids[:, :cut], past_key_values=past2, use_cache=True)
        lg_inc = [out.logits[0].float()]
        for t in range(cut, T):
            out = bm(ids[:, t:t + 1], past_key_values=past2, use_cache=True)
            lg_inc.append(out.logits[0].float())
        lg_inc = torch.cat(lg_inc, 0)
        del past2, _pd2, _cl2, out
        torch.cuda.empty_cache()
        sl = slice(cut - 1, T - 1)
        for name, lg in (("chunked_kv", lg_chunk), ("incremental_kv", lg_inc)):
            a, b = lg[sl], lg_full[sl]
            top2 = b.topk(2, -1)
            margin = (top2.values[:, 0] - top2.values[:, 1])
            dinf = (a - b).abs().max(-1).values
            rows.append(dict(
                target=tag, prompt=pi, path=name, n_positions=int(b.shape[0]),
                rel_l2=round(float((a - b).norm() / b.norm()), 6),
                max_abs=round(float((a - b).abs().max()), 4),
                top1_agree=round(float((a.argmax(-1) == b.argmax(-1))
                                       .float().mean()), 4),
                certified_frac=round(float((dinf < margin / 2).float().mean()), 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = args.device
    for d in ("fp16_equivalence", "dispatch_traces", "verifier_consistency",
              "mismatch_dumps"):
        os.makedirs(os.path.join(ART, d), exist_ok=True)
    with open(os.path.join(ART, "commands.sh"), "a") as f:
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

    rows, dispatch_rows, vc_rows, gate_reasons = [], [], [], []

    def run_config(model, name, adapter, ids_list, naive_ref):
        if adapter:
            adapter.install()
        accs, ok = [], []
        for pi, ids in enumerate(ids_list):
            if adapter is not None and hasattr(adapter, "set_context"):
                adapter.set_context(prompts[pi]["question_id"])
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            ar = naive_ref[pi]
            n = min(len(ar), len(eg))
            exact = bool(ar[:n] == eg[:n] and len(ar) == len(eg))
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am); ok.append(exact)
            rows.append(dict(config=name, prompt_id=prompts[pi]["question_id"],
                             mean_acceptance=round(am, 4), exact_match=exact))
            if not exact:
                with open(os.path.join(ART, "mismatch_dumps",
                                       f"{name}_p{pi}.json"), "w") as f:
                    json.dump(dict(config=name, naive=ar, eagle=eg), f)
        if adapter:
            if hasattr(adapter, "trace_rows"):
                for t in adapter.trace_rows:
                    dispatch_rows.append(dict(config=name, **t))
            if hasattr(adapter, "dispatch_summary"):
                print(f"[csfp] {name} dispatch: {adapter.dispatch_summary()}",
                      flush=True)
            adapter.uninstall()
        res = dict(config=name, mean_acceptance=round(float(np.mean(accs)), 4),
                   exact_match_rate=round(float(np.mean(ok)), 4),
                   n_prompts=len(accs))
        print(f"[csfp] {name}: accept={res['mean_acceptance']} "
              f"exact={res['exact_match_rate']}", flush=True)
        return res

    # ---------- stage 1: stock ----------
    print("[csfp] building STOCK target ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "none", "random_hadamard", "none", 0, device=dev, rotations_root=rr)
    R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    stash["R1"] = R["R1"].clone()
    stash["gamma_f"] = model.base_model.model.norm.weight.detach().float().cpu().clone()
    stash["lm_head_weight"] = model.base_model.lm_head.weight.detach().float().cpu().clone()
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
    naive_stock, seqs = [], []
    for ids in ids_list:
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_stock.append(t)
        seqs.append(ids[0].tolist() + t)
    agg = [run_config(model, "F0_stock", None, ids_list, naive_stock)]
    print("[csfp] verifier consistency (fp16 target) ...", flush=True)
    verifier_consistency(model, seqs[:4], dev, "fp16", vc_rows)
    del model; torch.cuda.empty_cache()

    # ---------- stage 2: rotated fp16 target ----------
    print("[csfp] building ROTATED fp16 target ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "none", 0, device=dev, rotations_root=rr)
    study.set_draft_tree(model, tree, dev)
    naive_rot, rot_match = [], []
    for pi, ids in enumerate(ids_list):
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_rot.append(t)
        rot_match.append(bool(t == naive_stock[pi]))
    target_rot_exact = float(np.mean(rot_match))
    print(f"[csfp] rotated-target greedy == stock: {target_rot_exact}", flush=True)

    mk = lambda **kw: ConcatSelectiveDraftAdapter(model, stash, dev,
                                                  torch.float16, **kw)
    configs = [
        ("F1_prev_B2_rotated_embedding",
         B2SplitDraftAdapter(model, stash, dev, torch.float16, arch="B")),
        ("F2_concat_selective_explicit", mk(variant="explicit")),
        ("F3_concat_selective_folded", mk(variant="folded")),
        ("F4_embedding_rotated", mk(variant="folded", nc="embedding_rotated")),
        ("F5_orig_PL_recurrent", mk(variant="folded", nc="orig_PL_recurrent")),
        ("N_no_output_R", mk(variant="folded", nc="no_output_R")),
        ("N_R_before_PL", mk(variant="folded", nc="R_before_PL")),
        ("N_first_for_recurrent", mk(variant="folded", nc="first_for_recurrent")),
        ("N_recurrent_for_first", mk(variant="folded", nc="recurrent_for_first")),
    ]
    for name, adapter in configs:
        agg.append(run_config(model, name, adapter, ids_list, naive_rot))

    # ---------- gate ----------
    per = {r["config"]: r for r in agg}
    ok_cfgs = ["F0_stock", "F1_prev_B2_rotated_embedding",
               "F2_concat_selective_explicit", "F3_concat_selective_folded"]
    neg_cfgs = ["F4_embedding_rotated", "F5_orig_PL_recurrent", "N_no_output_R",
                "N_R_before_PL", "N_first_for_recurrent",
                "N_recurrent_for_first"]
    if target_rot_exact < 1.0:
        gate_reasons.append(f"rotated target != stock ({target_rot_exact})")
    for c in ok_cfgs:
        if per[c]["exact_match_rate"] < 1.0:
            gate_reasons.append(f"{c} not output-preserving")
    base = per["F3_concat_selective_folded"]["mean_acceptance"]
    for c in ok_cfgs:
        if abs(per[c]["mean_acceptance"] - base) > TOL_ACCEPT:
            gate_reasons.append(f"{c} acceptance vs F3: "
                                f"{per[c]['mean_acceptance']} vs {base}")
    import pandas as pd
    piv = pd.DataFrame(rows).pivot_table(index="prompt_id", columns="config",
                                         values="mean_acceptance")
    neg_stats = {}
    for c in neg_cfgs:
        d = piv["F3_concat_selective_folded"] - piv[c]
        md, frac = float(d.mean()), float((d > 0).mean())
        neg_stats[c] = dict(paired_mean_drop=round(md, 4),
                            frac_prompts_degraded=frac)
        if not (md > 0.1 and frac >= 0.75):
            gate_reasons.append(f"NC {c} did not degrade ({md:.3f}, {frac:.0%})")
    gate = len(gate_reasons) == 0

    logging_utils.write_csv(os.path.join(ART, "fp16_equivalence",
                                         "fp_equivalence.csv"), rows)
    with open(os.path.join(ART, "fp16_equivalence", "fp_equivalence.json"),
              "w") as f:
        json.dump(dict(stop_gate_a_pass=gate, gate_reasons=gate_reasons,
                       target_rotation_exact_match=target_rot_exact,
                       negative_control_criterion="paired drop>0.1 & >=75% prompts",
                       negative_control_paired_stats=neg_stats,
                       configs=agg), f, indent=2)
    logging_utils.write_csv(os.path.join(ART, "verifier_consistency",
                                         "fp16_target.csv"), vc_rows)
    with open(os.path.join(ART, "dispatch_traces", "dispatch_trace.jsonl"),
              "w") as f:
        for r in dispatch_rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    cnt = Counter((r["config"], r.get("selected_projection") or r.get("projection_selected")) for r in dispatch_rows)
    logging_utils.write_csv(os.path.join(ART, "dispatch_traces",
                                         "dispatch_summary.csv"),
                            [dict(config=c, projection=p, calls=n)
                             for (c, p), n in sorted(cnt.items())])
    print(f"[csfp] STOP GATE A: {'PASS' if gate else 'FAIL'} {gate_reasons}",
          flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
