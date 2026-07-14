#!/usr/bin/env python
"""Correctness pass for mode w4a4_spinquant_draft_pure_r1.

--task units : PL conjugation fp64/fp16, draft head basis, R1/R2/R3/R4 audit
               (no 7B generation)
--task gen   : build target (--group stock|rotated) and run baseline/ablation
               configs: AR vs EAGLE greedy, verifier logits, acceptance, trace.

Baselines: B0 original EAGLE (stock) | B1 unfused target + original draft |
B2 previous failed [R1.T,I]/[R1.T,R1.T]+fused head | B3 NEW pure-R1 draft.
Ablations A1-A6 per spec.

Greedy only. Reuses study.run_one_prompt-free direct generator driving.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/validate_spinquant_draft_pure_r1.py \
      --run-dir runs/spinquant_draft_pure_r1_<ts> --task units
  CUDA_VISIBLE_DEVICES=4 python ... --task gen --group rotated \
      --configs B1,B2,B3 --quant none --num-prompts 2 --max-new-tokens 16
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             pure_r1_eagle as pr, rotation_aware as ra, study,
                             spinquant_draft as spd, w4a4_impl_fix as wf)

DEV = "cuda:0"


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)).item()


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def task_units(run_dir, paths, cfg):
    import torch.nn as nn
    R = torch.load(study.r_bin_path("random_hadamard", 0, paths["target_path"],
                   cfg.get("paths", {}).get("rotations_root")),
                   map_location="cpu", weights_only=False)
    R1 = R["R1"].double()
    import glob
    sdp = glob.glob(os.path.join(paths["draft_path"], "pytorch_model.bin"))[0]
    dstate = torch.load(sdp, map_location="cpu", weights_only=True)
    fc = dstate["fc.weight"].double(); bias = dstate["fc.bias"].double()
    D = 4096
    W_e, W_h = fc[:, :D], fc[:, D:]
    from transformers import AutoConfig
    gamma = None  # not needed for R1-only PL conj
    # PL conjugation (user formula) vs convert_draft_state
    W_e_R = R1.t() @ W_e @ R1; W_h_R = R1.t() @ W_h @ R1; b_R = bias @ R1
    g = torch.Generator().manual_seed(0)
    e = torch.randn(8, D, generator=g, dtype=torch.float64)
    h = torch.randn(8, D, generator=g, dtype=torch.float64)
    pl_rows = []
    for dt, tag in ((torch.float64, "fp64"), (torch.float16, "fp16")):
        lhs = (torch.cat([e @ R1, h @ R1], -1).to(dt)
               @ torch.cat([W_e_R, W_h_R], -1).to(dt).t() + b_R.to(dt))
        rhs = ((torch.cat([e, h], -1).to(dt) @ fc.to(dt).t() + bias.to(dt)) @ R1.to(dt))
        pl_rows.append(dict(dtype=tag,
            check="concat([e@R1,h@R1])@W_PL_R.T+b_R == (concat([e,h])@W_PL.T+b)@R1",
            rel_l2=rel(lhs, rhs)))
    # convert_draft_state fc matches W_PL_R (stored fp32 -> compare at fp32 scale)
    conv = pr.build_pure_r1_draft_state(dstate, R1, torch.ones(D, dtype=torch.float64))
    pl_rows.append(dict(dtype="fp32_storage", check="convert_draft_state fc == R1.T@W_PL@blkdiag(R1,R1)",
                        rel_l2=rel(conv["fc.weight"].double(), torch.cat([W_e_R, W_h_R], 1))))
    logging_utils.write_csv(os.path.join(run_dir, "pl_conjugation_correctness.csv"), pl_rows)
    with open(os.path.join(run_dir, "pl_conjugation_correctness.json"), "w") as f:
        json.dump(pl_rows, f, indent=2)

    # draft head basis: f_R scored by W_lm@R1 (correct) vs gamma-fused vs original
    W_lm = None
    from safetensors import safe_open
    idx = json.load(open(os.path.join(paths["target_path"], "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(paths["target_path"], idx["lm_head.weight"]),
                   framework="pt", device="cpu") as fsf:
        W_lm = fsf.get_tensor("lm_head.weight").double()
    with safe_open(os.path.join(paths["target_path"], idx["model.norm.weight"]),
                   framework="pt", device="cpu") as fsf:
        gamma_f = fsf.get_tensor("model.norm.weight").double()
    f = torch.randn(8, D, generator=g, dtype=torch.float64)
    f_R = f @ R1
    ref = f @ W_lm.t()
    head_rows = [
        dict(head="W_lm @ R1 (CORRECT)", rel_l2=rel(f_R @ (W_lm @ R1).t(), ref),
             top1_agree=float((( f_R @ (W_lm@R1).t()).argmax(-1) == ref.argmax(-1)).float().mean())),
        dict(head="W_lm @ diag(gamma_f) @ R1 (A6 WRONG for f_R)",
             rel_l2=rel(f_R @ (W_lm @ torch.diag(gamma_f) @ R1).t(), ref),
             top1_agree=float((( f_R @ (W_lm@torch.diag(gamma_f)@R1).t()).argmax(-1) == ref.argmax(-1)).float().mean())),
        dict(head="W_lm (original, WRONG for f_R)", rel_l2=rel(f_R @ W_lm.t(), ref),
             top1_agree=float(((f_R @ W_lm.t()).argmax(-1) == ref.argmax(-1)).float().mean())),
    ]
    logging_utils.write_csv(os.path.join(run_dir, "draft_head_basis_audit.csv"), head_rows)

    # R1/R2/R3/R4 draft rotation audit
    rot_rows = spd.draft_rotation_audit_rows(dstate)
    logging_utils.write_csv(os.path.join(run_dir, "draft_rotation_audit.csv"), rot_rows)

    verd = {
        "PL_conjugation_fp64_ok": all(r["rel_l2"] < 1e-12 for r in pl_rows if r["dtype"] == "fp64"),
        "PL_conjugation_fp64_rel_l2": next(r["rel_l2"] for r in pl_rows if r["dtype"] == "fp64"),
        "convert_draft_state_matches_fp32": next(r["rel_l2"] for r in pl_rows if r["dtype"] == "fp32_storage"),
        "PL_conjugation_fp16_rel_l2": next(r["rel_l2"] for r in pl_rows if r["dtype"] == "fp16"),
        "head_correct_top1": head_rows[0]["top1_agree"],
        "head_gamma_fused_top1": head_rows[1]["top1_agree"],
        "head_original_top1": head_rows[2]["top1_agree"],
        "R2_fp16_equiv": rot_rows[1]["fp16_equivalence_error"],
        "R3_fp16_equiv": rot_rows[2]["fp16_equivalence_error"],
        "R4_fp16_equiv": rot_rows[3]["fp16_equivalence_error"],
    }
    with open(os.path.join(run_dir, "units_verdicts.json"), "w") as f:
        json.dump(verd, f, indent=2)
    print(json.dumps(verd, indent=2))
    return verd


# ---------------------------------------------------------------------------
# generation configs
# ---------------------------------------------------------------------------

def make_config(name, model, stash, device):
    """Return (tail_adapter or None, draft_adapter or None, meta)."""
    if name == "B0":                                   # stock: unrotated, orig draft
        return None, None, dict(desc="original EAGLE (unrotated)")
    tail = wf.UnfusedTailAdapter(model, stash)         # exposes h, original head
    if name == "B1":
        return tail, None, dict(desc="unfused target + ORIGINAL draft on h")
    if name == "B2":
        d = wf.DraftProjTransformAdapter(model, stash, device, torch.float16,
              first_embed="R1T", first_hidden="I", rec_embed="R1T",
              rec_hidden="R1T", head_mode="fused")
        return tail, d, dict(desc="prev failed [R1.T,I]/[R1.T,R1.T]+fused head")
    if name == "B3":
        d = spd.SpinquantDraftPureR1Adapter(model, stash, device, torch.float16, trace=True)
        return tail, d, dict(desc="NEW pure-R1 draft")
    # ablations
    A = dict(
        A1=dict(rotate_external=False),
        A2=dict(once_per_prompt=True),
        A3=dict(rotate_recurrent=True),
        A6=dict(head_mode="gamma_fused"),
    )
    if name in A:
        d = spd.SpinquantDraftPureR1Adapter(model, stash, device, torch.float16,
                                            trace=(name == "B3"), **A[name])
        return tail, d, dict(desc=f"ablation {name}")
    if name == "A4":  # PL input rotated but PL weight NOT conjugated (orig draft)
        d = _RawRotateAdapter(model, stash, device, torch.float16)
        return tail, d, dict(desc="A4 rotate external h but original (un-conjugated) fc")
    if name == "A5":  # PL conjugated but no R2/R3/R4 == B3 in fp16 (identity)
        d = spd.SpinquantDraftPureR1Adapter(model, stash, device, torch.float16)
        return tail, d, dict(desc="A5 PL-conjugated, no R2/R3/R4 (fp16-identical to B3)")
    raise ValueError(name)


class _RawRotateAdapter(spd.SpinquantDraftPureR1Adapter):
    """A4: rotate external h@R1 but keep the ORIGINAL (un-conjugated) draft fc."""
    def install(self):
        # do NOT conjugate the draft; only substitute head + runtime rotate
        ea = self.ea_layer
        adapter = self
        self._orig_topk = ea.topK_genrate

        def wrapped(hidden_states, input_ids, head, logits_processor, *a, **k):
            hs = (hidden_states.to(torch.float32) @ adapter.R1).to(hidden_states.dtype)
            return adapter._orig_topk(hs, input_ids, adapter.head,
                                      logits_processor, *a, **k)
        ea.topK_genrate = wrapped
        return self

    def uninstall(self):
        if self._orig_topk is not None:
            self.ea_layer.topK_genrate = self._orig_topk
            self._orig_topk = None


@torch.no_grad()
def run_gen_tokens(gen, input_len, max_new):
    final, deltas, prev = None, [], input_len
    for out in gen:
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        final = out
        if cur - input_len >= max_new:
            break
    return final[0, input_len:input_len + max_new].tolist(), deltas


def task_gen(run_dir, paths, cfg, args):
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    rot = "none" if args.group == "stock" else "full"
    print(f"[spd] building target group={args.group} quant={args.quant}...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, "random_hadamard", args.quant, 0, device=DEV,
        rotations_root=cfg.get("paths", {}).get("rotations_root"))
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, DEV)
    model_pair = f"{cfg['model']['target']}+{cfg['model']['eagle_draft']}"

    rows, all_trace = [], []
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in configs:
        tail, draft, meta = make_config(name, model, stash, DEV)
        if tail:
            tail.install()
        if draft:
            draft.install()
        timers = study.PhaseTimers(model).install()
        for p in prompts:
            ids = build_prompt(tok, p["text"]).to(DEV)
            if draft is not None and hasattr(draft, "set_context"):
                draft.set_context(p["question_id"])
            ilen = ids.shape[1]
            ar, _ = run_gen_tokens(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ilen, args.max_new_tokens)
            eg, deltas = run_gen_tokens(model.ea_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 8, tree_choices=tree),
                       ilen, args.max_new_tokens)
            n = min(len(ar), len(eg))
            match = (ar[:n] == eg[:n]) and len(ar) == len(eg)
            # verifier logit rel-l2 on matched prefixes (same-path recompute)
            vl = []
            base = ids[0].tolist()
            for step in range(min(n, 8)):
                la = model.base_model(torch.tensor([base + ar[:step]], device=DEV),
                                      use_cache=False).logits[0, -1]
                lv = model.base_model(torch.tensor([base + eg[:step]], device=DEV),
                                      use_cache=False).logits[0, -1]
                vl.append(rel(la, lv))
            rows.append(dict(model_pair=model_pair, quant_mode=args.quant,
                baseline_name=name, prompt_id=p["question_id"],
                acceptance_length_mean=(sum(deltas)/len(deltas)) if deltas else 0.0,
                acceptance_length_list=str(deltas),
                exact_token_match=bool(match),
                verifier_logit_rel_l2_mean=(sum(vl)/len(vl)) if vl else 0.0,
                tokens_per_second="", notes=meta["desc"]))
        if name == "B3" and draft is not None and draft.forward_trace:
            all_trace.extend(draft.forward_trace)
        acc = [r["acceptance_length_mean"] for r in rows if r["baseline_name"] == name]
        mm = [r["exact_token_match"] for r in rows if r["baseline_name"] == name]
        print(f"[spd] {name:4s} accept={sum(acc)/len(acc):.3f}  "
              f"match={sum(mm)/len(mm):.2f}  ({meta['desc']})", flush=True)
        timers.uninstall()
        if draft:
            draft.uninstall()
        if tail:
            tail.uninstall()

    # write a per-invocation SHARD (parallel jobs must not clobber each other)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    tag = f"{args.group}_{args.quant}_{'-'.join(configs)}"
    out_csv = os.path.join(run_dir, "shards", f"{tag}.csv")
    logging_utils.write_csv(out_csv, rows)
    if all_trace:
        logging_utils.write_csv(os.path.join(run_dir, "draft_forward_basis_trace.csv"),
                                all_trace)
    print(f"[spd] wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", required=True, choices=["units", "gen"])
    ap.add_argument("--group", default="rotated", choices=["stock", "rotated"])
    ap.add_argument("--configs", default="B1,B2,B3")
    ap.add_argument("--quant", default="none", choices=["none", "w4a4"])
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    if not os.path.isfile(os.path.join(run_dir, "environment.txt")):
        with open(os.path.join(run_dir, "environment.txt"), "w") as f:
            f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    if args.task == "units":
        task_units(run_dir, paths, cfg)
    else:
        task_gen(run_dir, paths, cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
