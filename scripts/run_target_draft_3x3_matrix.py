#!/usr/bin/env python
"""Primary Target × Draft 3×3 precision matrix (bitwidth-AL component study).

Cells and architecture contracts (docs/INITIAL_BITWIDTH_AL_COMPONENT_CAUSALITY_AUDIT.md §4):
  T16_D16  stock target + stock EAGLE
  T8_D16 / T4_D16   fused fake-W8A8/W4A4 target + A-explicit bridge
                    (a_t → R1ᵀ → γ → h_t) + ORIGINAL fp16 draft
  T16_D8 / T16_D4   STOCK fp16 target (untouched) + concat-selective draft,
                    first block [W_e | W_h] (identity mode, h_t input),
                    recurrent [W_e | W_h·R1], post-R1, head W_lm·R1
  T8_D8 T8_D4 T4_D8 T4_D4   fused quantized target (a_t) + concat-selective
                    (gamma_R1 first mode)

Draft precision D8/D4 = fake W8A8/W4A4 on BOTH pre-R projections + the 7 AR
decoder linears (R2/R4 applied); draft embedding + scoring head stay fp16
ISOLATED (separate component ablations cover them). KV fp16 everywhere.

Per prompt records: mean AL (tau = accepted draft tokens + 1 bonus), full
acceptance list, per-cycle accepted depth, first rejection depth, token/cycle
counts, exact match vs the SAME target's naive decoding, mismatch position.

Also prints/saves the parameter-ownership audit (ids + data_ptrs) once.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_target_draft_3x3_matrix.py --run-dir <dir> \
      --group stock --device cuda:0 --num-prompts 20 --max-new-tokens 64
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study, fake_w4a4_draft as fq)
from eagle_spinquant.concat_selective_projection import (  # noqa: E402
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality")

GROUP_TARGET = {"stock": ("none", "none"), "t8": ("full", "w8a8"),
                "t4": ("full", "w4a4")}

def draft_kwargs(dbits, fhm):
    if dbits == 16:
        return None if fhm is None else dict(first_hidden_mode=fhm)
    mode = f"fake_w{dbits}a{dbits}"
    return dict(first_hidden_mode=fhm, quant_first=mode, quant_recurrent=mode,
                quant_ar=mode, ar_r2r4=True)

# cell -> (group, kind, kwargs)
CELLS = {
    "T16_D16": ("stock", "none", None),
    "T16_D8": ("stock", "CS", draft_kwargs(8, "identity")),
    "T16_D4": ("stock", "CS", draft_kwargs(4, "identity")),
    "T8_D16": ("t8", "A", None),
    "T8_D8": ("t8", "CS", draft_kwargs(8, "gamma_R1")),
    "T8_D4": ("t8", "CS", draft_kwargs(4, "gamma_R1")),
    "T4_D16": ("t4", "A", None),
    "T4_D8": ("t4", "CS", draft_kwargs(8, "gamma_R1")),
    "T4_D4": ("t4", "CS", draft_kwargs(4, "gamma_R1")),
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


def ownership_audit(model):
    bm = model.base_model
    ea = model.ea_layer
    return dict(
        target_embed=dict(module_id=id(bm.model.embed_tokens),
                          weight_id=id(bm.model.embed_tokens.weight),
                          data_ptr=bm.model.embed_tokens.weight.data_ptr()),
        draft_embed=dict(module_id=id(ea.embed_tokens),
                         weight_id=id(ea.embed_tokens.weight),
                         data_ptr=ea.embed_tokens.weight.data_ptr()),
        target_lm_head=dict(module_id=id(bm.lm_head),
                            weight_id=id(bm.lm_head.weight),
                            data_ptr=bm.lm_head.weight.data_ptr()),
        shared_embed_storage=bool(
            bm.model.embed_tokens.weight.data_ptr()
            == ea.embed_tokens.weight.data_ptr()),
        note=("stock EAGLE passes bm.lm_head INTO topK_genrate (alias); our "
              "adapters substitute an isolated head with its own storage"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--group", required=True, choices=list(GROUP_TARGET))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--cells", default=None, help="comma list; default all in group")
    ap.add_argument("--rotation-kind", default="random_hadamard",
                    help="R.bin selector: 'random_hadamard' (seed 0) or a "
                         "named dir under rotations_root, e.g. "
                         "'learned_chat_w4a4kv16'")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    # NOTE 2026-07-15: physical GPU 7 dropped off the bus. Original policy was
    # CVD="6,7"; on 2026-07-15 the user explicitly authorized GPUs 0-5 as well
    # (recorded in spinquant_ppl_reproduction_fix/environment/environment.txt),
    # so any single permitted GPU is now accepted.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        ("6,7", "0", "1", "2", "3", "4", "5", "6"), \
        os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() in (1, 2), torch.cuda.device_count()
    if torch.cuda.device_count() == 1:
        print("[gpu] WARNING: physical GPU 7 absent; running on physical GPU 6 only", flush=True)
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
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    rotation, quant = GROUP_TARGET[args.group]
    print(f"[3x3] target {rotation}/{quant} on {dev} ...", flush=True)
    model, stash, r_bin_used = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rotation, args.rotation_kind, quant, 0, device=dev,
        rotations_root=rr)
    if r_bin_used is not None:
        import hashlib
        _R = torch.load(r_bin_used, map_location="cpu", weights_only=False)
        _h = hashlib.sha256(_R["R1"].float().numpy().tobytes()).hexdigest()[:12]
        print(f"[3x3] rotation_kind={args.rotation_kind} r_bin={r_bin_used} "
              f"R1_sha={_h}", flush=True)
    if args.group == "stock":
        R = torch.load(study.r_bin_path(args.rotation_kind, 0,
                                        paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight.detach() \
            .float().cpu().clone()
        with open(os.path.join(ART, "logs", "parameter_ownership.json"), "w") as f:
            json.dump(ownership_audit(model), f, indent=2)
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
    print("[3x3] naive reference ...", flush=True)
    naive_ref = []
    for ids in ids_list:
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_ref.append(t)

    want = set(args.cells.split(",")) if args.cells else None
    rows = []
    for cell, (grp, kind, kwargs) in CELLS.items():
        if grp != args.group or (want and cell not in want):
            continue
        if kind == "none":
            adapter = None
        elif kind == "A":
            adapter = UnrotateAdapter(model, stash, dev, torch.float16,
                                      with_gamma=True)
        else:
            adapter = ConcatSelectiveDraftAdapter(
                model, stash, dev, torch.float16, variant="folded", **kwargs)
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
            mism = next((i for i in range(n) if ar[i] != eg[i]), -1)
            depths = [d - 1 for d in deltas]           # accepted draft tokens
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am)
            rows.append(dict(
                cell=cell, prompt_id=prompts[pi]["question_id"],
                mean_acceptance=round(am, 4),
                acceptance_list=json.dumps(deltas),
                accepted_depths=json.dumps(depths),
                first_rejection_depths=json.dumps(
                    [min(d + 1, 5) for d in depths]),
                n_new_tokens=len(eg), n_cycles=len(deltas), tree_size=26,
                exact_match=bool(ar[:n] == eg[:n] and len(ar) == len(eg)),
                first_mismatch=mism))
        cov_ok = True
        if adapter is not None and hasattr(adapter, "split") and \
                kind == "CS" and kwargs and kwargs.get("quant_first", "fp16") != "fp16":
            for sel in ("projection_first_preR", "projection_recurrent_preR"):
                m = getattr(adapter.split, sel)
                cov_ok &= isinstance(m, fq.FakeW4A4Linear) and m.n_forward > 0
            for parent, attr, _o in adapter._replaced_ar:
                cov_ok &= getattr(parent, attr).n_forward > 0
        if adapter:
            adapter.uninstall()
        print(f"[3x3] {cell}: AL={np.mean(accs):.4f} cov_ok={cov_ok}", flush=True)
        assert cov_ok, f"{cell}: coverage failure"

    logging_utils.write_csv(os.path.join(rd, "shards",
                                         f"al__{args.group}.csv"), rows)
    print(f"[3x3] group {args.group} DONE -> {rd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
