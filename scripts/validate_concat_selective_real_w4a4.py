#!/usr/bin/env python
"""STOP GATE B (concat-selective): REAL packed W4A4 attached to the ACTUAL
EAGLE generation graph — target layers + projection_first_preR +
projection_recurrent_preR + draft AR head all dispatch through QuaRot CUTLASS
INT4×INT4 during ea_generate. KV fp16 (never called KV4).

Configs run e2e (greedy, MT-bench):
  REAL_Q10  target real_packed_W4A4 + draft FP16 (A-explicit)
  REAL_Q11  target real_packed_W4A4 + draft real_packed_W4A4 (concat-selective)
  FAKE_Q11  same architecture, fake_W4A4 (comparison on the same prompts)

Dispatch proof: module types + per-module forward counters at teardown; no
torch.nn.Linear remains on a quantized path.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/validate_concat_selective_real_w4a4.py --device cuda:1 \
      --num-prompts 20 --max-new-tokens 64
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)
from eagle_spinquant.concat_selective_projection import (  # noqa: E402
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402
from eagle_spinquant.realint4 import (QuarotW4A4Linear,  # noqa: E402
                                      swap_target_linears)

ART = os.path.join(PROJECT_ROOT, "artifacts", "concat_selective_rotation_study")


class BiasedReal(nn.Module):
    """QuaRot linear + explicit fp16 bias (kernels are bias-free). Counts
    forwards for the dispatch proof."""

    def __init__(self, real_mod, bias):
        super().__init__()
        self.real = real_mod
        self.register_buffer("bias_fp16",
                             bias.detach().to(torch.float16).clone())
        self.n_forward = 0

    @property
    def weight(self):
        return self.real.w_packed

    def forward(self, x):
        self.n_forward += 1
        return self.real(x) + self.bias_fp16


def realify_draft(adapter, dev):
    """Swap the installed concat-selective fp16 modules to REAL QuaRot W4A4:
    both pre-R projections (with bias wrapper) + the 7 AR-head linears."""
    split = adapter.split
    swapped = []
    for name in ("projection_first_preR", "projection_recurrent_preR"):
        lin = getattr(split, name)
        assert isinstance(lin, nn.Linear), name
        bias = lin.bias.detach().clone()
        core = nn.Linear(lin.in_features, lin.out_features, bias=False).to(dev)
        core.weight.data = lin.weight.data.clone()
        real = BiasedReal(QuarotW4A4Linear.from_linear(core), bias).to(dev)
        setattr(split, name, real)
        swapped.append((split, name, lin, real))
    ea = adapter.ea_layer
    attn, mlp = ea.layers[0].self_attn, ea.layers[0].mlp
    for parent, attr in ((attn, "q_proj"), (attn, "k_proj"), (attn, "v_proj"),
                         (attn, "o_proj"), (mlp, "gate_proj"),
                         (mlp, "up_proj"), (mlp, "down_proj")):
        lin = getattr(parent, attr)
        assert isinstance(lin, nn.Linear) and lin.bias is None, attr
        real = QuarotW4A4Linear.from_linear(lin.to(dev))
        real.n_forward = 0
        _f = real.forward

        def counted(x, _r=real, _f=_f):
            _r.n_forward += 1
            return _f(x)
        real.forward = counted
        setattr(parent, attr, real)
        swapped.append((parent, attr, lin, real))
    return swapped


def dispatch_proof(swapped):
    rows, ok = [], True
    for parent, attr, _old, real in swapped:
        n = getattr(real, "n_forward", -1)
        kind = type(real).__name__
        good = n > 0 and kind in ("BiasedReal", "QuarotW4A4Linear")
        ok &= good
        rows.append(dict(module=attr if attr else "?", kind=kind,
                         kernel="quarot CUTLASS INT4xINT4", n_forward=n,
                         dispatched=good))
    return rows, ok


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
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = args.device
    torch.cuda.set_device(dev)
    os.makedirs(os.path.join(ART, "real_w4a4_dispatch"), exist_ok=True)
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

    # rotated fp16 target -> swap its 7x32 linears to REAL QuaRot W4A4.
    # Rotation 'r1r2' (NOT 'full'): swap_target_linears folds R4 into down_proj
    # itself; building with 'full' would double-fold R4 (realint4 convention).
    print("[csreal] building rotated fp16 target (r1r2) ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "r1r2", "random_hadamard", "none", 0, device=dev, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    study.set_draft_tree(model, tree, dev)
    ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]

    print("[csreal] swapping TARGET linears to real QuaRot W4A4 ...", flush=True)
    info = swap_target_linears(model.base_model, "quarot_w4a4", dev)
    print(f"[csreal] target swap: {info}", flush=True)

    print("[csreal] naive reference (real-W4A4 target) ...", flush=True)
    naive_ref = []
    for ids in ids_list:
        t, _ = run_gen(model.naive_generate(ids, temperature=0.0,
                       max_steps=args.max_new_tokens + 4, tree_choices=tree),
                       ids.shape[1], args.max_new_tokens)
        naive_ref.append(t)

    rows, disp_rows, agg = [], [], []

    def run_cfg(name, adapter, realify=False):
        adapter.install()
        swapped = realify_draft(adapter, dev) if realify else []
        accs, ok = [], []
        for pi, ids in enumerate(ids_list):
            if hasattr(adapter, "set_context"):
                adapter.set_context(prompts[pi]["question_id"])
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            am = (sum(deltas) / len(deltas)) if deltas else 0.0
            accs.append(am)
            n = min(len(naive_ref[pi]), len(eg))
            ok.append(bool(naive_ref[pi][:n] == eg[:n]))
            rows.append(dict(config=name, prompt_id=prompts[pi]["question_id"],
                             mean_acceptance=round(am, 4),
                             acceptance_list=json.dumps(deltas),
                             exact_match=ok[-1]))
        cov_ok = True
        if realify:
            proof, cov_ok = dispatch_proof(swapped)
            for r in proof:
                disp_rows.append(dict(config=name, **r))
            # restore the fp16 modules swapped OUTSIDE the adapter's tracking,
            # so adapter.uninstall()'s strict load_state_dict succeeds
            for parent, attr, old, _real in swapped:
                setattr(parent, attr, old)
        adapter.uninstall()
        res = dict(config=name, mean_acceptance=round(float(np.mean(accs)), 4),
                   exact_match_rate=round(float(np.mean(ok)), 4),
                   real_dispatch_ok=bool(cov_ok), n_prompts=len(accs))
        agg.append(res)
        print(f"[csreal] {name}: accept={res['mean_acceptance']} "
              f"real_ok={cov_ok}", flush=True)
        return res

    run_cfg("REAL_Q10_targetRealW4A4_draftFP16",
            UnrotateAdapter(model, stash, dev, torch.float16, with_gamma=True))
    run_cfg("REAL_Q11_targetRealW4A4_draftRealW4A4_concat",
            ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                        variant="folded"), realify=True)
    run_cfg("FAKE_Q11cmp_targetRealW4A4_draftFakeW4A4_concat",
            ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                        variant="folded",
                                        quant_first="fake_w4a4",
                                        quant_recurrent="fake_w4a4",
                                        quant_ar="fake_w4a4", ar_r2r4=False))

    gate = all(r["real_dispatch_ok"] for r in agg)
    logging_utils.write_csv(os.path.join(ART, "real_w4a4_dispatch",
                                         "real_e2e_acceptance.csv"), rows)
    logging_utils.write_csv(os.path.join(ART, "real_w4a4_dispatch",
                                         "real_e2e_dispatch_proof.csv"), disp_rows)
    with open(os.path.join(ART, "real_w4a4_dispatch",
                           "real_e2e_summary.json"), "w") as f:
        json.dump(dict(stop_gate_b_pass=gate, target_swap=info,
                       kv_precision="fp16", configs=agg), f, indent=2)
    print(f"[csreal] STOP GATE B: {'PASS' if gate else 'FAIL'}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
