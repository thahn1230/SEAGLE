#!/usr/bin/env python
"""Phase 3: Stock vs concat-selective FP16 acceptance-equivalence forensics.

Deterministic settings (TF32 off, fixed seeds, deterministic algorithms
warn-only). Runs three drafts on the SAME stock fp16 target:

    stock          stock EAGLE (reference)
    cs_fp16        concat-selective folded, identity first mode, all fp16
    cs_fp32fc      same but BOTH pre-R projections kept in FP32
                   (isolates transformed-weight fp16 casting)

Per prompt: acceptance lists compared cycle-by-cycle; for every prompt where
stock vs cs_fp16 DIVERGE, a per-level draft-logit trace at the first diverging
cycle: ||Δz||∞ between implementations, top-10 set equality, and the top-k
boundary margin certificate ||Δz||∞ < (z_k − z_{k+1})/2 at k=10 (root) and the
per-level tree fan-outs.

Writes artifacts/bitwidth_al_component_causality/equivalence_forensics/
  {per_prompt_al.csv, divergence_trace.csv, certificates.csv, summary.json}

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_acceptance_equivalence_forensics.py --device cuda:1
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)
from eagle_spinquant.concat_selective_projection import (  # noqa: E402
    ConcatSelectiveDraftAdapter)

ART = os.path.join(PROJECT_ROOT, "artifacts", "bitwidth_al_component_causality",
                   "equivalence_forensics")


def deterministic():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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


class HeadLogitTap:
    """Captures the DRAFT head logits per level inside topK_genrate calls."""

    def __init__(self, head_module):
        self.head = head_module
        self.per_cycle = []          # list per cycle: list of [k?] logits rows
        self._cur = None
        self._h = None

    def start_cycle(self):
        self._cur = []
        self.per_cycle.append(self._cur)

    def install(self):
        def hook(m, i, o):
            if self._cur is not None:
                # last row = the node whose top-k drives the next level
                self._cur.append(o.detach()[..., -1, :].reshape(-1).float().cpu()
                                 if o.dim() > 2 else
                                 o.detach().reshape(o.shape[0], -1)[-1].float().cpu())
        self._h = self.head.register_forward_hook(hook)
        return self

    def remove(self):
        if self._h:
            self._h.remove()


def fp32_projections(adapter):
    """Keep both pre-R projection Linears in FP32 (weights + compute)."""
    split = adapter.split
    for name in ("projection_first_preR", "projection_recurrent_preR"):
        lin = getattr(split, name)
        lin32 = torch.nn.Linear(lin.in_features, lin.out_features, bias=True)
        lin32.weight.data = lin.weight.data.float()
        lin32.bias.data = lin.bias.data.float()
        lin32 = lin32.to(lin.weight.device)
        orig_fwd = lin32.forward

        def f(x, _l=lin32, _o=orig_fwd):
            return _o(x.float()).to(x.dtype)
        lin32.forward = f
        setattr(split, name, lin32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    # NOTE 2026-07-15: physical GPU 7 dropped off the bus mid-study (nvidia-smi
    # shows indices 0-6 only). CVD stays "6,7" per the hard constraint; we run
    # on the remaining visible device (cuda:0 = physical 6) and never touch
    # 0-5. Documented in the final report as a hardware deviation.
    assert torch.cuda.device_count() in (1, 2), torch.cuda.device_count()
    if torch.cuda.device_count() == 1:
        print("[gpu] WARNING: physical GPU 7 absent; running on physical GPU 6 only", flush=True)
    deterministic()
    dev = args.device
    if torch.cuda.device_count() == 1 and dev != "cuda:0":
        print(f"[gpu] {dev} unavailable -> cuda:0 (physical 6)", flush=True)
        dev = "cuda:0"
    os.makedirs(ART, exist_ok=True)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    fanout = [10, 10, 10, 10, 10]      # top-k evaluated at k=10 per level (root fan)

    print("[forensic] building STOCK target ...", flush=True)
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

    # ---- run the three drafts, capturing per-level draft logits ----------
    def run_variant(name, adapter, tap_module):
        tap = HeadLogitTap(tap_module).install()
        orig_topk = model.ea_layer.topK_genrate

        def wrapped(*a, **k):
            tap.start_cycle()
            return orig_topk(*a, **k)
        model.ea_layer.topK_genrate = wrapped
        # adapter (if any) wraps AFTER us so its head substitution still works
        if adapter:
            adapter.install()
        res = []
        for pi, ids in enumerate(ids_list):
            if adapter is not None:
                adapter.set_context(prompts[pi]["question_id"])
            start = len(tap.per_cycle)
            eg, deltas = run_gen(model.ea_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            res.append(dict(prompt_id=prompts[pi]["question_id"], tokens=eg,
                            deltas=deltas,
                            cycles=tap.per_cycle[start:len(tap.per_cycle)]))
        if adapter:
            adapter.uninstall()
        model.ea_layer.topK_genrate = orig_topk
        tap.remove()
        al = float(np.mean([sum(r["deltas"]) / len(r["deltas"])
                            for r in res if r["deltas"]]))
        print(f"[forensic] {name}: AL={al:.4f}", flush=True)
        return res, al

    stock_res, stock_al = run_variant("stock", None, model.base_model.lm_head)

    ad = ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                     variant="folded",
                                     first_hidden_mode="identity", trace=False)
    cs_res, cs_al = run_variant("cs_fp16", ad, ad.head)

    ad32 = ConcatSelectiveDraftAdapter(model, stash, dev, torch.float16,
                                       variant="folded",
                                       first_hidden_mode="identity", trace=False)
    ad32.install()
    fp32_projections(ad32)
    ad32.uninstall_pending = True
    # rerun manually since install already done
    tap32 = HeadLogitTap(ad32.head).install()
    orig_topk = model.ea_layer.topK_genrate

    def wrapped32(*a, **k):
        tap32.start_cycle()
        return orig_topk(*a, **k)
    model.ea_layer.topK_genrate = wrapped32
    cs32_res = []
    for pi, ids in enumerate(ids_list):
        ad32.set_context(prompts[pi]["question_id"])
        start = len(tap32.per_cycle)
        eg, deltas = run_gen(model.ea_generate(
            ids, temperature=0.0, max_steps=args.max_new_tokens + 8,
            tree_choices=tree), ids.shape[1], args.max_new_tokens)
        cs32_res.append(dict(prompt_id=prompts[pi]["question_id"], tokens=eg,
                             deltas=deltas,
                             cycles=tap32.per_cycle[start:]))
    model.ea_layer.topK_genrate = orig_topk
    tap32.remove()
    ad32.uninstall()
    cs32_al = float(np.mean([sum(r["deltas"]) / len(r["deltas"])
                             for r in cs32_res if r["deltas"]]))
    print(f"[forensic] cs_fp32fc: AL={cs32_al:.4f}", flush=True)

    # ---- per-prompt AL comparison + first-divergence traces ---------------
    al_rows, div_rows, cert_rows = [], [], []
    n_diverge = 0
    for s_, c_, c32_ in zip(stock_res, cs_res, cs32_res):
        pid = s_["prompt_id"]
        same_tokens = s_["tokens"] == c_["tokens"]
        same_al = s_["deltas"] == c_["deltas"]
        same_al32 = s_["deltas"] == c32_["deltas"]
        al_rows.append(dict(prompt_id=pid,
                            stock_al=round(np.mean(s_["deltas"]), 4),
                            cs_fp16_al=round(np.mean(c_["deltas"]), 4),
                            cs_fp32fc_al=round(np.mean(c32_["deltas"]), 4),
                            tokens_equal=same_tokens,
                            al_list_equal=same_al,
                            al_list_equal_fp32fc=same_al32))
        # certificates on the FIRST few cycles (aligned state region)
        n_aligned = 0
        for k, (da, db) in enumerate(zip(s_["deltas"], c_["deltas"])):
            if da != db:
                break
            n_aligned += 1
        for cyc in range(min(n_aligned + 1, len(s_["cycles"]),
                             len(c_["cycles"]))):
            for lvl, (za, zb) in enumerate(zip(s_["cycles"][cyc],
                                               c_["cycles"][cyc])):
                if za.shape != zb.shape:
                    continue
                dinf = float((za - zb).abs().max())
                k = fanout[min(lvl, 4)]
                top = za.topk(k + 1).values
                margin = float(top[k - 1] - top[k])
                topk_same = bool(torch.equal(za.topk(k).indices.sort().values,
                                             zb.topk(k).indices.sort().values))
                cert_rows.append(dict(prompt_id=pid, cycle=cyc, level=lvl,
                                      delta_inf=round(dinf, 5),
                                      boundary_margin=round(margin, 5),
                                      certified=bool(dinf < margin / 2),
                                      topk_set_equal=topk_same))
                if not topk_same and cyc == n_aligned:
                    div_rows.append(dict(
                        prompt_id=pid, first_divergent_cycle=cyc, level=lvl,
                        delta_inf=round(dinf, 5),
                        boundary_margin=round(margin, 5),
                        margin_smaller_than_2delta=bool(margin < 2 * dinf)))
        if not same_al:
            n_diverge += 1

    logging_utils.write_csv(os.path.join(ART, "per_prompt_al.csv"), al_rows)
    logging_utils.write_csv(os.path.join(ART, "certificates.csv"), cert_rows)
    if div_rows:
        logging_utils.write_csv(os.path.join(ART, "divergence_trace.csv"),
                                div_rows)
    cert = [r for r in cert_rows]
    summary = dict(
        stock_al=round(stock_al, 4), cs_fp16_al=round(cs_al, 4),
        cs_fp32fc_al=round(cs32_al, 4),
        n_prompts=len(al_rows),
        prompts_al_diverging_fp16=n_diverge,
        prompts_al_diverging_fp32fc=int(sum(
            not r["al_list_equal_fp32fc"] for r in al_rows)),
        certified_fraction=round(float(np.mean([r["certified"]
                                                for r in cert])), 4) if cert else None,
        topk_stable_fraction=round(float(np.mean([r["topk_set_equal"]
                                                  for r in cert])), 4) if cert else None,
        deterministic_settings=dict(tf32=False, seed=0,
                                    deterministic_algorithms="warn_only"))
    with open(os.path.join(ART, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[forensic]", json.dumps(summary, indent=2), flush=True)
    print("[forensic] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
