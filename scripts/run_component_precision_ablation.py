#!/usr/bin/env python
"""Component × precision ablation matrix (§8) + branchwise/mixed experiments
(§9.4). Groups (one target build each; stock target unless noted):

  draft  : draft-side components on STOCK fp16 target + concat-selective
           (identity first mode). Components × {w8a16,w16a8,w8a8,w4a16,w16a4,w4a4}:
             first, recurrent, ar, embed (table W / output A / both), head, full
  thead  : TARGET LM-head-only fake quant (stock target body, stock draft)
  tembed : TARGET embedding-only fake quant (stock body, stock draft)
  tbody  : TARGET body full-path quant per mode (fused rotated target build
           per mode) + A-explicit fp16 draft
  branch : branchwise/mixed concat experiments (§9.4 A-E) + proj-vs-head mixes

Draft embedding/LM head remain fp16-isolated except in their own ablations.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_component_precision_ablation.py --run-dir <dir> \
      --group draft --device cuda:0 --num-prompts 20 --max-new-tokens 64
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
    QUANT_BITS, ConcatSelectiveDraftAdapter)
from eagle_spinquant.study import UnrotateAdapter  # noqa: E402

MODES = ("fake_w8a16", "fake_w16a8", "fake_w8a8",
         "fake_w4a16", "fake_w16a4", "fake_w4a4")


def draft_configs():
    cfgs = {}
    for m in MODES:
        tag = m.replace("fake_", "")
        w, a = QUANT_BITS[m]
        cfgs[f"draft_first__{tag}"] = dict(quant_first=m)
        cfgs[f"draft_recurrent__{tag}"] = dict(quant_recurrent=m)
        cfgs[f"draft_ar__{tag}"] = dict(quant_ar=m, ar_r2r4=True)
        cfgs[f"draft_head__{tag}"] = dict(quant_head=m)
        # embedding: weight-only / act-only / both, per §6.2 semantics
        if a == 16:                       # W8A16 / W4A16 -> table weight only
            cfgs[f"draft_embed__{tag}"] = dict(quant_embed=m)
        elif w == 16:                     # W16A8 / W16A4 -> lookup output only
            cfgs[f"draft_embed__{tag}"] = dict(quant_embed_act=m)
        else:                             # W8A8 / W4A4 -> both
            cfgs[f"draft_embed__{tag}"] = dict(
                quant_embed=f"fake_w{w}a16", quant_embed_act=f"fake_w16a{a}")
        cfgs[f"draft_full__{tag}"] = dict(quant_first=m, quant_recurrent=m,
                                          quant_ar=m, ar_r2r4=True)
    return cfgs


def branch_configs():
    return {
        # A vs B: activation-only 4-bit on BOTH projections
        "branch_A_fullconcat_W16A4": dict(quant_first="fake_w16a4",
                                          quant_recurrent="fake_w16a4"),
        "branch_B_branchwise_W16A4": dict(branch_act=(4, 4)),
        # C: mixed branch precisions (activation-only)
        "branch_C_eFP16_hA4": dict(branch_act=(16, 4)),
        "branch_C_eA8_hA4": dict(branch_act=(8, 4)),
        "branch_C_eA4_hFP16": dict(branch_act=(4, 16)),
        "branch_C_eA4_hA8": dict(branch_act=(4, 8)),
        # A-with-weights vs B-with-weights at W4
        "branch_A_fullconcat_W4A4": dict(quant_first="fake_w4a4",
                                         quant_recurrent="fake_w4a4"),
        "branch_B_branchwise_W4A4": dict(quant_first="fake_w4a16",
                                         quant_recurrent="fake_w4a16",
                                         branch_act=(4, 4)),
        # D/E: mixed first/recurrent precision
        "mixed_D_first8_rec4": dict(quant_first="fake_w8a8",
                                    quant_recurrent="fake_w4a4"),
        "mixed_E_first4_rec8": dict(quant_first="fake_w4a4",
                                    quant_recurrent="fake_w8a8"),
        # projection vs LM head mixes (§9.4 E')
        "mix_proj8_head4": dict(quant_first="fake_w8a8",
                                quant_recurrent="fake_w8a8",
                                quant_head="fake_w4a4"),
        "mix_proj4_head8": dict(quant_first="fake_w4a4",
                                quant_recurrent="fake_w4a4",
                                quant_head="fake_w8a8"),
    }


class TargetHeadEmbedPatch:
    """Isolated TARGET LM-head or embedding fake quant on the stock target."""

    def __init__(self, model, kind, mode, device):
        self.bm = model.base_model
        self.kind, self.mode, self.dev = kind, mode, device
        self._orig = None

    def install(self):
        w, a = QUANT_BITS[self.mode]
        if self.kind == "lm_head":
            lin = self.bm.lm_head
            self._orig = lin
            self.bm.lm_head = fq.FakeW4A4Linear(
                lin.weight, None, "target_lm_head",
                quant_weight=(w < 16), quant_act=(a < 16),
                w_bits=w, a_bits=a).to(self.dev)
        else:                                   # embedding
            emb = self.bm.model.embed_tokens
            self._orig_w = emb.weight.data.clone()
            if w < 16:
                emb.weight.data = fq._weight_fake_quant(
                    emb.weight.data, w).to(emb.weight.dtype)
            self._act_wrap = None
            if a < 16:
                from eagle_spinquant.concat_selective_projection import \
                    EmbedOutputActQuant
                self._act_wrap = EmbedOutputActQuant(emb, a).to(self.dev)
                self.bm.model.embed_tokens = self._act_wrap
        return self

    def uninstall(self):
        if self.kind == "lm_head":
            self.bm.lm_head = self._orig
        else:
            if getattr(self, "_act_wrap", None) is not None:
                self.bm.model.embed_tokens = self._act_wrap.embed
            self.bm.model.embed_tokens.weight.data = self._orig_w

    def set_context(self, *_):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--group", required=True,
                    choices=["draft", "thead", "tembed", "tbody", "branch"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() in (1, 2)
    dev = args.device if torch.cuda.device_count() > 1 or \
        args.device == "cuda:0" else "cuda:0"
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

    rows = []

    def run_on(model, stash, name, adapter, ids_list, naive_ref):
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
            rows.append(dict(config=name, group=args.group,
                             prompt_id=prompts[pi]["question_id"],
                             mean_acceptance=round(am, 4),
                             acceptance_list=json.dumps(deltas),
                             exact_match=bool(ar[:n] == eg[:n]
                                              and len(ar) == len(eg))))
        if adapter:
            adapter.uninstall()
        print(f"[comp] {name}: AL={np.mean(accs):.4f}", flush=True)

    def build(rotation, quant):
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"], cfg["model"]["target"],
            rotation, "random_hadamard", quant, 0, device=dev,
            rotations_root=rr)
        if rotation == "none":
            R = torch.load(study.r_bin_path("random_hadamard", 0,
                                            paths["target_path"], rr),
                           map_location="cpu", weights_only=False)
            stash["R1"] = R["R1"].clone()
            stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
                .float().cpu().clone()
            stash["lm_head_weight"] = model.base_model.lm_head.weight \
                .detach().float().cpu().clone()
        tok = eagle_bridge.get_tokenizer(model)
        study.set_draft_tree(model, tree, dev)
        ids_list = [build_prompt(tok, p["text"]).to(dev) for p in prompts]
        naive = []
        for ids in ids_list:
            t, _ = run_gen(model.naive_generate(
                ids, temperature=0.0, max_steps=args.max_new_tokens + 4,
                tree_choices=tree), ids.shape[1], args.max_new_tokens)
            naive.append(t)
        return model, stash, ids_list, naive

    if args.group in ("draft", "branch"):
        model, stash, ids_list, naive = build("none", "none")
        table = draft_configs() if args.group == "draft" else branch_configs()
        for name, kw in table.items():
            kw = dict(kw)
            if kw.get("quant_first") == "fp16":
                kw.pop("quant_first")
            ad = ConcatSelectiveDraftAdapter(
                model, stash, dev, torch.float16, variant="folded",
                first_hidden_mode="identity", trace=False, **kw)
            run_on(model, stash, name, ad, ids_list, naive)
    elif args.group in ("thead", "tembed"):
        model, stash, ids_list, naive = build("none", "none")
        kind = "lm_head" if args.group == "thead" else "embedding"
        for m in MODES:
            tag = m.replace("fake_", "")
            patch = TargetHeadEmbedPatch(model, kind, m, dev)
            patch.install()
            # target changed -> its naive reference changes too: recompute
            naive_q = []
            for ids in ids_list:
                t, _ = run_gen(model.naive_generate(
                    ids, temperature=0.0, max_steps=args.max_new_tokens + 4,
                    tree_choices=tree), ids.shape[1], args.max_new_tokens)
                naive_q.append(t)
            run_on(model, stash, f"target_{kind}__{tag}", None, ids_list,
                   naive_q)
            patch.uninstall()
    else:                                    # tbody: 6 fused builds
        import gc
        for m in MODES:
            tag = m.replace("fake_", "")
            qname = tag  # QUANT_CFGS key format wXaY
            model, stash, ids_list, naive = build("full", qname)
            ad = UnrotateAdapter(model, stash, dev, torch.float16,
                                 with_gamma=True)
            run_on(model, stash, f"target_body__{tag}", ad, ids_list, naive)
            # ad/stash/ids/naive keep GPU refs alive -> next 7B build OOMs
            del ad, model, stash, ids_list, naive
            gc.collect()
            torch.cuda.empty_cache()

    logging_utils.write_csv(os.path.join(rd, "shards",
                                         f"comp__{args.group}.csv"), rows)
    print(f"[comp] group {args.group} DONE -> {rd}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
