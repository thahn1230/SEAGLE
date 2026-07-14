#!/usr/bin/env python
"""Common-prefix counterfactual analysis (spec §12.1/§14/§19).

Fixed contexts = stock-target greedy trajectories (prompt + 64 tokens).
At every position of the SAME token sequence, teacher-forced forwards give:

    p_FP  stock target distribution            (post-norm h hooked)
    p_Q   fake-W4A4 target distribution        (fused tail, a_t hooked)
    q_FP  stock EAGLE first-step distribution  (stock draft on stock h)
    q_Q   B2 W4A4 draft first-step distribution (full fake-W4A4 draft on a_t;
          a full-sequence draft forward is all-first-path by construction)

Per position and per pairing (p_FP,q_FP) (p_Q,q_FP) (p_FP,q_Q) (p_Q,q_Q):
    overlap = 1 - TV(p, q), forward/reverse KL, JS
Per target pair (p_FP vs p_Q):
    top-1 flip, entropy delta, top1-prob & top1-top2 margin delta,
    delta log p(draft FP proposal), position class A-E (spec §14).

Writes artifacts/b2_split_study/{common_prefix_positions.csv,
common_prefix_summary.json}.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
    python scripts/run_b2_common_prefix.py --device cuda:0 --num-prompts 12
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)
from eagle_spinquant.b2_projection import B2SplitDraftAdapter  # noqa: E402

ART = os.path.join(PROJECT_ROOT, "artifacts", "b2_split_study")


@torch.no_grad()
def run_gen(gen, ilen, mx):
    final, prev = None, ilen
    for out in gen:
        final = out
        if out.shape[1] - ilen >= mx:
            break
    return final[0, :ilen + mx].tolist()


def dist_metrics(p_log, q_log):
    """p_log/q_log: [T, V] log-probs. Returns per-position overlap/KLs/JS."""
    p, q = p_log.exp(), q_log.exp()
    tv = 0.5 * (p - q).abs().sum(-1)
    fkl = (p * (p_log - q_log)).sum(-1)
    rkl = (q * (q_log - p_log)).sum(-1)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p_log - m.log())).sum(-1) + \
         0.5 * (q * (q_log - m.log())).sum(-1)
    return (1 - tv), fkl, rkl, js


@torch.no_grad()
def target_pass(model, seqs, dev, hooked_norm):
    """Teacher-forced full-sequence forward; returns per-prompt (logits, hidden)."""
    outs = []
    cap = {}
    h = hooked_norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o.detach()))
    for s in seqs:
        ids = torch.tensor([s], device=dev)
        logits = model.base_model(ids).logits[0].float()      # [T, V]
        outs.append((logits.cpu(), cap["h"][0].detach().to(torch.float16).cpu()))
    h.remove()
    return outs


@torch.no_grad()
def draft_first_pass(ea_layer, head, hidden, ids, dev):
    """Full-sequence draft forward (ALL rows are target-originated = first
    path); returns draft head logits [T-1, V] for positions 1..T-1."""
    hs = hidden[None, :-1].to(dev, torch.float16)
    inp = torch.tensor([ids[1:]], device=dev)
    ea_layer.reset()
    out = ea_layer(hs, input_ids=inp)
    return head(out[0].to(torch.float16)).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert torch.cuda.device_count() == 2
    dev = args.device

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]

    # ---------- stage 1: stock target -> trajectories + p_FP + q_FP --------
    print("[cpx] stock target ...", flush=True)
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

    seqs, plens = [], []
    for p in prompts:
        ids = build_prompt(tok, p["text"]).to(dev)
        seqs.append(run_gen(model.naive_generate(
            ids, temperature=0.0, max_steps=args.max_new_tokens + 4,
            tree_choices=tree), ids.shape[1], args.max_new_tokens))
        plens.append(ids.shape[1])
    tp_fp = target_pass(model, seqs, dev, model.base_model.model.norm)
    q_fp = [draft_first_pass(model.ea_layer, model.base_model.lm_head,
                             h, s, dev) for (_, h), s in zip(tp_fp, seqs)]
    del model; torch.cuda.empty_cache()

    # ---------- stage 2: fake-W4A4 target -> p_Q + q_Q ---------------------
    print("[cpx] fake-W4A4 target ...", flush=True)
    model, stash2, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "w4a4", 0, device=dev, rotations_root=rr)
    study.set_draft_tree(model, tree, dev)
    tp_q = target_pass(model, seqs, dev, model.base_model.model.norm)  # a_t hooked
    ad = B2SplitDraftAdapter(model, stash2, dev, torch.float16, arch="B",
                             quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
                             quant_ar="fake_w4a4", ar_r2r4=True,
                             trace=False).install()
    ad._fc_idx = 0        # full-sequence forward = first path (all target rows)
    q_q = []
    for (_, a_t), s in zip(tp_q, seqs):
        ad._fc_idx = 0    # re-arm first path per sequence
        q_q.append(draft_first_pass(model.ea_layer, ad.head, a_t, s, dev))
    ad.uninstall()

    # ---------- per-position metrics ----------------------------------------
    rows = []
    for pi, s in enumerate(seqs):
        L0 = plens[pi]
        pf = F.log_softmax(tp_fp[pi][0], -1)          # [T, V]
        pq = F.log_softmax(tp_q[pi][0], -1)
        qf = F.log_softmax(q_fp[pi], -1)              # [T-1, V] (pos 1..T-1)
        qq = F.log_softmax(q_q[pi], -1)
        # generated region only: predicting tokens L0..T-1 -> target rows L0-1..,
        # draft rows L0-2.. (draft row t predicts token t+1)
        T = pf.shape[0]
        sl_t = slice(L0 - 1, T - 1)
        sl_d = slice(L0 - 2, T - 2)
        PF, PQ, QF, QQ = pf[sl_t], pq[sl_t], qf[sl_d], qq[sl_d]
        pairs = {"pFP_qFP": (PF, QF), "pQ_qFP": (PQ, QF),
                 "pFP_qQ": (PF, QQ), "pQ_qQ": (PQ, QQ)}
        met = {k: dist_metrics(a, b) for k, (a, b) in pairs.items()}
        top2_f = PF.topk(2, -1)
        top2_q = PQ.topk(2, -1)
        ent_f = -(PF.exp() * PF).sum(-1)
        ent_q = -(PQ.exp() * PQ).sum(-1)
        prop = QF.argmax(-1)                            # fp draft proposal
        lp_f = PF.gather(-1, prop[:, None])[:, 0]
        lp_q = PQ.gather(-1, prop[:, None])[:, 0]
        for j in range(PF.shape[0]):
            flip = bool(top2_f.indices[j, 0] != top2_q.indices[j, 0])
            d_ov = float(met["pQ_qFP"][0][j] - met["pFP_qFP"][0][j])
            row = dict(prompt_id=prompts[pi]["question_id"], position=j,
                       target_fp_top1=int(top2_f.indices[j, 0]),
                       target_q_top1=int(top2_q.indices[j, 0]),
                       target_top1_flip=flip,
                       target_fp_entropy=float(ent_f[j]),
                       target_q_entropy=float(ent_q[j]),
                       d_entropy=float(ent_q[j] - ent_f[j]),
                       target_fp_margin=float(top2_f.values[j, 0] - top2_f.values[j, 1]),
                       target_q_margin=float(top2_q.values[j, 0] - top2_q.values[j, 1]),
                       d_logp_fp_draft_proposal=float(lp_q[j] - lp_f[j]),
                       d_overlap_pQqFP_vs_pFPqFP=d_ov)
            for k, (ov, fkl, rkl, js) in met.items():
                row[f"overlap_{k}"] = float(ov[j])
                row[f"fkl_{k}"] = float(fkl[j])
                row[f"js_{k}"] = float(js[j])
            # §14 classification for the target-only comparison
            if d_ov > 0.01:
                if flip:
                    cls = "D_incidental_rank_flip"
                elif row["d_entropy"] > 0.05 and \
                        row["target_q_margin"] < row["target_fp_margin"]:
                    cls = "B_distribution_flattening"
                else:
                    cls = "A_or_C_alignment(needs quality cross-check)"
            elif d_ov < -0.01:
                cls = "acceptance_decrease"
            else:
                cls = "E_within_noise"
            row["position_class"] = cls
            rows.append(row)

    logging_utils.write_csv(os.path.join(ART, "common_prefix_positions.csv"), rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = dict(
        n_prompts=args.num_prompts, n_positions=len(df),
        mean_overlap=dict(df[[c for c in df.columns
                              if c.startswith("overlap_")]].mean().round(4)),
        target_top1_flip_rate=round(float(df.target_top1_flip.mean()), 4),
        mean_d_entropy=round(float(df.d_entropy.mean()), 4),
        mean_d_margin=round(float((df.target_q_margin
                                   - df.target_fp_margin).mean()), 4),
        mean_d_logp_draft_proposal=round(
            float(df.d_logp_fp_draft_proposal.mean()), 4),
        position_class_counts=dict(df.position_class.value_counts()),
        flip_rate_by_fp_margin_bin={
            f"{lo:.1f}-{hi:.1f}": round(float(
                df[(df.target_fp_margin >= lo)
                   & (df.target_fp_margin < hi)].target_top1_flip.mean()), 4)
            for lo, hi in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 99)]})
    with open(os.path.join(ART, "common_prefix_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str))
    print("[cpx] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
