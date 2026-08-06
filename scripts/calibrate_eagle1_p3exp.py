#!/usr/bin/env python
"""Exponent-P3 calibration (study §9, §12). m(D, beta) = D**beta, D=4096.

Stages (held-out calib data only; never MT-Bench):
  stage capture : collect first + recurrent projection inputs from the
                  deployed D4P3 draft on the calib pool (per target)
  stage coarse  : pathwise beta grid 0.00..1.00 step 0.05 — projection
                  NMSE (total/embed-branch/hidden-branch), cosine,
                  W_e-effective W4 NMSE, per path
  stage fine    : +-0.05 around each path optimum, step 0.01
  stage pairs   : 5x5 best-first x best-rec pairs ranked by projection
                  metrics -> tables/p3exp_pairs.json (top-9 for the
                  20-prompt acceptance stage, top-3 for MT-Bench)

Mapping checks (test_p3exp_factor_mapping): beta 0 -> m 1;
log(32)/log(4096) -> 32; log(45.254834)/log(4096) -> 45.254834;
0.5 -> 64.
"""
import argparse, json, math, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

D = 4096


def m_of(beta):
    return float(D) ** float(beta)


def beta_of(m):
    return math.log(m) / math.log(D)


def nmse(y, ref):
    return float(((y - ref).pow(2).sum())
                 / (ref.pow(2).sum() + 1e-12))


def eval_beta(Z_unscaled, W0, beta, fq, dev):
    """Z_unscaled: (N, 8192) with RAW e-slice; W0: unquantized fold."""
    m = m_of(beta)
    Wa = W0.clone()
    Wa[:, :D] = Wa[:, :D] / m
    Wq = fq._weight_fake_quant(Wa.half().to(dev), 4).float()
    Za = Z_unscaled.clone()
    Za[:, :D] = Za[:, :D] * m
    aq = fq._act_quantizer(4)
    aq.find_params(Za.half().to(dev))
    Zq = aq(Za.half().to(dev)).float()
    aq.free()
    ref = Z_unscaled.float().to(dev) @ W0.t().float().to(dev)
    y = Zq @ Wq.t()
    ref_e = Z_unscaled[:, :D].float().to(dev) @ \
        W0[:, :D].t().float().to(dev)
    y_e = Zq[:, :D] @ Wq[:, :D].t()
    ref_h = ref - ref_e
    y_h = y - y_e
    cos = float(torch.nn.functional.cosine_similarity(
        y.flatten(), ref.flatten(), dim=0))
    we_nmse = nmse(Wq[:, :D].cpu() * m, W0[:, :D].float())
    return dict(beta=round(float(beta), 4), m=round(m, 4),
                nmse=nmse(y, ref), nmse_e=nmse(y_e, ref_e),
                nmse_h=nmse(y_h, ref_h), cos=cos, we_w4_nmse=we_nmse)


def load_capture(rd, target):
    p = os.path.join(rd, "tables", f"p3exp_capture_{target}.pt")
    return torch.load(p, weights_only=False)


def stage_capture(args):
    from eagle_spinquant import eagle_bridge, experiment, study
    from eagle_spinquant.concat_selective_projection import (
        ConcatSelectiveDraftAdapter, build_concat_selective_weights)
    from eagle_spinquant.eval_datasets import load_eval_prompts
    dev = "cuda:0"
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w8a8") if args.target == "w8a8"
                  else ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, "learned_chat_w4a4kv16", quant, 0, device=dev,
        rotations_root=rr)
    if stash.get("R1") is None:
        R = torch.load(study.r_bin_path("learned_chat_w4a4kv16", 0,
                                        paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight \
            .detach().float().cpu().clone()
    ea = model.ea_layer
    sd = torch.load(args.anchor, map_location="cpu", weights_only=False)
    sd = sd.get("draft_state_dict", sd.get("model", sd))
    ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                        for k, v in sd.items()}, strict=True)
    ea.to(dev)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    fhm = "identity" if args.target == "fp16" else "gamma_R1"
    dep_alpha = args.alpha
    ad = ConcatSelectiveDraftAdapter(
        model, stash, dev, torch.float16, variant="folded",
        first_hidden_mode=fhm, trace=False, embed_scale_alpha=dep_alpha,
        quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True)
    ad.install()
    caps = {"first": [], "rec": []}
    orig = ad.split.forward

    def spy(z):
        key = "first" if ad.split.select == "first" else "rec"
        if len(caps[key]) < 60:
            zz = z.reshape(-1, z.shape[-1]).detach().float().cpu()
            idx = torch.randperm(zz.shape[0])[:64]
            caps[key].append(zz[idx])
        return orig(z)

    ad.split.forward = spy
    prompts, _ = load_eval_prompts("c4", args.n_prompts, "calib")
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :768].to(dev)
        for out in model.ea_generate(ids, temperature=0.0, max_steps=44,
                                     tree_choices=tree):
            if out.shape[1] - ids.shape[1] >= 36:
                break
    ad.split.forward = orig
    ad.uninstall()
    sd0 = {k: v.detach().cpu() for k, v in ea.state_dict().items()}
    Wf0, Wr0, _b = build_concat_selective_weights(
        sd0, stash["R1"].double(), stash["gamma_f"].double(), None,
        first_hidden_mode=fhm)
    blob = {}
    for key in caps:
        Z = torch.cat(caps[key])
        Z[:, :D] = Z[:, :D] / dep_alpha        # store RAW e-slice
        blob[key] = Z.half()
    blob["W_first"] = Wf0.float()
    blob["W_rec"] = Wr0.float()
    blob["deployed_alpha"] = dep_alpha
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    torch.save(blob, os.path.join(args.run_dir, "tables",
                                  f"p3exp_capture_{args.target}.pt"))
    print(f"[p3exp] captured first={blob['first'].shape} "
          f"rec={blob['rec'].shape}")
    return 0


def stage_search(args):
    from eagle_spinquant import fake_w4a4_draft as fq
    dev = "cuda:0"
    blob = load_capture(args.run_dir, args.target)
    out = {"target": args.target, "paths": {}}
    for key, Wkey in (("first", "W_first"), ("rec", "W_rec")):
        Z = blob[key].float()
        W0 = blob[Wkey]
        coarse = [eval_beta(Z, W0, b / 100.0, fq, dev)
                  for b in range(0, 101, 5)]
        b0 = min(coarse, key=lambda r: r["nmse"])["beta"]
        fine = [eval_beta(Z, W0, max(0, min(1, b0 + d / 100.0)), fq, dev)
                for d in range(-5, 6)]
        allr = {r["beta"]: r for r in coarse + fine}
        rows = sorted(allr.values(), key=lambda r: r["nmse"])
        out["paths"][key] = dict(grid=sorted(allr.values(),
                                             key=lambda r: r["beta"]),
                                 best5=[r["beta"] for r in rows[:5]],
                                 best=rows[0])
        print(f"[p3exp] {args.target}/{key}: best beta "
              f"{rows[0]['beta']} (m={rows[0]['m']}, "
              f"nmse={rows[0]['nmse']:.5f})")
    # joint 5x5 pairs ranked by combined projection NMSE (first-path
    # NMSE at beta_f + rec-path NMSE at beta_r; separable by design)
    pf = {r["beta"]: r for r in out["paths"]["first"]["grid"]}
    pr = {r["beta"]: r for r in out["paths"]["rec"]["grid"]}
    pairs = []
    for bf in out["paths"]["first"]["best5"]:
        for br in out["paths"]["rec"]["best5"]:
            pairs.append(dict(beta_first=bf, beta_rec=br,
                              m_first=round(m_of(bf), 4),
                              m_rec=round(m_of(br), 4),
                              nmse_sum=pf[bf]["nmse"] + pr[br]["nmse"]))
    pairs.sort(key=lambda r: r["nmse_sum"])
    out["pairs25"] = pairs
    out["top9"] = pairs[:9]
    out["top3_by_nmse"] = pairs[:3]
    path = os.path.join(args.run_dir, "tables",
                        f"p3exp_search_{args.target}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[p3exp] -> {path}; top pair {pairs[0]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["capture", "search"])
    ap.add_argument("--target", required=True,
                    choices=["fp16", "int4", "w8a8"])
    ap.add_argument("--anchor")
    ap.add_argument("--alpha", type=float, default=None,
                    help="deployed capture alpha (legacy calibrated)")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    # mapping self-checks
    assert abs(m_of(0.0) - 1.0) < 1e-9
    assert abs(m_of(beta_of(32.0)) - 32.0) < 1e-6
    assert abs(m_of(beta_of(45.254834)) - 45.254834) < 1e-6
    assert abs(m_of(0.5) - 64.0) < 1e-9
    return dict(capture=stage_capture,
                search=stage_search)[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
