#!/usr/bin/env python
"""First-vs-recurrent projection audit (study spec §15, RQ5/RQ6).

Runs the deployed D4P3 draft (fresh anchor) under the chosen target on
calib prompts, capturing every projection call's INPUT (with path label
first/rec and recurrent depth 1..4). Then, offline, for each alpha in
the pre-registered grid it recomputes the quantized projection output on
the captured inputs and reports per-path/per-depth:

  absmax, RMS, p99, p99.9, A4 saturation rate, A4 zero-code rate,
  projection-output NMSE vs the unquantized reference, per-branch NMSE.

Output: tables/first_recurrent_audit.json (path-optimal alphas by NMSE
+ all distribution stats). Deployment-level path-specific alpha is
adopted only under the pre-registered rule (tau +0.10, significant,
acceptable overhead) — decided at analysis time from these diagnostics.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter, build_concat_selective_weights)
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"
GRID = [8.0, 11.3137085, 16.0, 22.627417, 32.0, 45.254834, 64.0,
        90.509668, 128.0]
D = 4096


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="int4", choices=["fp16", "int4"])
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="deployed global alpha for the capture run")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, quant = (("none", "none") if args.target == "fp16"
                  else ("full", "w4a4"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    if stash.get("R1") is None:
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                        rr), map_location="cpu",
                       weights_only=False)
        stash["R1"] = R["R1"].clone()
        stash["gamma_f"] = model.base_model.model.norm.weight.detach() \
            .float().cpu().clone()
        stash["lm_head_weight"] = model.base_model.lm_head.weight \
            .detach().float().cpu().clone()
    ea = model.ea_layer
    sd_new = torch.load(args.anchor, map_location="cpu",
                        weights_only=False)
    sd_new = sd_new.get("draft_state_dict", sd_new.get("model", sd_new))
    ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                        for k, v in sd_new.items()}, strict=True)
    ea.to(dev)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    fhm = "identity" if args.target == "fp16" else "gamma_R1"
    ad = ConcatSelectiveDraftAdapter(
        model, stash, dev, torch.float16, variant="folded",
        first_hidden_mode=fhm, trace=False, embed_scale_alpha=args.alpha,
        quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True)
    ad.install()

    caps = {"first": [], **{f"rec{k}": [] for k in range(1, 5)}}
    state = {"depth": 0}
    orig_fwd = ad.split.forward

    def spy(z):
        if ad.split.select == "first":
            state["depth"] = 0
            key = "first"
        else:
            state["depth"] += 1
            key = f"rec{min(state['depth'], 4)}"
        if len(caps[key]) < 40:
            zz = z.reshape(-1, z.shape[-1])
            idx = torch.randperm(zz.shape[0])[:64]
            caps[key].append(zz[idx].detach().float().cpu())
        return orig_fwd(z)

    ad.split.forward = spy
    prompts, _ = load_eval_prompts("c4", args.n_prompts, "calib")
    for p in prompts:
        ids = build_prompt(tok, p["text"])[:, :768].to(dev)
        for out in model.ea_generate(ids, temperature=0.0, max_steps=48,
                                     tree_choices=tree):
            if out.shape[1] - ids.shape[1] >= 40:
                break
    ad.split.forward = orig_fwd
    ad.uninstall()

    # offline: unquantized fold reference + per-alpha quantized NMSE
    sd0 = {k: v.detach().cpu() for k, v in ea.state_dict().items()}
    R1c = stash["R1"].double()
    gc = stash["gamma_f"].double()
    Wf0, Wr0, bias = build_concat_selective_weights(
        sd0, R1c, gc, None, first_hidden_mode=fhm)
    out = {"deployed_alpha": args.alpha, "target": args.target,
           "paths": {}}
    for key, zs in caps.items():
        if not zs:
            continue
        Z = torch.cat(zs)                       # (N, 8192), alpha-scaled e
        Zr = Z.clone()
        Zr[:, :D] = Zr[:, :D] / args.alpha      # UNSCALED embed slice
        h = Z[:, D:]
        stats = dict(n=int(Z.shape[0]),
                     hidden_absmax=float(h.abs().max()),
                     hidden_rms=float(h.pow(2).mean().sqrt()),
                     hidden_p99=float(h.abs().flatten().quantile(0.99)),
                     hidden_p999=float(h.abs().flatten()
                                       .quantile(0.999)))
        W0 = (Wf0 if key == "first" else Wr0).float()
        ref = Zr.float() @ torch.cat(
            [W0[:, :D] , W0[:, D:]], dim=1).t()
        curves = {}
        for a in GRID:
            Wa = W0.clone()
            Wa[:, :D] = Wa[:, :D] / a
            Wq = fq._weight_fake_quant(Wa.half().to(dev), 4).float().cpu()
            Za = Zr.clone()
            Za[:, :D] = Za[:, :D] * a
            aq = fq._act_quantizer(4)
            aq.find_params(Za.half().to(dev))
            Zq = aq(Za.half().to(dev)).float().cpu()
            aq.free()
            sat = float((Za.abs() >= Za.abs().amax(-1, True) * 0.999)
                        .float().mean())
            zero = float((Zq == 0).float().mean())
            y = Zq @ Wq.t()
            nmse = float(((y - ref).pow(2).sum())
                         / (ref.pow(2).sum() + 1e-9))
            curves[str(a)] = dict(nmse=round(nmse, 6),
                                  a4_sat_rate=round(sat, 5),
                                  a4_zero_rate=round(zero, 5))
        best = min(curves, key=lambda k2: curves[k2]["nmse"])
        out["paths"][key] = dict(stats=stats, curves=curves,
                                 nmse_optimal_alpha=float(best))
    path = os.path.join(args.run_dir, "tables",
                        f"first_recurrent_audit_{args.target}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v["nmse_optimal_alpha"]
                      for k, v in out["paths"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
