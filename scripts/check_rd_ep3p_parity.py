#!/usr/bin/env python
"""Gate P: exact training/runtime parity under the EP3-P pathwise fold.

EP3-P twin of check_exact_path_parity.py (Gate D). Builds the runtime
ConcatSelectiveDraftAdapter with EP3-P pathwise scales
(embed_scale_alpha = D^0.40, embed_scale_alpha_rec = D^0.45,
first_fold_R = R_T) for a given R_D, and the differentiable
ExactQuantizedRotationForward with alpha_rec_init set — then requires:

  1. identical fp16 quantized weights for the 9 quantized tensors + head
     (max_abs_diff <= 2e-3 tolerance, same as Gate D);
  2. identical projection outputs, hidden, logits, greedy tokens along a
     K=4 teacher-forced chain;
  3. the same for a NON-trivial R_D (residual-perturbed R_T) — the study
     deployment case.

Legs: RT_ep3p_kv16 (R_D = R_T), RESID_ep3p_kv16 (perturbed R_D).
Writes <run>/gradchecks/gateP_ep3p_parity.json; exits 1 on failure.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               ResidualRotation,
                                               FullRotation)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_exact_path_parity import whash, run_runtime_chain  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_F, ALPHA_R = D ** 0.40, D ** 0.45
D4EP3P = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True,
              embed_scale_alpha=ALPHA_F, embed_scale_alpha_rec=ALPHA_R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    ea = model.ea_layer
    sd0 = {k: v.detach().cpu().clone() for k, v in ea.state_dict().items()}
    R_T = stash["R1"].clone()

    results, fails = {}, []
    for tag in ("RT_ep3p_kv16", "RESID_ep3p_kv16"):
        if tag.startswith("RESID"):
            rr_mod = ResidualRotation(R_T.float()).to(dev)
            with torch.no_grad():
                rr_mod.W[:64, 64:128] += 0.02
            rot = FullRotation(rr_mod.R().detach())
        else:
            rot = SharedRotation(R_T.float())
        R_D = rot.R().detach().cpu().double()

        st = dict(stash)
        st["R1"] = R_D
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=R_T, **D4EP3P)
        ad.install()

        eq = ExactQATRotatedDraft(
            sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
            rot.to(dev), alpha_init=ALPHA_F, alpha_rec_init=ALPHA_R,
            w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
            first_fold_R=R_T)
        qw, _tw = eq.quantized_weights()

        rt_w = dict(
            W_first=ad.split.projection_first_preR.w_fake,
            W_rec=ad.split.projection_recurrent_preR.w_fake,
            q=ea.layers[0].self_attn.q_proj.w_fake,
            k=ea.layers[0].self_attn.k_proj.w_fake,
            v=ea.layers[0].self_attn.v_proj.w_fake,
            o=ea.layers[0].self_attn.o_proj.w_fake,
            gate=ea.layers[0].mlp.gate_proj.w_fake,
            up=ea.layers[0].mlp.up_proj.w_fake,
            down=ea.layers[0].mlp.down_proj.w_fake,
            head=ad.head.weight)
        wres = {}
        for name in rt_w:
            h_rt, h_eq = whash(rt_w[name]), whash(qw[name])
            ok = h_rt == h_eq
            mx = float((rt_w[name].float() - qw[name].float())
                       .abs().max())
            wres[name] = dict(runtime=h_rt, exact=h_eq, equal=ok,
                              max_abs_diff=mx)
            if not ok and mx > 2e-3:
                fails.append(f"{tag}:{name} weight mismatch max={mx}")

        g = torch.Generator().manual_seed(7)
        T = 8
        tok_ids = torch.randint(10, 3000, (1, T + 1),
                                generator=g).to(dev)
        a_seq = (torch.randn(1, T, eq.D, generator=g) * 1.0).to(dev)
        rt = run_runtime_chain(ea, ad, tok_ids, a_seq, 4, ALPHA_F, dev,
                               kv_bits=16)
        tr = eq.forward_chain(tok_ids, a_seq, K=4, exact=True)
        cres = []
        for k, ((ry, rh, rlg, rtk, rkv), (tlg, th, ttk)) in \
                enumerate(zip(rt, tr)):
            d_h = float((rh.float() - th.float()).abs().max())
            d_lg = float((rlg - tlg).abs().max())
            tok_eq = bool((rtk == ttk).all())
            cres.append(dict(depth=k, max_dh=d_h, max_dlogits=d_lg,
                             greedy_tok_equal=tok_eq))
            if d_lg > 5e-2 or not tok_eq:
                fails.append(f"{tag}:depth{k} dlogits={d_lg} "
                             f"tok={tok_eq}")
        results[tag] = dict(weights=wres, chain=cres)
        ad.uninstall()
        del eq
        torch.cuda.empty_cache()

    out = os.path.join(args.run_dir, "gradchecks",
                       "gateP_ep3p_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(alpha_first=ALPHA_F, alpha_rec=ALPHA_R,
                   results=results, fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateP] {'PASS' if not fails else 'FAIL'} -> {out}")
    for f in fails:
        print(f"[gateP] FAIL: {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
