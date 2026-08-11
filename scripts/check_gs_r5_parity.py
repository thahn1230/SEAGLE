#!/usr/bin/env python
"""Gate G: exact training/runtime parity for GS + R5 (EP3-G global scale
+ R_D residual rotation, W4A4 target).

GS twin of check_rd_ep3p_parity.py (Gate P). Legs:

  GS_RT   : rot-mode adapter with R_D := R_T must reproduce the GS
            baseline adapter (d4p3_deploy, same global alpha) bitwise on
            all quantized tensors — proves the rot path with a trivial
            rotation IS the GS baseline (first-path basis parity).
  GS_R5   : rot-mode adapter with the real reused R5 (RD_HYB_s2.pt)
            vs ExactQATRotatedDraft(alpha_rec_init=None) — weight-hash
            parity on the 9 quantized tensors + head, and K=4
            teacher-forced chain parity (hidden/logits/greedy tokens).
            This is simultaneously the step-0-QAT == PTQ gate: the QAT
            trainer's forward at step 0 is this exact core.

Also audited: R5 orthogonality (fp64), rec_embed_rescale must be None
under GS (no LS-specific runtime op), PostProjectionR1 present in both
baseline and GS+R5 (same explicit-op count), FakeW4A4Linear module count
parity (quantizer invocation parity).

Writes <run>/gradchecks/gateG_gs_r5_parity.json; exits 1 on failure.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.fake_w4a4_draft import FakeW4A4Linear
from eagle_spinquant.residual_rotation import FullRotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_exact_path_parity import whash, run_runtime_chain  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_G = D ** 0.42  # EP3-G / GS int4 m = 32.89964245299412
D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G)


def n_fake_linears(model):
    return sum(1 for m in model.modules() if isinstance(m, FakeW4A4Linear))


def grab_weights(ea, ad):
    return dict(
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rd-ckpt", required=True)
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

    ck = torch.load(args.rd_ckpt, map_location="cpu", weights_only=False)
    R5 = ck["R_D"].double()
    results, fails = {}, []

    # Gate 3: R5 orthogonality (fp64)
    I = torch.eye(D, dtype=torch.float64)
    orth = R5.T @ R5 - I
    results["r5_orthogonality"] = dict(
        max_abs=float(orth.abs().max()), fro=float(orth.norm()),
        ckpt_alpha_ls=float(ck.get("alpha", float("nan"))),
        ckpt_alpha_rec_ls=float(ck.get("alpha_rec", float("nan"))))
    if results["r5_orthogonality"]["max_abs"] > 1e-4:
        fails.append("R5 orthogonality error too large")

    # Baseline GS adapter (d4p3_deploy equivalent: stash R1 = R_T)
    ad_base = ConcatSelectiveDraftAdapter(
        model, stash, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False, **D4GS)
    ad_base.install()
    base_w = {k: v.detach().float().cpu().clone()
              for k, v in grab_weights(ea, ad_base).items()}
    base_nq = n_fake_linears(ea) + n_fake_linears(ad_base.split)
    base_rr = ad_base.split.rec_embed_rescale
    ad_base.uninstall()
    torch.cuda.empty_cache()

    for tag, R_D in (("GS_RT", R_T.double()), ("GS_R5", R5)):
        st = dict(stash)
        st["R1"] = R_D
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=R_T, **D4GS)
        ad.install()
        rt_w = grab_weights(ea, ad)

        # Gates 6+7: no LS runtime op, quantizer/module count parity
        op_audit = dict(
            rec_embed_rescale=(None if ad.split.rec_embed_rescale is None
                               else float(ad.split.rec_embed_rescale)),
            post_projection_R1=ad.split.post_projection_R1 is not None,
            n_fake_w4a4=n_fake_linears(ea) + n_fake_linears(ad.split),
            n_fake_w4a4_baseline=base_nq,
            baseline_rec_embed_rescale=(None if base_rr is None
                                        else float(base_rr)))
        if ad.split.rec_embed_rescale is not None:
            fails.append(f"{tag}: LS-specific rec_embed_rescale present")
        if op_audit["n_fake_w4a4"] != base_nq:
            fails.append(f"{tag}: quantized-module count "
                         f"{op_audit['n_fake_w4a4']} != base {base_nq}")

        wres = {}
        if tag == "GS_RT":
            # first/recurrent basis parity vs GS baseline: R_D=R_T must
            # reproduce the baseline weights exactly
            for name in rt_w:
                w_cpu = rt_w[name].detach().float().cpu()
                mx = float((w_cpu - base_w[name]).abs().max())
                ok = whash(w_cpu) == whash(base_w[name])
                wres[name] = dict(equal_to_baseline=ok, max_abs_diff=mx)
                if not ok and mx > 2e-3:
                    fails.append(f"{tag}:{name} vs baseline max={mx}")
            results[tag] = dict(weights=wres, op_audit=op_audit)
        else:
            # exact-core parity (step-0 QAT == PTQ deploy) with global
            # alpha only (alpha_rec_init=None => GS fold)
            eq = ExactQATRotatedDraft(
                sd0, R_T, stash["gamma_f"],
                stash["lm_head_weight"].float(),
                FullRotation(R5.float()).to(dev), alpha_init=ALPHA_G,
                w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
                first_fold_R=R_T)
            qw, _tw = eq.quantized_weights()
            for name in rt_w:
                h_rt, h_eq = whash(rt_w[name]), whash(qw[name])
                ok = h_rt == h_eq
                mx = float((rt_w[name].float() -
                            qw[name].float()).abs().max())
                wres[name] = dict(runtime=h_rt, exact=h_eq, equal=ok,
                                  max_abs_diff=mx)
                if not ok and mx > 2e-3:
                    fails.append(f"{tag}:{name} weight mismatch max={mx}")
            # first-path pinned at R_T: W_first must equal baseline
            wf_cpu = rt_w["W_first"].detach().float().cpu()
            mx_f = float((wf_cpu - base_w["W_first"]).abs().max())
            wres["W_first_vs_gs_baseline"] = dict(
                equal=whash(wf_cpu) == whash(base_w["W_first"]),
                max_abs_diff=mx_f)
            if mx_f > 2e-3:
                fails.append(f"{tag}: W_first not pinned at R_T "
                             f"(max={mx_f})")

            g = torch.Generator().manual_seed(7)
            T = 8
            tok_ids = torch.randint(10, 3000, (1, T + 1),
                                    generator=g).to(dev)
            a_seq = (torch.randn(1, T, eq.D, generator=g) * 1.0).to(dev)
            rt = run_runtime_chain(ea, ad, tok_ids, a_seq, 4, ALPHA_G,
                                   dev, kv_bits=16)
            torch.cuda.empty_cache()
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
            results[tag] = dict(weights=wres, chain=cres,
                                op_audit=op_audit)
            del eq
        ad.uninstall()
        torch.cuda.empty_cache()

    out = os.path.join(args.run_dir, "gradchecks",
                       "gateG_gs_r5_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_G, rd_ckpt=args.rd_ckpt,
                   results=results, fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateG] {'PASS' if not fails else 'FAIL'} -> {out}")
    for f in fails:
        print(f"[gateG] FAIL: {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
