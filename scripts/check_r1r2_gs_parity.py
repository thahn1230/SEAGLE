#!/usr/bin/env python
"""Gate GS-R1 + Gate R2-C: trainer<->runtime parity under the GS fold.

GS (Global Scaling, legacy EP3-G): ONE global m = D^beta_GS with the
canonical T4 beta_GS = 0.42 (pmg_configs.EP3G['int4']), no pathwise
alpha_rec, no recurrent e-slice rescale. The trainer uses its legacy
fold branch (fp16 fold divided by the full-precision python scalar on
CPU in exact mode) which literally replicates the deploy adapter's op.

Four legs, all W4A4 KV16:

  BASE_gs   : R_D = R_T,          R2 = R2_B          (baseline parity)
  R1D_gs    : perturbed R_D,      R2 = R2_B          (Gate GS-R1)
  R2D_gs    : R_D = R_T,          perturbed R2_D     (Gate R2-C)
  JOINT_gs  : perturbed R_D,      perturbed R2_D     (Gate R2-C joint)

Per leg (Gate-D/P contract): 10 fp16 quantized tensors bytewise-hash
equal (FAIL if hash differs AND max_abs_diff > 2e-3) + K=4 teacher-forced
chain with identical logits (<= 5e-2) and greedy tokens.

Cross-leg structural checks:
  * W_first hash IDENTICAL across all four legs — the first target->draft
    interface stays pinned to R1_T; neither R_D nor R2_D may move it.
  * R2D_gs vs BASE_gs: ONLY v/o hashes differ (R2 is attention-local).
  * R1D_gs vs BASE_gs: W_rec differs (R_D moves the draft-internal basis).

Writes <run>/gradchecks/gate_gs_r1r2_parity.json; exit 1 on failure.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.pmg_configs import EP3G
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               FullRotation,
                                               ResidualRotation,
                                               ResidualR2Rotation)
from check_exact_path_parity import whash, run_runtime_chain

KIND = "learned_chat_w4a4kv16"
ALPHA_GS = float(EP3G["int4"])          # 4096**0.42 = 32.8996... (T4 GS)


class FixedR2(nn.Module):
    """Frozen R2_D holder so trainer and runtime consume the SAME fp64
    matrix (recomputing the Cayley on different devices differs in ulps)."""
    trainable = False

    def __init__(self, R2_64):
        super().__init__()
        self.register_buffer("R2_64f", R2_64.double())

    def R(self, dtype=torch.float32):
        return self.R2_64f.to(dtype)


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

    # perturbed R_D (identical construction to Gate-P's RESID leg)
    rr_mod = ResidualRotation(R_T.float()).to(dev)
    with torch.no_grad():
        rr_mod.W[:64, 64:128] += 0.02
    R_D_pert = rr_mod.R().detach()
    # perturbed R2_D (residual Cayley around R2_B, frozen fp64 product)
    r2_mod = ResidualR2Rotation(fq.baseline_r2(0)).to(dev)
    with torch.no_grad():
        r2_mod.W[:16, 16:32] += 0.02
    R2_D_pert = r2_mod.R64().detach().cpu()

    D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
                quant_ar="fake_w4a4", ar_r2r4=True,
                embed_scale_alpha=ALPHA_GS)
    LEGS = {
        "BASE_gs":  dict(r1d=False, r2d=False),
        "R1D_gs":   dict(r1d=True,  r2d=False),
        "R2D_gs":   dict(r1d=False, r2d=True),
        "JOINT_gs": dict(r1d=True,  r2d=True),
    }
    results, fails = {}, []
    leg_whash = {}
    for tag, leg in LEGS.items():
        rot = (FullRotation(R_D_pert) if leg["r1d"]
               else SharedRotation(R_T.float()))
        r2_rot = FixedR2(R2_D_pert) if leg["r2d"] else None
        R_D = rot.R().detach().cpu().double()
        st = dict(stash)
        st["R1"] = R_D
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=(R_T if leg["r1d"] else None),
            ar_r2_override=(R2_D_pert if leg["r2d"] else None), **D4GS)
        ad.install()
        eq = ExactQATRotatedDraft(
            sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
            rot.to(dev), alpha_init=ALPHA_GS, w_bits=4, a_bits=4,
            draft_kv_bits=16, device=dev, first_fold_R=R_T,
            r2_rot=(r2_rot.to(dev) if r2_rot is not None else None))
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
        leg_whash[tag] = {}
        for name in rt_w:
            h_rt, h_eq = whash(rt_w[name]), whash(qw[name])
            leg_whash[tag][name] = h_rt
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
        rt = run_runtime_chain(ea, ad, tok_ids, a_seq, 4, ALPHA_GS, dev,
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

    # ---- cross-leg structural checks -------------------------------------
    struct = {}
    wf = {t: leg_whash[t]["W_first"] for t in LEGS}
    struct["w_first_pinned"] = dict(
        ok=len(set(wf.values())) == 1, hashes=wf,
        claim="first T->D interface fold identical across all legs "
              "(pinned to R1_T; untouched by R_D and R2_D)")
    if len(set(wf.values())) != 1:
        fails.append(f"W_first not pinned: {wf}")
    only_vo = {n: (leg_whash["R2D_gs"][n] != leg_whash["BASE_gs"][n])
               for n in leg_whash["BASE_gs"]}
    expect = {n: (n in ("v", "o")) for n in only_vo}
    struct["r2_touches_only_vo"] = dict(
        ok=only_vo == expect, changed=only_vo,
        claim="perturbed R2_D changes ONLY the v/o folds")
    if only_vo != expect:
        fails.append(f"R2D leg changed unexpected tensors: {only_vo}")
    r1_moves = leg_whash["R1D_gs"]["W_rec"] != leg_whash["BASE_gs"]["W_rec"]
    struct["r1d_moves_recurrent"] = dict(
        ok=bool(r1_moves),
        claim="perturbed R_D changes the recurrent fold (draft-internal "
              "basis) while W_first stays pinned")
    if not r1_moves:
        fails.append("R1D leg did not change W_rec")

    out = os.path.join(args.run_dir, "gradchecks",
                       "gate_gs_r1r2_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_GS, results=results, structural=struct,
                   fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateGS] {'PASS' if not fails else 'FAIL'} "
          f"(alpha_GS={ALPHA_GS:.6f}) -> {out}")
    for f in fails:
        print(f"[gateGS] FAIL: {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
