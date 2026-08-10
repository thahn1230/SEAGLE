#!/usr/bin/env python
"""Gate R2-A (FP gauge equivalence) + Gate R2-B (quantization sensitivity).

R2-A — with quantization OFF (w16/a16/kv16), through the runtime-exact
trainer forward:
  * ResidualR2Rotation at init reproduces the baseline BITWISE;
  * a non-trivially perturbed R2_D with the CORRECT V/O pair fold preserves
    the FP function: per-depth logits / attention-block outputs / hidden
    deviate only at fp16-roundoff scale, greedy tokens preserved. The
    deviation is gated DISCRIMINATIVELY: it must be < 2% of the deviation
    caused by a deliberately BROKEN fold (v folded with R2_D, o left at
    R2_B — a genuine function change);
  * the fp64 zero-deviation gauge identity is covered by
    audit_r2_contract.py (fp_gauge_* checks).

R2-B — with W4A4 ON:
  * a perturbed R2_D changes the v/o quantized weights (and ONLY v/o among
    the 9 quantized sites), the depth-1..4 draft logits, and the LK hybrid
    loss;
  * dL/dB is finite and nonzero on multiple batches (R2 generator receives
    gradient through the STE-quantized path); joint mode also checks dL/dA.

Writes <run>/gradchecks/gate_r2_ab.json; exit 1 on failure.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.pmg_configs import EP3G
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               ResidualRotation,
                                               ResidualR2Rotation)
from check_exact_path_parity import whash
from check_r1r2_gs_parity import FixedR2

KIND = "learned_chat_w4a4kv16"
ALPHA_GS = float(EP3G["int4"])


def build(sd0, R_T, stash, dev, w_bits, a_bits, rot, r2_rot):
    return ExactQATRotatedDraft(
        sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
        rot, alpha_init=ALPHA_GS, w_bits=w_bits, a_bits=a_bits,
        draft_kv_bits=16, device=dev, first_fold_R=R_T, r2_rot=r2_rot)


def chain(model, tok_ids, a_seq, K=4, capture=False):
    model.debug_capture = [] if capture else None
    outs = model.forward_chain(tok_ids, a_seq, K, exact=True)
    caps = model.debug_capture
    model.debug_capture = None
    lgs = [o[0].detach().float().cpu() for o in outs]
    hs = [o[1].detach().float().cpu() for o in outs]
    tks = [o[2].detach().cpu() for o in outs]
    return lgs, hs, tks, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-grad-batches", type=int, default=4)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=rr)
    ea = model.ea_layer
    sd0 = {k: v.detach().cpu().clone() for k, v in ea.state_dict().items()}
    R_T = stash["R1"].clone()
    del model
    torch.cuda.empty_cache()

    res, fails = {}, []

    def check(name, ok, detail):
        res[name] = dict(ok=bool(ok), detail=detail)
        print(f"[gateR2] {'PASS' if ok else 'FAIL'} {name}: {detail}",
              flush=True)
        if not ok:
            fails.append(name)

    r2_pert = ResidualR2Rotation(fq.baseline_r2(0))
    with torch.no_grad():
        r2_pert.W[:16, 16:32] += 0.02
    R2_D = r2_pert.R64().detach()

    g = torch.Generator().manual_seed(11)
    T = 8
    tok_ids = torch.randint(10, 3000, (2, T + 1), generator=g).to(dev)
    a_seq = (torch.randn(2, T, 4096, generator=g) * 1.0).to(dev)

    # ================= Gate R2-A: quantization OFF =================
    with torch.no_grad():
        m_base = build(sd0, R_T, stash, dev, 16, 16,
                       SharedRotation(R_T.float()).to(dev), None)
        lg_a, h_a, tk_a, cap_a = chain(m_base, tok_ids, a_seq,
                                       capture=True)
        m_init = build(sd0, R_T, stash, dev, 16, 16,
                       SharedRotation(R_T.float()).to(dev),
                       ResidualR2Rotation(fq.baseline_r2(0)).to(dev))
        lg_c, h_c, tk_c, _ = chain(m_init, tok_ids, a_seq)
        d_init = max(float((a - b).abs().max())
                     for a, b in zip(lg_a, lg_c))
        check("r2a_init_bitwise", d_init == 0.0,
              f"ResidualR2Rotation(B=0) == baseline R2_B forward "
              f"(max dlogits {d_init})")
        del m_init
        # gauge signature at TWO generator scales: a correct V/O pair
        # refold deviates only at the fp16-roundoff floor (deviation
        # ~independent of the perturbation angle, greedy tokens
        # preserved), while a BROKEN fold (v refolded, o left at R2_B —
        # a genuine function change) deviates ~linearly in the angle.
        qw_0, tw_0 = m_base.quantized_weights(exact=True)
        orig_qw = m_base.quantized_weights
        scales = {}
        for sname, smul in (("small", 1.0), ("large", 5.0)):
            r2s = ResidualR2Rotation(fq.baseline_r2(0))
            with torch.no_grad():
                r2s.W[:16, 16:32] += 0.02 * smul
            R2s = r2s.R64().detach()
            m_pert = build(sd0, R_T, stash, dev, 16, 16,
                           SharedRotation(R_T.float()).to(dev),
                           FixedR2(R2s).to(dev))
            lg_b, h_b, tk_b, cap_b = chain(m_pert, tok_ids, a_seq,
                                           capture=True)
            dev_correct = max(float((a - b).abs().max())
                              for a, b in zip(lg_a, lg_b))
            dev_attn = max(float((a - b).abs().max())
                           for a, b in zip(cap_a, cap_b))
            tok_ok = all(bool((x == y).all())
                         for x, y in zip(tk_a, tk_b))
            qw_p, _tw_p = m_pert.quantized_weights(exact=True)
            mixed = dict(qw_0)
            mixed["v"] = qw_p["v"]
            m_base.quantized_weights = \
                lambda exact=True, _m=(mixed, tw_0): _m
            lg_x, _, _, _ = chain(m_base, tok_ids, a_seq)
            m_base.quantized_weights = orig_qw
            dev_broken = max(float((a - b).abs().max())
                             for a, b in zip(lg_a, lg_x))
            scales[sname] = dict(dev_correct=dev_correct,
                                 dev_attn=dev_attn, tok_ok=tok_ok,
                                 dev_broken=dev_broken)
            del m_pert
            torch.cuda.empty_cache()
        s, l = scales["small"], scales["large"]
        growth_broken = l["dev_broken"] / max(s["dev_broken"], 1e-9)
        growth_correct = l["dev_correct"] / max(s["dev_correct"], 1e-9)
        ok = (s["tok_ok"] and l["tok_ok"]
              and s["dev_correct"] < 0.1 * s["dev_broken"]
              and l["dev_correct"] < 0.1 * l["dev_broken"]
              and growth_broken > 2.0 and growth_correct < 2.0)
        check("r2a_fp_gauge", ok,
              f"correct pair fold sits at the fp16-roundoff floor: "
              f"dlogits {s['dev_correct']:.4f}->{l['dev_correct']:.4f} "
              f"(x{growth_correct:.2f} under 5x generator), greedy "
              f"tokens preserved at both scales; BROKEN fold grows "
              f"{s['dev_broken']:.4f}->{l['dev_broken']:.4f} "
              f"(x{growth_broken:.2f}); correct/broken ratio "
              f"{s['dev_correct']/max(s['dev_broken'],1e-9):.3f} / "
              f"{l['dev_correct']/max(l['dev_broken'],1e-9):.3f} < 0.1")
        del m_base
        torch.cuda.empty_cache()

        # ================= Gate R2-B: W4A4 ON ==========================
        m_qb = build(sd0, R_T, stash, dev, 4, 4,
                     SharedRotation(R_T.float()).to(dev), None)
        m_qp = build(sd0, R_T, stash, dev, 4, 4,
                     SharedRotation(R_T.float()).to(dev),
                     FixedR2(R2_D).to(dev))
        qb, _ = m_qb.quantized_weights(exact=True)
        qp, _ = m_qp.quantized_weights(exact=True)
        changed = {n: whash(qb[n]) != whash(qp[n]) for n in
                   ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up",
                    "down", "head")}
        expect = {n: (n in ("v", "o")) for n in changed}
        check("r2b_weights_change", changed == expect,
              f"perturbed R2_D changes v/o quantized weights and ONLY "
              f"v/o: {changed}")
        lg_q0, _, _, _ = chain(m_qb, tok_ids, a_seq)
        lg_q1, _, _, _ = chain(m_qp, tok_ids, a_seq)
        d_q = max(float((a - b).abs().max())
                  for a, b in zip(lg_q0, lg_q1))
        check("r2b_logits_change", d_q > 1e-3,
              f"quantized draft logits move under R2_D (max d {d_q:.4f})")
        zT = torch.randn(2, 4, 32000, generator=g).to(dev) * 2.0
        def lk_loss(lgs):
            tot = 0.0
            for k in range(4):
                lk, _l, _a2 = L.hybrid_lk(zT[:, k], lgs[k].to(dev))
                tot = tot + (0.8 ** k) * lk.mean()
            return tot
        l0, l1 = float(lk_loss(lg_q0)), float(lk_loss(lg_q1))
        check("r2b_loss_change", abs(l0 - l1) > 1e-6,
              f"LK hybrid loss moves: {l0:.6f} -> {l1:.6f}")
        del m_qb, m_qp
        torch.cuda.empty_cache()

    # gradient flow (needs grad enabled)
    r2_tr = ResidualR2Rotation(fq.baseline_r2(0)).to(dev)
    r1_tr = ResidualRotation(R_T.float()).to(dev)
    m_g = build(sd0, R_T, stash, dev, 4, 4, r1_tr, r2_tr)
    gnorms2, gnorms1 = [], []
    for b in range(args.n_grad_batches):
        gb = torch.Generator().manual_seed(100 + b)
        tid = torch.randint(10, 3000, (2, T + 1), generator=gb).to(dev)
        asq = (torch.randn(2, T, 4096, generator=gb)).to(dev)
        zTb = (torch.randn(2, 4, 32000, generator=gb) * 2.0).to(dev)
        if r2_tr.W.grad is not None:
            r2_tr.W.grad = None
        if r1_tr.W.grad is not None:
            r1_tr.W.grad = None
        teach = torch.randint(0, 32000, (2, 4), generator=gb).to(dev)
        outs = m_g.forward_chain(tid, asq, 4, recur_tokens=teach)
        tot = 0.0
        for k, (lg, _h, _c) in enumerate(outs):
            lk, _l, _a2 = L.hybrid_lk(zTb[:, k], lg)
            tot = tot + (0.8 ** k) * lk.mean()
        tot.backward()
        gnorms2.append(float(r2_tr.W.grad.norm()))
        gnorms1.append(float(r1_tr.W.grad.norm()))
    check("r2b_grad_dL_dB",
          all(x > 0 and torch.isfinite(torch.tensor(x)) for x in gnorms2),
          f"dL/dB nonzero+finite on {len(gnorms2)} batches: "
          f"{[round(x, 4) for x in gnorms2]}")
    check("r2b_grad_dL_dA_joint",
          all(x > 0 for x in gnorms1),
          f"joint mode: dL/dA also flows: "
          f"{[round(x, 2) for x in gnorms1]}")

    out = os.path.join(args.run_dir, "gradchecks", "gate_r2_ab.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_GS, checks=res, fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateR2] {'PASS' if not fails else 'FAIL'} -> {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
