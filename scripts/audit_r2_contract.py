#!/usr/bin/env python
"""Executable audit of the draft R2 contract (GS/R2 study section 5).

Prints and asserts the DISCOVERED contract of the baseline draft R2 (R2_B):

  1. provenance   : deterministic seed-0 Haar-QR orthogonal (QR of Gaussian),
                    regenerated at install — NOT loaded from R.bin, NOT the
                    target's per-layer SpinQuant R2, NOT a Hadamard;
  2. shape        : one [128,128] fp64 matrix, SHARED across all 32 heads of
                    the single draft decoder layer (layers.0);
  3. folding      : per head h, V_h' = R2 @ V_h (v_proj output rows),
                    O_h' = O_h @ R2^T (o_proj input cols), i.e.
                    v' = blkdiag(R2 x32) v and o' = o blkdiag(R2)^T,
                    so o'(R2 ctx) = o(ctx);
  4. FP gauge     : the V/O pair conjugation preserves the attention block
                    output exactly in fp64 (function-preserving basis change);
  5. quant order  : v/o weights are W4-quantized AFTER the R2 fold; the
                    o_proj input activation is A4-quantized in the R2 basis
                    (quantize-then-GEMM);
  6. F0 fold      : R2 lives only in folded weights — no runtime R2 operator.

Levels: --level cpu (default, no GPU/model needed) runs 1-5 structurally;
--level model additionally builds the deployed adapter on a real
target+draft and verifies 2/5/6 on the installed modules (needs R.bin).

Writes <run-dir>/audit/r2_contract_audit.json; exit 1 on any failure.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant import spinquant_draft as spd

NH, HD, D = 32, 128, 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--level", default="cpu", choices=["cpu", "model"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    checks = {}
    fails = []

    def check(name, ok, detail):
        checks[name] = dict(ok=bool(ok), detail=detail)
        print(f"[r2audit] {'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            fails.append(name)

    # ---- 1. provenance ----------------------------------------------------
    R2 = fq.baseline_r2(0)
    g = torch.Generator().manual_seed(0)
    R2_re = torch.linalg.qr(torch.randn(HD, HD, generator=g,
                                        dtype=torch.float64))[0]
    sha = hashlib.sha256(R2.numpy().tobytes()).hexdigest()
    check("provenance_seed0_qr", torch.equal(R2, R2_re),
          f"R2_B = QR(randn(128,128, seed 0, fp64)); sha256={sha[:16]}")
    I = torch.eye(HD, dtype=torch.float64)
    oe = float((R2.t() @ R2 - I).norm())
    check("orthogonality", oe < 1e-12, f"||R2^T R2 - I||_F = {oe:.2e}")
    # a Hadamard-type matrix has all |elements| = 1/sqrt(128)
    aM, am = float(R2.abs().max()), float(R2.abs().min())
    had = 1.0 / HD ** 0.5
    check("not_hadamard",
          abs(aM - had) > 1e-6 or abs(am - had) > 1e-6,
          f"|elem| in [{am:.4f},{aM:.4f}] vs Hadamard {had:.4f} — Haar-QR, "
          f"not Hadamard (the code comment previously mislabeled this)")

    # ---- 2/3. shape, granularity, fold equations --------------------------
    check("shape", tuple(R2.shape) == (HD, HD), f"shape={tuple(R2.shape)}")
    g2 = torch.Generator().manual_seed(1)
    v_w = torch.randn(NH * HD, D, generator=g2, dtype=torch.float64)
    o_w = torch.randn(D, NH * HD, generator=g2, dtype=torch.float64)
    v2, o2 = spd.conjugate_v_o(v_w.clone(), o_w.clone(), R2)
    v_ref = torch.cat([R2 @ v_w[h * HD:(h + 1) * HD] for h in range(NH)], 0)
    o_ref = torch.cat([o_w[:, h * HD:(h + 1) * HD] @ R2.t()
                       for h in range(NH)], 1)
    dv = float((v2 - v_ref).abs().max())
    do = float((o2 - o_ref).abs().max())
    check("fold_equations", dv == 0.0 and do == 0.0,
          f"conjugate_v_o == per-head V_h'=R2@V_h / O_h'=O_h@R2^T "
          f"(dv={dv}, do={do}); ONE matrix shared across {NH} heads")
    B = torch.block_diag(*([R2] * NH))
    dvb = float((v2 - (B @ v_w)).abs().max())
    dob = float((o2 - (o_w @ B.t())).abs().max())
    check("blockdiag_equiv", dvb < 1e-12 and dob < 1e-12,
          f"v'=blkdiag(R2)v, o'=o blkdiag(R2)^T (dv={dvb:.1e}, "
          f"do={dob:.1e})")

    # ---- 4. FP gauge equivalence at the attention-block level (fp64) ------
    T = 5
    att = torch.softmax(torch.randn(NH, T, T, generator=g2,
                                    dtype=torch.float64), -1)
    x = torch.randn(T, D, generator=g2, dtype=torch.float64)
    def attn_out(vw, ow):
        vh = (vw @ x.t()).reshape(NH, HD, T)              # per-head V x
        ctx = torch.einsum("hts,hds->hdt", att, vh)       # att @ v
        ctx = ctx.reshape(NH * HD, T)
        return ow @ ctx                                    # o_proj
    y_base = attn_out(v_w, o_w)
    y_conj = attn_out(v2, o2)
    dg = float((y_base - y_conj).abs().max() / y_base.abs().max())
    check("fp_gauge_block", dg < 1e-12,
          f"attention-block output invariant under V/O conjugation "
          f"(rel max dev {dg:.2e}) — R2 is a function-preserving gauge")
    # and for an ARBITRARY second orthogonal R2' (gauge freedom, not a
    # property of the specific seed-0 draw)
    g3 = torch.Generator().manual_seed(99)
    R2p = torch.linalg.qr(torch.randn(HD, HD, generator=g3,
                                      dtype=torch.float64))[0]
    v3, o3 = spd.conjugate_v_o(v_w.clone(), o_w.clone(), R2p)
    dg2 = float((y_base - attn_out(v3, o3)).abs().max()
                / y_base.abs().max())
    check("fp_gauge_any_orthogonal", dg2 < 1e-12,
          f"holds for an arbitrary orthogonal R2' (rel dev {dg2:.2e})")

    # ---- 5. quantization boundary: W4 AFTER fold, A4 after R2 basis -------
    v16 = v2.half()
    q_after = fq._weight_fake_quant(v16.clone(), 4)
    lin = fq.FakeW4A4Linear(v16.clone(), None, "audit_v", w_bits=4,
                            a_bits=4)
    dqa = float((lin.w_fake.float() - q_after.float()).abs().max())
    q_before = spd.conjugate_v_o(
        fq._weight_fake_quant(v_w.half().clone(), 4).double(),
        o_w.clone(), R2)[0].half()
    d_wrong = float((lin.w_fake.float() - q_before.float()).abs().max())
    check("w4_after_fold", dqa == 0.0 and d_wrong > 1e-4,
          f"FakeW4A4Linear quantizes the FOLDED weight (dev {dqa}); "
          f"quantize-then-fold differs by {d_wrong:.2e} — boundary is "
          f"fold -> W4")
    # o_proj-style module on the folded o-weight; input arrives in the
    # R2 basis (ctx' = blkdiag(R2) ctx)
    lin_o = fq.FakeW4A4Linear(o2.half(), None, "audit_o", w_bits=4,
                              a_bits=4)
    ctx = torch.randn(3, NH * HD, generator=g2).half()
    ctx_rot = (ctx.double() @ B.t()).half()          # R2-basis input
    n0 = lin_o.n_act_quant
    y = lin_o(ctx_rot)
    aq = fq._act_quantizer(4)
    aq.find_params(ctx_rot)
    y_ref = torch.nn.functional.linear(aq(ctx_rot), lin_o.w_fake)
    aq.free()
    d_q2g = float((y - y_ref).abs().max())
    # basis-dependence: quantizing in the R2 basis differs from rotating
    # an already-quantized pre-R2 activation (the A4 grid is not gauge-
    # invariant) — this is what makes R2 quantization-relevant at all
    aq2 = fq._act_quantizer(4)
    aq2.find_params(ctx)
    ctx_q_pre = aq2(ctx)
    aq2.free()
    aq3 = fq._act_quantizer(4)
    aq3.find_params(ctx_rot)
    ctx_q_post = aq3(ctx_rot)
    aq3.free()
    d_grid = float((ctx_q_post.double()
                    - (ctx_q_pre.double() @ B.t())).abs().max())
    check("a4_after_r2_basis",
          lin_o.n_act_quant == n0 + 1 and d_q2g == 0.0 and d_grid > 1e-4,
          f"o_proj input A4-quantized in the folded (R2) basis then fp16 "
          f"GEMM: forward == linear(A4(x), w_fake) exactly (d={d_q2g}); "
          f"A4 grid is basis-dependent: Q(x R2^T) != Q(x) R2^T "
          f"(d={d_grid:.4f})")

    # ---- 6/model. installed-adapter checks --------------------------------
    if args.level == "model":
        from eagle_spinquant import experiment, study
        from eagle_spinquant.concat_selective_projection import (
            ConcatSelectiveDraftAdapter)
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        rr = cfg.get("paths", {}).get("rotations_root")
        dcfg = json.load(open(os.path.join(paths["draft_path"],
                                           "config.json")))
        check("single_ar_layer", dcfg.get("num_hidden_layers", 1) == 1,
              f"draft num_hidden_layers = {dcfg.get('num_hidden_layers')}")
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], "full", "learned_chat_w4a4kv16",
            "w4a4", 0, device=args.device, rotations_root=rr)
        ea = model.ea_layer
        sd0 = {k: v.detach().cpu().clone() for k, v in
               ea.state_dict().items()}
        from eagle_spinquant.pmg_configs import EP3G
        ad = ConcatSelectiveDraftAdapter(
            model, stash, args.device, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            embed_scale_alpha=float(EP3G["int4"]),
            quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True)
        ad.install()
        # deployed v_proj w_fake equals W4(R2-fold(R1-conj(v)))
        conv, meta = fq.build_spinquant_w4a4_draft_state(
            sd0, stash["R1"].detach().cpu().double(),
            stash["gamma_f"].detach().cpu().double())
        vq_ref = fq._weight_fake_quant(
            conv["layers.0.self_attn.v_proj.weight"]
            .to(ea.layers[0].self_attn.v_proj.w_fake.dtype)
            .to(args.device), 4)
        dv_dep = float((ea.layers[0].self_attn.v_proj.w_fake
                        - vq_ref).abs().max())
        check("deployed_v_fold_quant", dv_dep == 0.0,
              f"installed v_proj.w_fake == W4(R2-folded weight), "
              f"dev={dv_dep}; r2_source={meta['r2_source']} "
              f"sha={meta['r2_sha']}")
        # F0: no module holds an R2 tensor / applies it at runtime
        r2_mods = [n for n, m in ea.named_modules()
                   for bn, b in getattr(m, "_buffers", {}).items()
                   if b is not None and tuple(b.shape) == (HD, HD)]
        check("f0_no_runtime_r2", len(r2_mods) == 0,
              f"no [128,128] runtime buffer in the installed draft "
              f"(modules scanned: {len(list(ea.named_modules()))}); "
              f"online ops are only R4 Hadamard (down_proj) and "
              f"PostProjectionR1 — R2 is F0 offline-folded")
        ad.uninstall()

    out = os.path.join(args.run_dir, "audit", "r2_contract_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(level=args.level, checks=checks, fails=fails,
                   r2_sha256=sha,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[r2audit] {'PASS' if not fails else 'FAIL'} -> {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
