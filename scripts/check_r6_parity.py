#!/usr/bin/env python
"""Gate R6: correctness gates for the draft-aware attention V/O rotation
R6 = R2_base @ C(B) (shared [128,128], baseline granularity), stacked on
the frozen R5 draft residual rotation under the GS deployment contract.

TEST-R6-1  orthogonality of R2_base, identity-init R6 and perturbed R6.
TEST-R6-2  identity init: adapter(r2_override=R2_base@C(0)) bitwise ==
           adapter(default seeded R2) on all quantized tensors.
TEST-R6-3  FP gauge/fold equivalence (fp64, quant OFF): with the actual
           conjugate_v_o fold, O'(A ⊗ V'x) == O(A ⊗ Vx) for a perturbed
           R6 — function-preserving basis change, explicit==folded.
TEST-R6-4  fold-before-quant: deployed v/o w_fake equals an independent
           re-quantization of the fp64-folded fp16 weight (order:
           fold R5/R6 -> quantize -> forward).
TEST-R6-5  no runtime R6 operator: quantized-module count and runtime-op
           audit identical to the R5-only baseline.
Gate R2-B  quantization sensitivity: perturbed R6 changes v/o quantized
           weights and K=4 chain logits; dL/dB > 0 on 3 batches.
Gate R2-C  trainer/runtime parity: exact core (rot2=R6) vs deploy
           adapter (r2_override=R6) — weight hashes + K=4 chain.

Writes <run>/gradchecks/gateR6_parity.json; exits 1 on failure.
"""
import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant import spinquant_draft as spd
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.fake_w4a4_draft import FakeW4A4Linear
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               ResidualRotation, cayley)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_exact_path_parity import whash, run_runtime_chain  # noqa: E402

KIND = "learned_chat_w4a4kv16"
D, NH, HD = 4096, 32, 128
ALPHA_G = D ** 0.42
D4GS = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True,
            embed_scale_alpha=ALPHA_G)


def r2_base_matrix():
    g = torch.Generator().manual_seed(0)
    return torch.linalg.qr(torch.randn(HD, HD, generator=g,
                                       dtype=torch.float64))[0]


def grab(ea, ad):
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


def n_fake(ea, ad):
    return (sum(1 for m in ea.modules() if isinstance(m, FakeW4A4Linear))
            + sum(1 for m in ad.split.modules()
                  if isinstance(m, FakeW4A4Linear)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--r5-ckpt", required=True)
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
    R5 = torch.load(args.r5_ckpt, map_location="cpu",
                    weights_only=False)["R_D"].double()

    R2B = r2_base_matrix()
    gA = torch.Generator().manual_seed(7)
    Apert = torch.randn(HD, HD, generator=gA, dtype=torch.float64) * 0.02
    Apert = Apert - Apert.t()
    # fp32 round-trip: the trainer saves R6 as float32 and the deploy
    # path loads that fp32 tensor and upcasts (.double()) — the gate
    # must exercise the SAME precision path on both sides
    R6P = R2B @ cayley(Apert)
    R6P32 = R6P.float()
    R6P = R6P32.double()

    results, fails = {}, []

    # ---- TEST-R6-1: orthogonality ----
    I = torch.eye(HD, dtype=torch.float64)
    orth = {}
    for nm, M in (("R2_base", R2B), ("R6_identity", R2B @ cayley(
            torch.zeros(HD, HD, dtype=torch.float64))),
            ("R6_perturbed", R6P)):
        e = float((M.t() @ M - I).abs().max())
        orth[nm] = e
        # fp32 round-trip (the saved-checkpoint precision) bounds the
        # orthogonality error at ~1e-8; 1e-6 is the validated tolerance
        if e > 1e-6:
            fails.append(f"TEST-R6-1 {nm} orth {e}")
    results["test_r6_1_orthogonality_maxabs"] = orth

    # ---- TEST-R6-3: FP gauge equivalence on the ACTUAL fold ----
    v0 = sd0["layers.0.self_attn.v_proj.weight"].double()
    o0 = sd0["layers.0.self_attn.o_proj.weight"].double()
    vb, ob = spd.conjugate_v_o(v0, o0, R2B)
    vp, op_ = spd.conjugate_v_o(v0, o0, R6P)
    gX = torch.Generator().manual_seed(11)
    x = torch.randn(16, D, generator=gX, dtype=torch.float64)
    Aw = torch.softmax(torch.randn(NH, 16, 16, generator=gX,
                                   dtype=torch.float64), dim=-1)

    def attn_out(vw, ow, xx):
        vs = (xx @ vw.t()).reshape(16, NH, HD).permute(1, 0, 2)
        ctx = torch.einsum("hts,hsd->htd", Aw, vs)
        ctx = ctx.permute(1, 0, 2).reshape(16, NH * HD)
        return ctx @ ow.t()

    y_base = attn_out(vb, ob, x)
    y_pert = attn_out(vp, op_, x)
    y_raw = attn_out(v0, o0, x)
    d_bp = float((y_base - y_pert).abs().max())
    d_br = float((y_base - y_raw).abs().max())
    results["test_r6_3_fp_gauge"] = dict(
        base_vs_perturbed=d_bp, base_vs_unrotated=d_br,
        scale=float(y_base.abs().max()))
    # perturbed R6 carries fp32 checkpoint rounding (~1e-7 relative);
    # the un-perturbed fold is exact fp64
    if d_bp > 1e-5 or d_br > 1e-10:
        fails.append(f"TEST-R6-3 FP gauge broken: {d_bp} / {d_br}")

    # ---- baseline adapter (R5, default seeded R2) ----
    st = dict(stash)
    st["R1"] = R5
    ad0 = ConcatSelectiveDraftAdapter(
        model, st, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
        **D4GS)
    ad0.install()
    w0 = {k: v.detach().float().cpu().clone()
          for k, v in grab(ea, ad0).items()}
    nq0 = n_fake(ea, ad0)
    rr0 = ad0.split.rec_embed_rescale
    ad0.uninstall()
    torch.cuda.empty_cache()

    # ---- TEST-R6-2: identity-init parity ----
    ad1 = ConcatSelectiveDraftAdapter(
        model, st, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
        r2_override=R2B, **D4GS)
    ad1.install()
    w1 = grab(ea, ad1)
    res2 = {}
    for nm in w0:
        c = w1[nm].detach().float().cpu()
        ok = whash(c) == whash(w0[nm])
        mx = float((c - w0[nm]).abs().max())
        res2[nm] = dict(equal=ok, max_abs_diff=mx)
        if not ok:
            fails.append(f"TEST-R6-2 {nm} identity-init mismatch {mx}")
    results["test_r6_2_identity_init"] = res2
    ad1.uninstall()
    torch.cuda.empty_cache()

    # ---- perturbed adapter: R2-B sensitivity + R6-4 + R6-5 ----
    ad2 = ConcatSelectiveDraftAdapter(
        model, st, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False, first_fold_R=R_T,
        r2_override=R6P, **D4GS)
    ad2.install()
    w2 = grab(ea, ad2)
    sens = {}
    for nm in ("v", "o"):
        changed = whash(w2[nm]) != whash(w0[nm])
        sens[nm + "_wfake_changed"] = changed
        if not changed:
            fails.append(f"GateR2-B {nm} w_fake UNCHANGED under "
                         f"perturbed R6 (fold on wrong side of quant?)")
    for nm in ("q", "k", "gate", "up", "down", "W_first", "W_rec",
               "head"):
        same = whash(w2[nm]) == whash(w0[nm])
        sens[nm + "_untouched"] = same
        if not same:
            fails.append(f"GateR2-B {nm} unexpectedly CHANGED by R6")
    results["gate_r2b_weight_sensitivity"] = sens

    # TEST-R6-4: independent re-quantization of the folded weight
    conv_p, _m = fq.build_spinquant_w4a4_draft_state(
        {k: v.clone() for k, v in sd0.items()}, R5, stash["gamma_f"]
        .detach().cpu().double(), r2_override=R6P)
    req = {}
    for nm, key in (("v", "layers.0.self_attn.v_proj.weight"),
                    ("o", "layers.0.self_attn.o_proj.weight")):
        ref = fq._weight_fake_quant(conv_p[key].half().to(dev), 4)
        mx = float((w2[nm].float() - ref.float()).abs().max())
        req[nm] = dict(equal=whash(w2[nm]) == whash(ref),
                       max_abs_diff=mx)
        if not req[nm]["equal"] and mx > 2e-3:
            fails.append(f"TEST-R6-4 {nm} fold-before-quant mismatch "
                         f"{mx}")
    results["test_r6_4_fold_before_quant"] = req

    # TEST-R6-5: no runtime R6 op
    op5 = dict(n_fake_w4a4=n_fake(ea, ad2), n_fake_w4a4_baseline=nq0,
               rec_embed_rescale=(None if ad2.split.rec_embed_rescale
                                  is None else float(
                                      ad2.split.rec_embed_rescale)),
               post_projection_R1=ad2.split.post_projection_R1
               is not None)
    if op5["n_fake_w4a4"] != nq0:
        fails.append("TEST-R6-5 quantized-module count changed")
    if op5["rec_embed_rescale"] != (None if rr0 is None else float(rr0)):
        fails.append("TEST-R6-5 rec_embed_rescale changed")
    results["test_r6_5_no_runtime_op"] = op5

    # Gate R2-C: trainer/runtime parity + chain sensitivity
    eq = ExactQATRotatedDraft(
        sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
        SharedRotation(R5.float()).to(dev), alpha_init=ALPHA_G,
        w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
        first_fold_R=R_T, rot2=SharedRotation(R6P32).to(dev))
    qw, _tw = eq.quantized_weights()
    resC = {}
    for nm in w2:
        h_rt, h_eq = whash(w2[nm]), whash(qw[nm])
        ok = h_rt == h_eq
        mx = float((w2[nm].float() - qw[nm].float()).abs().max())
        resC[nm] = dict(equal=ok, max_abs_diff=mx)
        if not ok and mx > 2e-3:
            fails.append(f"GateR2-C {nm} core/runtime mismatch {mx}")
    g = torch.Generator().manual_seed(7)
    T = 8
    tok_ids = torch.randint(10, 3000, (1, T + 1), generator=g).to(dev)
    a_seq = (torch.randn(1, T, eq.D, generator=g) * 1.0).to(dev)
    rt = run_runtime_chain(ea, ad2, tok_ids, a_seq, 4, ALPHA_G, dev,
                           kv_bits=16)
    torch.cuda.empty_cache()
    tr = eq.forward_chain(tok_ids, a_seq, K=4, exact=True)
    cres = []
    for k, ((ry, rh, rlg, rtk, rkv), (tlg, th, ttk)) in \
            enumerate(zip(rt, tr)):
        d_lg = float((rlg - tlg).abs().max())
        tok_eq = bool((rtk == ttk).all())
        cres.append(dict(depth=k, max_dlogits=d_lg,
                         greedy_tok_equal=tok_eq))
        if d_lg > 5e-2 or not tok_eq:
            fails.append(f"GateR2-C depth{k} dlogits={d_lg} tok={tok_eq}")
    resC["chain"] = cres
    results["gate_r2c_core_runtime_parity"] = resC
    ad2.uninstall()
    del eq
    torch.cuda.empty_cache()

    # Gate R2-B: gradient dL/dB > 0 on 3 batches (trainable R6 core)
    torch.set_grad_enabled(True)
    rot2 = ResidualRotation(R2B.float(), trust="none").to(dev)
    eqg = ExactQATRotatedDraft(
        sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
        SharedRotation(R5.float()).to(dev), alpha_init=ALPHA_G,
        w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
        first_fold_R=R_T, rot2=rot2)
    gnorms = []
    for b in range(3):
        gb = torch.Generator().manual_seed(100 + b)
        tok_ids = torch.randint(10, 3000, (1, T + 1),
                                generator=gb).to(dev)
        a_seq = (torch.randn(1, T, eqg.D, generator=gb)).to(dev)
        out = eqg.forward_chain(tok_ids, a_seq, K=4, exact=False)
        # any logits-dependent loss proves the dL/dB path through the
        # STE-quantized V/O folds; the LK-loss smoke run proves the
        # training objective itself moves (separate gate)
        loss = sum(lg.float().pow(2).mean() for lg, _h, _t in out)
        rot2.W.grad = None
        loss.backward()
        gn = float(rot2.W.grad.norm())
        gnorms.append(gn)
        if gn <= 0:
            fails.append(f"GateR2-B batch{b} dL/dB == 0")
    results["gate_r2b_grad_norms"] = gnorms
    torch.set_grad_enabled(False)

    out = os.path.join(args.run_dir, "gradchecks", "gateR6_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(alpha_gs=ALPHA_G, r5_ckpt=args.r5_ckpt,
                   perturbation="Cayley A~N(0,0.02) seed 7, generator "
                                "fro=%.4f" % float(Apert.norm()),
                   results=results, fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateR6] {'PASS' if not fails else 'FAIL'} -> {out}")
    for f in fails:
        print(f"[gateR6] FAIL: {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
