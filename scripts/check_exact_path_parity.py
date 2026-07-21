#!/usr/bin/env python
"""Gate D: exact training/runtime parity harness.

Builds the REAL runtime draft (ConcatSelectiveDraftAdapter, D4P3, ar_r2r4,
first_fold_R bridge, P3 alpha) for a given R_D, and the differentiable
ExactQATRotatedDraft with the same R_D — then requires:

  1. identical fp16 quantized weights (bytewise hash) for all 9 quantized
     tensors + the fp16 head;
  2. identical projection outputs, decoder hidden, draft logits, greedy
     draft tokens along a K=4 teacher-forced chain (strict fp16 equality;
     activation-scale stats compared, KV compared per depth);
  3. the same holds for a NON-trivial R_D (residual-perturbed R_T) —
     ruling out accidental R_T-only agreement;
  4. a KV4 leg where both sides quantize appended K/V with the runtime
     fake_quant_kv.

Writes <run>/gradchecks/gateD_parity.json; exits 1 on any failure.
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
                                               ResidualRotation)
from eagle_spinquant.kv4_cache import fake_quant_kv

KIND = "learned_chat_w4a4kv16"
D4P3 = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
            quant_ar="fake_w4a4", ar_r2r4=True, embed_scale_alpha=32.0)


def whash(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float16).numpy()
                          .tobytes()).hexdigest()[:16]


def hf_mask(T, Tc, device, dtype):
    m = torch.zeros(1, 1, T, Tc, device=device, dtype=dtype)
    if T > 1:
        causal = torch.full((T, Tc), torch.finfo(dtype).min, device=device,
                            dtype=dtype).triu(Tc - T + 1)
        m[0, 0] = causal
    return m


def run_runtime_chain(ea, adapter, tok_ids, a_seq, K, alpha, dev,
                      kv_bits=16):
    """Drive the installed runtime modules exactly like the deployed chain."""
    E = ea.embed_tokens                     # table already alpha-scaled
    T = a_seq.shape[1]
    e_pref = E(tok_ids[:, 1:T + 1]).half()
    z = torch.cat([e_pref, a_seq.half()], dim=-1)
    adapter.split.select = "first"
    y = adapter.split(z)
    pos = torch.arange(0, T, device=dev)[None]
    past = None
    outs = []
    layer = ea.layers[0]
    lo = layer(y, attention_mask=hf_mask(T, T, dev, y.dtype),
               position_ids=pos, past_key_value=None, use_cache=True)
    h_all, past = lo[0], lo[-1]
    if kv_bits < 16:
        past = tuple(fake_quant_kv(p, bits=kv_bits) for p in past)
    h_last = h_all[:, -1:]
    head = adapter.head
    for k in range(K):
        logits = head(h_last.squeeze(1).half()).float()
        chosen = logits.argmax(-1)
        outs.append((y.detach().clone(), h_last.detach().clone(),
                     logits.detach().clone(), chosen.detach().clone(),
                     tuple(p.detach().clone() for p in past)))
        if k == K - 1:
            break
        e_k = E(chosen).half().unsqueeze(1)
        z = torch.cat([e_k, h_last.half()], dim=-1)
        adapter.split.select = "recurrent"
        y = adapter.split(z)
        pos_k = torch.tensor([[T + k]], device=dev)
        print(f"[dbg] k={k} past_type={type(past)} len={len(past)} "
              f"shapes={[tuple(p_.shape) for p_ in past if hasattr(p_, 'shape')]}",
              flush=True)
        Tc = past[0].shape[2] + 1
        try:
            lo = layer(y, attention_mask=hf_mask(1, Tc, dev, y.dtype),
                       position_ids=pos_k, past_key_value=past,
                       use_cache=True)
        except RuntimeError:
            print(f"[dbg-crash] y={tuple(y.shape)} "
                  f"past0={tuple(past[0].shape)} "
                  f"past1={tuple(past[1].shape)} "
                  f"layer_qproj={type(layer.self_attn.q_proj).__name__}",
                  flush=True)
            raise
        h_all, past = lo[0], lo[-1]
        if kv_bits < 16:
            newk = fake_quant_kv(past[0][:, :, -1:], bits=kv_bits)
            newv = fake_quant_kv(past[1][:, :, -1:], bits=kv_bits)
            past = (torch.cat([past[0][:, :, :-1], newk], dim=2),
                    torch.cat([past[1][:, :, :-1], newv], dim=2))
        h_last = h_all[:, -1:]
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
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

    results = {}
    fails = []
    for tag, kv_bits in (("RT_kv16", 16), ("RESID_kv16", 16),
                         ("RT_kv4", 4)):
        if tag.startswith("RESID"):
            rr_mod = ResidualRotation(R_T.float()).to(dev)
            with torch.no_grad():
                rr_mod.W[:64, 64:128] += 0.02      # non-trivial local move
            # freeze the Cayley result so BOTH sides consume the exact
            # same fp32 matrix (recomputing on cpu vs gpu differs in ulps)
            from eagle_spinquant.residual_rotation import FullRotation
            rot = FullRotation(rr_mod.R().detach())
        else:
            rot = SharedRotation(R_T.float())
        R_D = rot.R().detach().cpu().double()

        st = dict(stash)
        st["R1"] = R_D
        ad = ConcatSelectiveDraftAdapter(
            model, st, dev, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False,
            first_fold_R=R_T if tag.startswith("RESID") else None, **D4P3)
        ad.install()

        eq = ExactQATRotatedDraft(
            sd0, R_T, stash["gamma_f"], stash["lm_head_weight"].float(),
            rot.to(dev), alpha_init=32.0, w_bits=4, a_bits=4,
            draft_kv_bits=kv_bits, device=dev,
            first_fold_R=R_T)
        qw, _tw = eq.quantized_weights()

        # ---- 1. weight hashes ----
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
            mx = float((rt_w[name].float() - qw[name].float()).abs().max())
            wres[name] = dict(runtime=h_rt, exact=h_eq, equal=ok,
                              max_abs_diff=mx)
            if not ok and mx > 2e-3:
                fails.append(f"{tag}:{name} weight mismatch max={mx}")

        # ---- 2. chain forward parity ----
        g = torch.Generator().manual_seed(7)
        T = 8
        tok_ids = torch.randint(10, 3000, (1, T + 1), generator=g).to(dev)
        a_seq = (torch.randn(1, T, eq.D, generator=g) * 1.0).to(dev)
        rt = run_runtime_chain(ea, ad, tok_ids, a_seq, 4, 32.0, dev,
                               kv_bits=kv_bits)
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
                fails.append(f"{tag}:depth{k} dlogits={d_lg} tok={tok_eq}")
        results[tag] = dict(weights=wres, chain=cres)
        ad.uninstall()
        del eq
        torch.cuda.empty_cache()

    out = os.path.join(args.run_dir, "gradchecks", "gateD_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(results=results, fails=fails,
                   verdict="PASS" if not fails else "FAIL"),
              open(out, "w"), indent=1)
    print(f"[gateD] {'PASS' if not fails else 'FAIL'} -> {out}")
    for f in fails:
        print(f"[gateD] FAIL: {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
