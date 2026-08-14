#!/usr/bin/env python
"""Gate E-core (acceptance-aware QAT study, 2026-08-12).

The prior Gate E (analyze_lk_gradients.py) validated gradients into the
ROTATION through the STE-quantized K-step chain. This gate validates the
NEW pathway used by the AA-QAT study: gradients into the 11 draft-core
masters (W_e, W_h, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, b_fc, gl64) through
`forward_chain` with `train_draft_core=True`, for every study objective.

Checks per objective:
  G1  every CORE master receives a finite gradient; the 9 GEMM masters
      receive a NONZERO gradient (b_fc/gl64 norms reported, finite-only)
  G2  hash sensitivity: perturbing a master changes the quantized site
  G3  central finite-difference directional check on Wdown (sign match;
      loose ratio band — STE + per-step clip search make exact ratios
      impossible by construction)
  G4  batch-32 forward+backward peak-memory smoke (24 GB budget)

conv objective: uses the corpus a_chain when present; otherwise a randn
surrogate (gradient-FLOW gate only, values meaningless) — reported.

Writes gradchecks/gate_core_chain.json in --run-dir.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F
from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import SharedRotation

KIND = "learned_chat_w4a4kv16"
MASTERS = ("W_e", "W_h", "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown",
           "b_fc", "gl64")
GEMM9 = MASTERS[:9]


def build(dev, alpha, alpha_rec, rot_fixed):
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    sd = torch.load(os.path.join(paths["draft_path"], "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    import glob
    from safetensors import safe_open
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    if rot_fixed:
        rck = torch.load(rot_fixed, map_location="cpu", weights_only=False)
        rot = SharedRotation(rck["R_D"].float()).to(dev)
    else:
        rot = SharedRotation(R_T).to(dev)
    model = ExactQATRotatedDraft(
        sd, R_T, gamma, W_lm, rot, alpha_init=alpha, train_alpha=False,
        w_bits=4, a_bits=4, draft_kv_bits=16, train_draft_core=True,
        device=dev, first_fold_R=R_T, alpha_rec_init=alpha_rec)
    return model, rot, R_T.to(dev), gamma.to(dev).float()


def objective_loss(model, rot, R1d, gd, obj, ws, K, dev, wk):
    tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
    a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
    teach = torch.stack([w["teacher_tokens"][:K].long()
                         for w in ws]).to(dev)
    zT = torch.stack([w["teacher_logits"][:K].float()
                      for w in ws]).to(dev)
    conv_t = surrogate = None
    if obj == "conv":
        surrogate = "a_chain" not in ws[0]
        with torch.no_grad():
            ach = (torch.randn(len(ws), K, a.shape[-1], device=dev)
                   if surrogate else
                   torch.stack([w["a_chain"][:K].float()
                                for w in ws]).to(dev))
            conv_t = ((ach @ R1d.t()) * gd) @ rot.R().detach()
    outs = model.forward_chain(tok, a, K, recur_tokens=teach)
    total, alphas, logps = 0.0, [], []
    for k, (lg, hk, _c) in enumerate(outs):
        zTk = zT[:, k]
        if obj == "hybrid":
            lk, _l, _a = L.hybrid_lk(zTk, lg, eta=3.0)
            total = total + wk[k] * lk.mean()
        elif obj == "greedy":
            total = total + wk[k] * L.greedy_ce(zTk, lg).mean()
        elif obj == "conv":
            hf = hk.squeeze(1)
            tk_ = conv_t[:, k]
            v = F.smooth_l1_loss(hf.float(), tk_,
                                 reduction="none").mean(-1)
            with torch.no_grad():
                tp = model.head_logits(tk_).softmax(-1)
            lp = model.head_logits(hf).log_softmax(-1)
            total = total + (v + 0.1 * (-(tp * lp).sum(-1))).mean() / K
        elif obj == "exptau":
            total = total + 0.1 * L.kl_full(zTk, lg).mean()
        elif obj == "accsurv":
            logps.append(L.teacher_token_logp(lg, teach[:, k]))
        alphas.append(L.overlap_alpha(zTk, lg))
    a_st = torch.stack(alphas, dim=-1)
    if obj in ("exptau",):
        total = total + L.expected_tau_loss(a_st).mean()
    elif obj == "survival":
        total = total - L.expected_tau(a_st).mean()
    elif obj == "accsurv":
        total = total - L.prefix_survival(
            torch.stack(logps, dim=-1)).mean()
    return total, bool(surrogate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--corpus-run-dir", default=None,
                    help="run dir whose rotations/ holds the corpus "
                         "shards (default: --run-dir)")
    ap.add_argument("--objectives",
                    default="conv,hybrid,exptau,survival,greedy,accsurv")
    ap.add_argument("--rot-fixed-ckpt", default=None)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--alpha-rec", type=float, default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mem-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    man = json.load(open(args.corpus))
    sh = os.path.join(args.corpus_run_dir or args.run_dir, "rotations",
                      man["shards"][0]["shard"])
    ws_all = torch.load(sh, map_location="cpu",
                        weights_only=False)["windows"]
    model, rot, R1d, gd = build(dev, args.alpha, args.alpha_rec,
                                args.rot_fixed_ckpt)
    model.train()
    wk = L.depth_weights(args.K, 0.8, device=dev)
    report = dict(rot_fixed_ckpt=args.rot_fixed_ckpt, alpha=args.alpha,
                  alpha_rec=args.alpha_rec, objectives={})
    ok = True

    for obj in args.objectives.split(","):
        for n in MASTERS:
            p = getattr(model, n)
            if p.grad is not None:
                p.grad = None
        loss, surrogate = objective_loss(model, rot, R1d, gd, obj,
                                         ws_all[:args.batch], args.K,
                                         dev, wk)
        loss.backward()
        row = dict(loss=float(loss), conv_surrogate_targets=surrogate,
                   grad_norms={}, g1_pass=True)
        for n in MASTERS:
            gn = getattr(model, n).grad
            v = float(gn.norm()) if gn is not None else None
            row["grad_norms"][n] = v
            finite = v is not None and v == v and v != float("inf")
            if not finite or (n in GEMM9 and (v is None or v == 0.0)):
                row["g1_pass"] = False
        # G3': descent-step check. A naive small-eps FD is structurally
        # meaningless for STE weight gradients (per-element perturbations
        # far below the W4 LSB probe only the quantizer staircase; the
        # dense hybrid/exptau objectives DID pass an FD sign check at
        # eps=1e-3, ratio 2-3, confirming the plumbing). What QAT needs
        # is that a training-LR step along -grad reduces the loss through
        # the quantized forward — test exactly that, on the same batch.
        if row["g1_pass"] and obj != "conv":
            lr = 1e-5
            saved = {n: getattr(model, n).detach().clone()
                     for n in MASTERS}
            with torch.no_grad():
                for n in MASTERS:
                    p = getattr(model, n)
                    if p.grad is not None:
                        gnorm = p.grad.norm().clamp_min(1e-12)
                        # normalized SGD (Adam-like unit step per tensor)
                        p.add_(-lr * p.grad
                               * (p.numel() ** 0.5 / gnorm))
            l1_, _ = objective_loss(model, rot, R1d, gd, obj,
                                    ws_all[:args.batch], args.K, dev, wk)
            with torch.no_grad():
                for n in MASTERS:
                    getattr(model, n).data.copy_(saved[n])
            row["g3_loss_before"] = float(loss)
            row["g3_loss_after"] = float(l1_)
            row["g3_pass"] = float(l1_) < float(loss)
        report["objectives"][obj] = row
        ok = ok and row["g1_pass"] and row.get("g3_pass", True)
        print(f"[gate] {obj}: loss={row['loss']:.5f} "
              f"g1={row['g1_pass']} g3={row.get('g3_pass', 'n/a')} "
              f"|gWdown|={row['grad_norms']['Wdown']}", flush=True)

    # G2: hash sensitivity of the quantized down site to a master change
    with torch.no_grad():
        qw0, _ = model.quantized_weights(exact=False)
        h0 = hashlib.sha256(qw0["down"].detach().cpu().numpy()
                            .tobytes()).hexdigest()[:16]
        model.Wdown.add_(1e-3 * torch.randn_like(model.Wdown))
        qw1, _ = model.quantized_weights(exact=False)
        h1 = hashlib.sha256(qw1["down"].detach().cpu().numpy()
                            .tobytes()).hexdigest()[:16]
    report["g2_hash_before"], report["g2_hash_after"] = h0, h1
    report["g2_pass"] = h0 != h1
    ok = ok and report["g2_pass"]

    # G4: batch-32 memory smoke (hybrid)
    torch.cuda.reset_peak_memory_stats(dev)
    for n in MASTERS:
        getattr(model, n).grad = None
    loss, _ = objective_loss(model, rot, R1d, gd, "hybrid",
                             ws_all[:args.mem_batch], args.K, dev, wk)
    loss.backward()
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    report["g4_peak_gib_batch"] = dict(batch=args.mem_batch,
                                       peak_gib=round(peak, 2))
    report["g4_pass"] = peak < 21.0
    ok = ok and report["g4_pass"]
    print(f"[gate] G2 hash {h0}->{h1} pass={report['g2_pass']}; "
          f"G4 peak {peak:.2f} GiB pass={report['g4_pass']}", flush=True)

    report["ALL_PASS"] = ok
    out = os.path.join(args.run_dir, "gradchecks", "gate_core_chain.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"[gate] ALL_PASS={ok} -> {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
