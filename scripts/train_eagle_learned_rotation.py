#!/usr/bin/env python
"""Learned projection rotation with EAGLE/acceptance/RC objectives
(study §6-§9). ALL model weights frozen — only rotation parameters
(and, in labeled arms, log-beta) train. Deployment quantizers via the
validated ExactQuantizedRotationForward STE core (arm C7 pipeline:
W4A4 target teacher, gamma_R1 basis).

Modes:
  --precompute-teachers : cache t_q (W4A4-target greedy) and t_0
        (FP16-target greedy) token ids per training row.
  --objective {nmse,eagle,acc,rc,hybrid} x --param
        {cayley,givens,householder,signperm} x --granularity block
        sizes; --pathwise for separate Q_first/Q_rec.

Objectives (all execute the deployment fake quantizer via STE):
  nmse   L = NMSE(Y_q, Y_fp) on the projection (control = NMSE-LR)
  eagle  official single-step: SmoothL1(h_pred, h_teacher)
         + 0.1 * SoftCE(logits)                      (EAGLE-LR)
  acc    -sum_k prod_{j<=k} p_D,j[t_q,j]  (log-space) (ACC-LR)
  rc     same with reference-consistency mask c_k=[t_q=t_0] (RC-LR)
  hybrid lambda-weighted combination                  (HYBRID-LR)
"""
import argparse, json, math, os, sys, time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle_spinquant.exact_quantized_rotation_forward import (
    ExactQuantizedRotationForward)
from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 interleave_perm)

D = 4096
KDEPTH = 5


# ------------------------- learned rotation modules -----------------
class LearnedRotation(torch.nn.Module):
    """Orthogonal-by-construction learned rotation on the 2D concat.
    param:
      cayley      blockwise Cayley over interleaved layout
      givens      one angle per interleaved (e_i, h_i) pair (block 2)
      householder K reflections over the full 8192 vector
    An optional FIXED structured cross rotation (spec dict) composes
    first: Q = Q_fixed Q_learned.
    """

    def __init__(self, param="cayley", block=32, K=8, fixed_spec=None,
                 n=2 * D, seed=0, device="cuda:0"):
        super().__init__()
        self.param, self.block, self.n = param, block, n
        torch.manual_seed(seed)
        self.fixed = (StructuredRotation(fixed_spec, n=n,
                                         device=device)
                      if fixed_spec else None)
        self.perm = interleave_perm(n).to(device)
        self.inv_perm = torch.argsort(self.perm)
        if param == "cayley":
            nb = n // block
            self.a = torch.nn.Parameter(
                torch.zeros(nb, block, block))
        elif param == "givens":
            self.theta = torch.nn.Parameter(torch.zeros(n // 2))
        elif param == "householder":
            self.v = torch.nn.Parameter(
                torch.randn(K, n) * 0.02)
        else:
            raise ValueError(param)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _learned(self, y):
        if self.param == "cayley":
            A = self.a - self.a.transpose(1, 2)
            eye = torch.eye(self.block, device=y.device,
                            dtype=torch.float32).expand_as(A)
            Q = torch.linalg.solve(eye + A, eye - A).to(y.dtype)
            z = y.reshape(*y.shape[:-1], self.n // self.block,
                          self.block)
            z = torch.einsum("...bi,bij->...bj", z, Q)
            return z.reshape(*y.shape)
        if self.param == "givens":
            c = torch.cos(self.theta).to(y.dtype)
            s = torch.sin(self.theta).to(y.dtype)
            z = y.reshape(*y.shape[:-1], self.n // 2, 2)
            e, h = z[..., 0], z[..., 1]
            return torch.stack([e * c + h * s, -e * s + h * c],
                               dim=-1).reshape(*y.shape)
        # householder
        z = y.float()
        for i in range(self.v.shape[0]):
            v = self.v[i]
            v = v / v.norm().clamp_min(1e-8)
            z = z - 2.0 * torch.outer(z.reshape(-1, self.n) @ v,
                                      v).reshape(z.shape)
        return z.to(y.dtype)

    def apply(self, x):
        y = x
        if self.fixed is not None:
            y = self.fixed.apply(y)
        y = y[..., self.perm.to(y.device)]
        y = self._learned(y)
        return y[..., self.inv_perm.to(y.device)]

    def orth_error(self):
        with torch.no_grad():
            eye = torch.eye(min(self.n, 512), self.n,
                            device=self.perm.device)
            q = self.apply(eye.float())
            g = q @ q.t()
            return float((g - torch.eye(q.shape[0],
                                        device=q.device)).norm()
                         / math.sqrt(q.shape[0]))


class RotCore(ExactQuantizedRotationForward):
    """Frozen core + learned projection-input rotation on both paths.
    Rotation applied to z AND to the W_first/W_rec rows before the
    weight quantizer — the exact deployed R-EP3-P contract."""

    def attach(self, rot_f, rot_r):
        self.rot_f, self.rot_r = rot_f, rot_r

    def quantized_weights(self, exact=True):
        tw = self.transformed_weights(exact)
        tw = dict(tw)
        tw["W_first"] = self.rot_f.apply(
            tw["W_first"].float()).to(tw["W_first"].dtype)
        tw["W_rec"] = self.rot_r.apply(
            tw["W_rec"].float()).to(tw["W_rec"].dtype)
        out = {k: self._qw(tw[k]) for k in
               ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up",
                "down")}
        out["head"] = tw["head"]
        return out, tw

    def forward_chain(self, tok_ids, a_seq, K, pos_offset=0,
                      recur_tokens=None, rollout=None, exact=False):
        qw, tw = self.quantized_weights(exact)
        R = tw["R"]
        a = self.alpha_exact
        B, T = a_seq.shape[0], a_seq.shape[1]
        E = self.E[tok_ids] * a
        z = torch.cat([E[:, 1:T + 1].half(), a_seq.half()], dim=-1)
        z = self.rot_f.apply(z.float()).half()
        y = self._proj(z, qw["W_first"], R)
        cache = {}
        pos = torch.arange(pos_offset, pos_offset + T,
                           device=y.device)
        h_all = self._attn_mlp(y, qw, pos, cache)
        h_last = h_all[:, -1:]
        outs = []
        for k in range(K):
            logits = F.linear(h_last.squeeze(1).half(),
                              qw["head"]).float()
            if recur_tokens is not None \
                    and k < recur_tokens.shape[1]:
                chosen = recur_tokens[:, k]
            else:
                chosen = logits.argmax(-1)
            outs.append((logits, h_last, chosen))
            if k == K - 1:
                break
            e_k = (self.E[chosen] * a).half().unsqueeze(1)
            z = torch.cat([e_k, h_last.half()], dim=-1)
            z = self.rot_r.apply(z.float()).half()
            y = self._proj(z, qw["W_rec"], R)
            pos_k = torch.arange(pos_offset + T + k,
                                 pos_offset + T + k + 1,
                                 device=y.device)
            h_all = self._attn_mlp(y, qw, pos_k, cache)
            h_last = h_all[:, -1:]
        return outs


# ------------------------- data + teachers --------------------------
def load_corpus(n_rows, tok):
    """Rows from the validated QAT tokenized cache (ShareGPT,
    eval/calib-pool-excluded). Uses a ROTATION-TRAIN slice DISJOINT
    from the QAT train slice (offset 20000) and from val."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qt", os.path.join(ROOT, "scripts",
                           "train_eagle_draft_int4_qat.py"))
    qt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qt)
    blob = qt.build_tokenized_split(tok)
    rows = blob["rows"][12400:12400 + n_rows]   # after QAT train(12000)+val(400): disjoint
    return [dict(input_ids=r["input_ids"][:640],
                 loss_mask=r["loss_mask"][:640]) for r in rows]


@torch.no_grad()
def precompute(args, rd):
    from eagle_spinquant import experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    from eagle_spinquant import eagle_bridge
    out = {}
    rows = None
    for name, (rot, quant) in (("t0", ("none", "none")),
                               ("tq", ("full", "w4a4"))):
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], rot, "learned_chat_w4a4kv16",
            quant, 0, device="cuda:0", rotations_root=rr)
        if rows is None:
            rows = load_corpus(args.n_rows,
                               eagle_bridge.get_tokenizer(model))
        toks = []
        for r in rows:
            ids = r["input_ids"][None].to("cuda:0")
            lg = model.base_model(input_ids=ids).logits
            toks.append(lg.argmax(-1)[0].cpu())
        out[name] = toks
        del model
        torch.cuda.empty_cache()
    agree = torch.cat([ (a == b).float()
                        for a, b in zip(out["t0"], out["tq"]) ])
    torch.save(dict(t0=out["t0"], tq=out["tq"],
                    rows=[dict(input_ids=r["input_ids"],
                               loss_mask=r["loss_mask"])
                          for r in rows],
                    t0_tq_agreement=float(agree.mean())),
               os.path.join(rd, "tensors", "teacher_cache.pt"))
    print(f"[teach] cached {len(rows)} rows, t0-tq agreement "
          f"{float(agree.mean()):.4f}")


# ------------------------- objectives -------------------------------
def surrogate_loss(core, ids, feat, tq_tok, t0_tok, mask, rc):
    """log-space soft prefix survival over KDEPTH teacher-forced
    steps starting at sampled anchor positions."""
    B, T = ids.shape[0], feat.shape[1]
    anchors = torch.randint(8, max(T - KDEPTH - 1, 9), (1,)).item()
    a0 = anchors
    prefix_feat = feat[:, :a0]
    rec = tq_tok[:, a0:a0 + KDEPTH]
    outs = core.forward_chain(ids[:, :a0 + 1], prefix_feat, KDEPTH,
                              recur_tokens=rec)
    logp_sum = 0.0
    total = 0.0
    masked = 0
    for k, (logits, _h, _c) in enumerate(outs):
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = tq_tok[:, a0 + k]
        a_k = lp.gather(-1, tgt[:, None]).squeeze(-1)
        if rc:
            c_k = (tq_tok[:, a0 + k] == t0_tok[:, a0 + k]).float()
            masked += int((1 - c_k).sum())
            a_k = a_k + torch.log(c_k.clamp_min(1e-9))
        logp_sum = logp_sum + a_k
        total = total + torch.exp(logp_sum.clamp(max=0))
    return -total.mean(), masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train",
                    choices=["train", "precompute-teachers"])
    ap.add_argument("--objective", default="eagle",
                    choices=["nmse", "eagle", "acc", "rc", "hybrid"])
    ap.add_argument("--param", default="cayley",
                    choices=["cayley", "givens", "householder"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--hh-k", type=int, default=8)
    ap.add_argument("--pathwise", action="store_true")
    ap.add_argument("--fixed-init", action="store_true",
                    help="compose after the best fixed cross rotation")
    ap.add_argument("--lambdas", default="1.0,0.1,0.1,0.0",
                    help="hybrid: lE,lA,lR,lN")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rows", type=int, default=256)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    rd = args.run_dir
    dev = "cuda:0"
    if args.mode == "precompute-teachers":
        return precompute(args, rd)

    tag = args.tag or (f"{args.objective}_{args.param}"
                       f"{args.block}_s{args.seed}"
                       + ("_pw" if args.pathwise else "_sh")
                       + ("_fx" if args.fixed_init else ""))
    cache = torch.load(os.path.join(rd, "tensors",
                                    "teacher_cache.pt"),
                       map_location="cpu", weights_only=False)
    from eagle_spinquant import experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", "learned_chat_w4a4kv16",
        "w4a4", 0, device=dev, rotations_root=rr)
    ea_sd = {k: v.detach().cpu()
             for k, v in model.ea_layer.state_dict().items()}
    R1 = stash["R1"].float()
    gamma = stash["gamma_f"].float()
    W_lm = stash["lm_head_weight"].float()

    class _FixedRot(torch.nn.Module):
        def __init__(self, R):
            super().__init__()
            self.register_buffer("R", R)

        def forward(self):
            return self.R
    core = RotCore(ea_sd, R1, gamma, W_lm, _FixedRot(R1.to(dev)),
                   alpha_init=float(D ** 0.40), train_alpha=False,
                   w_bits=4, a_bits=4, draft_kv_bits=16,
                   train_draft_core=False, r2_seed=0, device=dev,
                   first_fold_R=R1).to(dev)
    for p in core.parameters():
        p.requires_grad_(False)
    fixed_spec = (dict(family="cross", block=8192, seed=12,
                       interleave_chunk=1) if args.fixed_init
                  else None)
    rot_f = LearnedRotation(args.param, args.block, args.hh_k,
                            fixed_spec, seed=args.seed,
                            device=dev).to(dev)
    rot_r = (LearnedRotation(args.param, args.block, args.hh_k,
                             fixed_spec, seed=args.seed + 100,
                             device=dev).to(dev)
             if args.pathwise else rot_f)
    core.attach(rot_f, rot_r)
    params = list(rot_f.parameters()) + (
        list(rot_r.parameters()) if args.pathwise else [])
    assert all(not p.requires_grad for p in core.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    lE, lA, lR, lN = [float(x) for x in args.lambdas.split(",")]
    rows = cache["rows"][: args.n_rows]
    tq, t0 = cache["tq"], cache["t0"]
    crit = torch.nn.SmoothL1Loss()
    R1d = R1.to(dev)
    gd = gamma.to(dev)
    hist = []
    t_start = time.time()
    os.makedirs(os.path.join(rd, "rotations"), exist_ok=True)
    for step in range(args.steps):
        i = step % len(rows)
        ids = rows[i]["input_ids"][None].to(dev)
        with torch.no_grad():
            h = model.base_model.model(
                input_ids=ids).last_hidden_state.float()
        feat = h[:, :-1]
        h_tgt = ((h @ R1d.t()) * gd) @ R1d           # C7 basis
        loss = torch.zeros((), device=dev)
        parts = {}
        if args.objective in ("eagle", "hybrid") and lE > 0:
            h_pred = core.forward_train(ids, feat)
            pred_lg = core.head_logits(h_pred)
            vl = crit(h_pred.float(), h_tgt[:, 1:])
            with torch.no_grad():
                t_lg = F.linear(
                    h_tgt[:, 1:].half().squeeze(0),
                    core.transformed_weights(False)["head"]).float()
            pl = -(t_lg.softmax(-1)
                   * pred_lg.float().log_softmax(-1)).sum(-1).mean()
            le = vl + 0.1 * pl
            loss = loss + (lE if args.objective == "hybrid"
                           else 1.0) * le
            parts["eagle"] = float(le)
        if args.objective in ("acc", "rc", "hybrid"):
            want_rc = args.objective == "rc" or (
                args.objective == "hybrid" and lR > 0)
            want_acc = args.objective == "acc" or (
                args.objective == "hybrid" and lA > 0)
            if want_acc:
                la, _ = surrogate_loss(core, ids, feat,
                                       tq[i][None].to(dev),
                                       t0[i][None].to(dev),
                                       None, rc=False)
                loss = loss + (lA if args.objective == "hybrid"
                               else 1.0) * la
                parts["acc"] = float(la)
            if want_rc:
                lr_, msk = surrogate_loss(core, ids, feat,
                                          tq[i][None].to(dev),
                                          t0[i][None].to(dev),
                                          None, rc=True)
                loss = loss + (lR if args.objective == "hybrid"
                               else 1.0) * lr_
                parts["rc"] = float(lr_)
                parts["rc_masked"] = msk
        if args.objective == "nmse" or (
                args.objective == "hybrid" and lN > 0):
            qw, tw = core.quantized_weights(False)
            E = core.E[ids[0]] * core.alpha_exact
            z = torch.cat([E[1:].half(),
                           feat[0].half()], dim=-1)
            zr = core.rot_f.apply(z.float()).half()
            yq = core._proj(zr[None], qw["W_first"], tw["R"])
            with torch.no_grad():
                Wf = core.transformed_weights(True)["W_first"]
                y0 = core._proj_fp_reference(z, Wf, tw["R"]) \
                    if hasattr(core, "_proj_fp_reference") else None
            if y0 is None:
                with torch.no_grad():
                    y0 = (F.linear(z, Wf.half(),
                                   core.b_fc.half()).float()
                          @ tw["R"]).half()[None]
            ln = ((yq.float() - y0.float()) ** 2).sum() \
                / (y0.float() ** 2).sum()
            loss = loss + (lN if args.objective == "hybrid"
                           else 1.0) * ln
            parts["nmse"] = float(ln)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"[{tag}] step {step} loss {float(loss):.4f} "
                  f"{parts}", flush=True)
            hist.append(dict(step=step, loss=float(loss), **parts))
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            torch.save(dict(
                rot_f=rot_f.state_dict(),
                rot_r=(rot_r.state_dict() if args.pathwise
                       else None),
                cfg=vars(args), step=step + 1,
                orth=rot_f.orth_error()),
                os.path.join(rd, "rotations",
                             f"{tag}_step{step + 1}.pt"))
    wall = time.time() - t_start
    torch.save(dict(rot_f=rot_f.state_dict(),
                    rot_r=(rot_r.state_dict() if args.pathwise
                           else None),
                    cfg=vars(args), step=args.steps,
                    orth=rot_f.orth_error(), wall=wall,
                    n_params=rot_f.n_params()
                    * (2 if args.pathwise else 1), hist=hist),
               os.path.join(rd, "rotations", f"{tag}_last.pt"))
    json.dump(dict(tag=tag, wall=wall, orth=rot_f.orth_error(),
                   n_params=rot_f.n_params(), hist=hist),
              open(os.path.join(rd, "rotations",
                                f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {wall:.0f}s orth {rot_f.orth_error():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
