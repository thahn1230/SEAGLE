#!/usr/bin/env python
"""Exact-path LK trainer (study spec sections 5-14).

Trains a draft rotation R_D (shared/full/residual-Cayley), optional P3
log-alpha, and optionally the draft-core weights (C2/C3 capacity levels)
against FULL-VOCABULARY deployed-target teachers through the runtime-exact
quantized forward (ExactQATRotatedDraft; Gate D).

Objectives: kl (exact-path full-vocab KL), tv, neglog (-log alpha),
hybrid (adaptive KL/TV, eta), hybrid_fixed (lambda=0.5), exptau
(expected accepted length + 0.1*KL trust), topk_kl (previous-study proxy,
ablation only). Optional greedy auxiliary CE.

Trajectories: teacher-forced, or on-policy with the LIVE deployed target
evaluated on the visited prefix (curriculum/fifty; spec 11.2).

Optimizer: AdamW(b1=.9,b2=.95) + cosine schedule + 100-step warmup +
grad-clip 0.5; effective batch = --batch * --accum.
"""
import argparse, glob, hashlib, json, math, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import (SharedRotation, FullRotation,
                                               ResidualRotation,
                                               rotation_geometry)
from eagle_spinquant.onpolicy_draft_rollout import (
    curriculum_p_onpolicy, greedy_rollout, stochastic_rollout,
    DivergenceTracker)
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
TARGETS = {"t8": ("full", "w8a8", 16), "t4": ("full", "w4a4", 16),
           "t4kv4": ("full", "w4a4", 4)}


class OnlineTeacher:
    """Live deployed target for on-policy LK: logits at visited prefixes."""

    def __init__(self, teacher, device, cfg, paths, rr):
        rot, quant, kv = TARGETS[teacher]
        model, _stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], rot, KIND, quant, 0, device=device,
            rotations_root=rr)
        self.bm = model.base_model
        self.bm.model.tree_mask = None
        self.past, _pd, self.cur = initialize_past_key_values(self.bm)
        if kv < 16:
            install_kv4_on_past(self.past, bits=kv)

    @torch.no_grad()
    def visited_logits(self, prefix_ids, chosen_tokens):
        """prefix_ids: (Lp,) window prefix; chosen_tokens: (K,) visited.
        Returns (K, V): target logits after prefix + chosen[:k] for k=0..K-1
        (i.e. the distribution the verifier would use at each depth)."""
        dev = prefix_ids.device
        self.cur.zero_()
        h = self.bm.model(input_ids=prefix_ids[None],
                          past_key_values=self.past, use_cache=True)[0]
        lg = self.bm.lm_head(h[:, -1:])[0, 0]
        outs = [lg]
        for k in range(chosen_tokens.shape[0] - 1):
            step = chosen_tokens[k].view(1, 1)
            h1 = self.bm.model(input_ids=step, past_key_values=self.past,
                               use_cache=True)[0]
            outs.append(self.bm.lm_head(h1)[0, 0])
        return torch.stack(outs)


def load_corpus(manifest_path, run_dir):
    man = json.load(open(manifest_path))
    windows = []
    for sh in man["shards"]:
        p = os.path.join(run_dir, "rotations", sh["shard"])
        windows += torch.load(p, map_location="cpu",
                              weights_only=False)["windows"]
    return windows, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="manifest json path")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="t4kv4", choices=list(TARGETS))
    ap.add_argument("--rot", default="residual",
                    choices=["shared", "full", "residual"])
    ap.add_argument("--rot-init", default="RT",
                    choices=["RT", "identity", "hadamard", "random"])
    ap.add_argument("--trust", default="none",
                    choices=["none", "weak", "medium", "hard"])
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--objective", default="hybrid",
                    choices=["kl", "tv", "neglog", "hybrid", "hybrid_fixed",
                             "exptau", "topk_kl", "accsurv"])
    ap.add_argument("--eta", type=float, default=3.0)
    ap.add_argument("--aux-greedy", type=float, default=0.0)
    ap.add_argument("--gamma-depth", type=float, default=0.8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--core-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--alpha-init", type=float, default=32.0)
    ap.add_argument("--alpha-rec-init", type=float, default=None,
                    help="EP3-P pathwise m_rec; enables the deploy-"
                         "adapter EP3-P fold (e-slice divided pre-cast, "
                         "recurrent e-slice rescale)")
    ap.add_argument("--train-alpha", action="store_true")
    ap.add_argument("--train-draft-core", action="store_true")
    ap.add_argument("--kv-bits", type=int, default=4)
    ap.add_argument("--onpolicy", default="tf",
                    choices=["tf", "curriculum", "fifty", "onpolicy"])
    ap.add_argument("--rollout", default="greedy",
                    choices=["greedy", "t1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    torch.manual_seed(args.seed)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    from safetensors.torch import load_file, safe_open
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(paths["draft_path"], fn)
        if os.path.exists(p):
            sd = load_file(p) if fn.endswith("safetensors") else \
                torch.load(p, map_location="cpu", weights_only=True)
            break
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()

    if args.rot == "shared":
        rot = SharedRotation(R_T)
    elif args.rot == "full":
        if args.rot_init == "RT":
            R0 = R_T.clone()
        elif args.rot_init == "identity":
            R0 = torch.eye(R_T.shape[0])
        elif args.rot_init == "hadamard":
            import scipy.linalg as sl
            D_ = R_T.shape[0]
            R0 = torch.tensor(sl.hadamard(D_) / D_ ** 0.5,
                              dtype=torch.float32)
        else:
            g = torch.Generator().manual_seed(args.seed)
            R0 = torch.linalg.qr(torch.randn(*R_T.shape, generator=g))[0]
        rot = FullRotation(R0)
    else:
        rot = ResidualRotation(R_T, trust=args.trust, radius=args.radius)
    rot = rot.to(dev)

    model = ExactQATRotatedDraft(
        sd, R_T, gamma, W_lm, rot, alpha_init=args.alpha_init,
        train_alpha=args.train_alpha, w_bits=4, a_bits=4,
        draft_kv_bits=args.kv_bits, train_draft_core=args.train_draft_core,
        device=dev, first_fold_R=R_T, alpha_rec_init=args.alpha_rec_init)

    windows, man = load_corpus(args.corpus, args.run_dir)
    n_val = max(len(windows) // 10, 16)
    W_tr, W_va = windows[:-n_val], windows[-n_val:]
    print(f"[lk] {len(W_tr)} train / {len(W_va)} val windows "
          f"obj={args.objective} rot={args.rot} trust={args.trust} "
          f"onpolicy={args.onpolicy} core={args.train_draft_core}",
          flush=True)

    groups = []
    if rot.trainable:
        groups.append(dict(params=[p for p in rot.parameters()],
                           lr=args.lr))
    if args.train_alpha:
        groups.append(dict(params=[model.log_alpha], lr=args.lr))
    if args.train_draft_core:
        core = [getattr(model, n) for n in model.CORE]
        groups.append(dict(params=core, lr=args.core_lr))
    assert groups, "nothing trainable"
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    teacher_live = None
    if args.onpolicy != "tf":
        teacher_live = OnlineTeacher(args.teacher, dev, cfg, paths, rr)
    g = torch.Generator().manual_seed(args.seed)
    gen_dev = torch.Generator(device=dev)
    gen_dev.manual_seed(args.seed)
    wk = L.depth_weights(args.K, args.gamma_depth, device=dev)
    div = DivergenceTracker(args.K)
    log = []
    t0 = time.time()

    def batch_loss(ws, onpol):
        """Real batched loss over a list of windows (uniform T, K).
        pos_offset is dropped (RoPE is relative; full-prefix attention is
        shift-invariant — verified in the previous study)."""
        tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
        a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
        teach = torch.stack([w["teacher_tokens"][:args.K].long()
                             for w in ws]).to(dev)
        if onpol:
            roll = (greedy_rollout if args.rollout == "greedy" else
                    lambda lg: stochastic_rollout(lg, 1.0, gen_dev))
            outs = model.forward_chain(tok, a, args.K, rollout=roll)
            chosen = torch.stack([o[2] for o in outs], dim=1)  # (B, K)
            for k in range(args.K):
                div.update(k, chosen[:, k], teach[:, k])
            zT = torch.stack([
                teacher_live.visited_logits(
                    tok[b, :a.shape[1] + 1], chosen[b])
                for b in range(tok.shape[0])], dim=0)      # (B, K, V)
        else:
            outs = model.forward_chain(tok, a, args.K,
                                       recur_tokens=teach)
            zT = torch.stack([w["teacher_logits"][:args.K].float()
                              for w in ws]).to(dev)
        total = 0.0
        alphas = []
        logps = []                          # accsurv per-depth log q(t_k)
        for k, (lg, _h, _c) in enumerate(outs):
            zTk, zDk = zT[:, k], lg
            if args.objective == "accsurv":
                # LRGF-validated ACC surrogate: soft prefix survival of
                # the teacher-forced greedy tokens (loss applied after
                # the depth loop; no per-depth weighting — the survival
                # product couples depths already)
                logps.append(L.teacher_token_logp(zDk, teach[:, k]))
                alphas.append(L.overlap_alpha(zTk, zDk))
                continue
            if args.objective == "kl":
                lk = L.kl_full(zTk, zDk)
            elif args.objective == "tv":
                lk = L.tv(zTk, zDk)
            elif args.objective == "neglog":
                lk = L.neg_log_alpha(zTk, zDk)
            elif args.objective == "hybrid":
                lk, _lam, _a = L.hybrid_lk(zTk, zDk, eta=args.eta)
            elif args.objective == "hybrid_fixed":
                lk, _lam, _a = L.hybrid_lk(zTk, zDk, fixed_lambda=0.5)
            elif args.objective == "topk_kl":
                lk = L.kl_topk(zTk, zDk, k=64)
            else:                                   # exptau
                lk = 0.1 * L.kl_full(zTk, zDk)      # trust regularizer
            alphas.append(L.overlap_alpha(zTk, zDk))
            if args.objective != "exptau":
                total = total + wk[k] * lk.mean()
            else:
                total = total + lk.mean()
            if args.aux_greedy > 0:
                total = total + args.aux_greedy * wk[k] * \
                    L.greedy_ce(zTk, zDk).mean()
        a_stack = torch.stack(alphas, dim=-1)       # (B, K)
        if args.objective == "exptau":
            total = total + L.expected_tau_loss(a_stack).mean()
        elif args.objective == "accsurv":
            total = total - L.prefix_survival(
                torch.stack(logps, dim=-1)).mean()
        return total, a_stack.detach()

    best_val = None
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        p_on = curriculum_p_onpolicy(step, args.steps, args.onpolicy)
        acc_alpha = []
        for _ in range(args.accum):
            idxs = torch.randint(0, len(W_tr), (args.batch,),
                                 generator=g).tolist()
            onpol = bool(torch.rand((), generator=g) < p_on)
            lw, aw = batch_loss([W_tr[i] for i in idxs], onpol)
            acc_alpha.append(aw)
            (lw / args.accum).backward()
        pen = rot.penalty()
        if float(pen) != 0.0:
            pen.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for gp in groups for p in gp["params"]], args.clip)
        opt.step()
        sched.step()
        rot.post_step()
        if hasattr(rot, "orth_error"):
            oe = rot.orth_error()
            assert oe < 1e-3, f"Gate G: orth {oe}"
        if step % 20 == 0 or step == args.steps - 1:
            am = torch.cat(acc_alpha).mean(0)
            log.append(dict(step=step,
                            alpha_by_depth=[round(float(x), 4)
                                            for x in am],
                            lr=float(sched.get_last_lr()[0]),
                            p_onpolicy=p_on,
                            alpha_p3=float(model.log_alpha.exp())))
            print(f"[lk] step {step}: mean_alpha="
                  f"{[round(float(x),3) for x in am]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                va = []
                for i0 in range(0, min(len(W_va), 64), 16):
                    _lw, aw = batch_loss(W_va[i0:i0 + 16], False)
                    va.append(aw)
                vam = torch.cat(va).mean(0)
            et = float(L.expected_tau(torch.cat(va)).mean())
            print(f"[lk] VAL step {step}: alpha_by_depth="
                  f"{[round(float(x),4) for x in vam]} "
                  f"expected_tau={et:.4f}", flush=True)
            log.append(dict(step=step, val_alpha=[float(x) for x in vam],
                            val_expected_tau=et))
            if best_val is None or et > best_val[0]:
                best_val = (et, step)

    geo = rotation_geometry(rot.R().detach().cpu(), R_T) \
        if rot.trainable else rotation_geometry(R_T, R_T)
    save = dict(R_D=rot.R().detach().cpu(),
                alpha=(model.alpha_exact
                       if not args.train_alpha
                       else float(model.log_alpha.exp())),
                alpha_rec=model.alpha_rec_exact,
                meta=dict(vars(args), corpus_n=len(windows),
                          geometry=geo, best_val=best_val,
                          divergence_rates=div.rates(),
                          counters=model.counters),
                log=log)
    if args.train_draft_core:
        save["core_weights"] = model.export_original_state()
    torch.save(save, args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    open(args.out + ".sha256", "w").write(sha + "\n")
    print(f"[lk] saved {args.out} sha={sha[:16]} "
          f"best_val_exptau={best_val}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
