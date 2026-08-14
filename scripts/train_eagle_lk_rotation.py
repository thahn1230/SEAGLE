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
import torch.nn.functional as F
from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import (SharedRotation, FullRotation,
                                               ResidualRotation,
                                               ResidualR2Rotation,
                                               rotation_geometry)
from eagle_spinquant.onpolicy_draft_rollout import (
    curriculum_p_onpolicy, greedy_rollout, stochastic_rollout,
    DivergenceTracker)
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
TARGETS = {"t16": ("none", "none", 16),
           "t8": ("full", "w8a8", 16), "t4": ("full", "w4a4", 16),
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
    # GS/R2 study (2026-08): draft attention V/O R2 basis
    ap.add_argument("--r2-mode", default="frozen",
                    choices=["frozen", "residual"],
                    help="frozen: deployed baseline R2_B (seed-0 Haar-QR, "
                         "legacy behavior); residual: learnable "
                         "R2_D = R2_B C(B), B skew, one [128,128] shared "
                         "across heads (baseline granularity)")
    ap.add_argument("--r2-trust", default="none",
                    choices=["none", "weak", "medium", "hard"])
    ap.add_argument("--r2-radius", type=float, default=None)
    ap.add_argument("--r2-lr", type=float, default=None,
                    help="LR for the R2 generator; default = --lr")
    ap.add_argument("--save-best-val", action="store_true",
                    help="save the TRUE best-validation rotation as --out "
                         "(final-step rotation goes to <out>.final.pt); "
                         "default keeps the legacy final-step save")
    ap.add_argument("--val-metric", default="loss",
                    choices=["loss", "exptau"],
                    help="best-val selection metric: held-out objective "
                         "loss (lower better) or expected tau (higher)")
    ap.add_argument("--objective", default="hybrid",
                    choices=["kl", "tv", "neglog", "hybrid", "hybrid_fixed",
                             "exptau", "topk_kl", "accsurv",
                             # acceptance-aware QAT study (2026-08-12):
                             # conv     = conventional EAGLE loss in-chain
                             #            (SmoothL1(h,t) + 0.1 SoftCE,
                             #            t from corpus a_chain, C7
                             #            transform ((a R1^T)*gamma) R_D)
                             # greedy   = depth-weighted -log q(argmax p)
                             # survival = -expected_tau(alpha_1..K)
                             #            (= -(1 + sum_k prod_j<=k a_j))
                             "conv", "greedy", "survival"])
    ap.add_argument("--conv-depth", default="uniform",
                    choices=["uniform", "gamma"],
                    help="conv objective depth weighting: uniform (matches "
                         "the conventional trainer's uniform token mean) "
                         "or gamma^k (LK convention)")
    ap.add_argument("--rot-fixed-ckpt", default=None,
                    help="load a pretrained R_D from a rotation ckpt "
                         "(e.g. RD_GS_A1_s1001.pt) and hold it FIXED "
                         "(SharedRotation); overrides --rot")
    ap.add_argument("--save-core-steps", default="",
                    help="comma list of optimizer-step counts at which to "
                         "export the original-basis core weights to "
                         "<out>.step<N>.pt (requires --train-draft-core); "
                         "N=0 is the pristine init, N=--steps the final")
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
    ap.add_argument("--first-mode", default="gamma_R1",
                    choices=["gamma_R1", "identity"],
                    help="identity: fp16-target interface (draft "
                         "consumes h_t; W_first=[W_e|W_h], no gamma, "
                         "no R_T bridge fold)")
    ap.add_argument("--onpolicy", default="tf",
                    choices=["tf", "curriculum", "fifty", "onpolicy"])
    ap.add_argument("--rollout", default="greedy",
                    choices=["greedy", "t1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()
    assert not (args.save_best_val and
                (args.train_draft_core or args.train_alpha)), \
        "--save-best-val snapshots rotations only; trained core weights " \
        "or alpha would pair best-step rotations with final-step values"
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
    if args.rot_fixed_ckpt:
        # frozen pretrained R5 (draft-aware residual rotation): the QAT
        # arms train weights UNDER the fixed R_D basis (study section 15)
        rck = torch.load(args.rot_fixed_ckpt, map_location="cpu",
                         weights_only=False)
        rot = SharedRotation(rck["R_D"].float())
        rck_sha = hashlib.sha256(
            open(args.rot_fixed_ckpt, "rb").read()).hexdigest()[:16]
        print(f"[lk] fixed R_D from {args.rot_fixed_ckpt} "
              f"(sha {rck_sha}, trained_step "
              f"{rck.get('meta', {}).get('checkpoint_policy')})",
              flush=True)
    rot = rot.to(dev)

    r2_rot = None
    if args.r2_mode == "residual":
        # R2_B = the deployed baseline (same deterministic seed-0 draw the
        # runtime folds); B=0 init makes R2_D == R2_B bitwise at step 0
        r2_rot = ResidualR2Rotation(fq.baseline_r2(0), trust=args.r2_trust,
                                    radius=args.r2_radius).to(dev)

    if args.first_mode == "identity":
        gamma_eff = torch.ones_like(gamma)
        first_fold = torch.eye(R_T.shape[0])
    else:
        gamma_eff, first_fold = gamma, R_T
    model = ExactQATRotatedDraft(
        sd, R_T, gamma_eff, W_lm, rot, alpha_init=args.alpha_init,
        train_alpha=args.train_alpha, w_bits=4, a_bits=4,
        draft_kv_bits=args.kv_bits, train_draft_core=args.train_draft_core,
        device=dev, first_fold_R=first_fold,
        alpha_rec_init=args.alpha_rec_init, r2_rot=r2_rot)

    windows, man = load_corpus(args.corpus, args.run_dir)
    if args.objective == "conv":
        assert "a_chain" in windows[0], \
            "conv objective needs an a_chain-extended corpus (rebuild " \
            "with the 2026-08-12 builder)"
        assert args.onpolicy == "tf", "conv objective is teacher-forced"
    # conv/diagnostic transform constants (C7 semantics, see
    # train_eagle_draft_int4_qat.py teacher(): t = ((a R1^T)*gamma_f) R_D)
    conv_R1 = R_T.to(dev)
    conv_g = gamma.to(dev).float()
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
    if r2_rot is not None:
        groups.append(dict(params=[p for p in r2_rot.parameters()],
                           lr=(args.r2_lr if args.r2_lr is not None
                               else args.lr)))
    if args.train_alpha:
        groups.append(dict(params=[model.log_alpha], lr=args.lr))
    if args.train_draft_core:
        # b_fc/gl64 are part of the original EAGLE trainable set (see
        # exact_quantized_rotation_forward) — include them so the param
        # audit's trainable list matches what the optimizer steps
        core = [getattr(model, n) for n in model.CORE] + \
            [model.b_fc, model.gl64]
        groups.append(dict(params=core, lr=args.core_lr))
    assert groups, "nothing trainable"
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    # ---- parameter audit (study section 8): prove that ONLY rotation
    # generators (and explicitly opted-in alpha/core) receive gradients,
    # and that every model weight is a frozen buffer
    def _weight_hashes():
        hs = {}
        for n in list(model.CORE) + ["E", "W_lm16", "b_fc"]:
            t = getattr(model, n)
            hs[n] = hashlib.sha256(
                t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return hs

    freeze_before = _weight_hashes()
    audit_rows = [dict(param=n, shape=list(p.shape),
                       requires_grad=bool(p.requires_grad))
                  for n, p in model.named_parameters()]
    trainable_names = sorted(r["param"] for r in audit_rows
                             if r["requires_grad"])
    allowed = {"rot.W", "rot.R_D", "r2_rot.W"}
    if args.train_alpha:
        allowed.add("log_alpha")
    if args.train_draft_core:
        allowed |= set(model.CORE) | {"b_fc", "gl64"}
    unexpected = set(trainable_names) - allowed
    assert not unexpected, f"param audit: unexpected trainable {unexpected}"
    opt_param_ids = {id(p) for gp in groups for p in gp["params"]}
    audit = dict(trainable=trainable_names,
                 optimizer_param_count=len(opt_param_ids),
                 all_params=audit_rows,
                 weight_hashes_before=freeze_before)
    audit_path = os.path.join(
        args.run_dir, "gradchecks",
        f"param_audit__{os.path.basename(args.out)}.json")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    json.dump(audit, open(audit_path, "w"), indent=1)
    print(f"[lk] param audit: trainable={trainable_names} "
          f"-> {audit_path}", flush=True)

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
        conv_t = None
        if args.objective == "conv":
            with torch.no_grad():
                ach = torch.stack([w["a_chain"][:args.K].float()
                                   for w in ws]).to(dev)
                conv_t = ((ach @ conv_R1.t()) * conv_g) @ rot.R().detach()
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
            elif args.objective == "greedy":
                # Q3 (study section 10): hard verifier-token objective,
                # -log q(argmax p_k), gamma^k depth weights (LK convention)
                lk = L.greedy_ce(zTk, zDk)
            elif args.objective == "conv":
                # Q0-in-chain (study sections 11-12): the conventional
                # EAGLE draft loss computed at the SAME K chain positions
                # on the SAME corpus as the acceptance-aware objectives —
                # vloss = SmoothL1(h_k, t_k) with t_k the C7-transformed
                # teacher hidden, ploss = SoftCE through the deployed head
                hkf = _h.squeeze(1)
                tk = conv_t[:, k]
                vloss = F.smooth_l1_loss(hkf.float(), tk,
                                         reduction="none").mean(-1)
                with torch.no_grad():
                    tp = model.head_logits(tk).softmax(-1)
                lp = model.head_logits(hkf).log_softmax(-1)
                lk = vloss + 0.1 * (-(tp * lp).sum(-1))
            elif args.objective == "survival":
                lk = None            # coupled across depths, added below
            else:                                   # exptau
                lk = 0.1 * L.kl_full(zTk, zDk)      # trust regularizer
            alphas.append(L.overlap_alpha(zTk, zDk))
            if args.objective == "survival":
                pass
            elif args.objective == "exptau":
                total = total + lk.mean()
            elif args.objective == "conv" and args.conv_depth == "uniform":
                total = total + lk.mean() / args.K
            else:
                total = total + wk[k] * lk.mean()
            if args.aux_greedy > 0:
                total = total + args.aux_greedy * wk[k] * \
                    L.greedy_ce(zTk, zDk).mean()
        a_stack = torch.stack(alphas, dim=-1)       # (B, K)
        if args.objective == "exptau":
            total = total + L.expected_tau_loss(a_stack).mean()
        elif args.objective == "survival":
            # Q2 (study section 9): L = -R_surrogate = -sum_k prod_j<=k
            # alpha_j; expected_tau = 1 + R_surrogate (constant offset,
            # identical gradient; alphas carry grad through min(p,q))
            total = total - L.expected_tau(a_stack).mean()
        elif args.objective == "accsurv":
            total = total - L.prefix_survival(
                torch.stack(logps, dim=-1)).mean()
        return total, a_stack.detach()

    @torch.no_grad()
    def diag_panel(ws):
        """Cross-objective diagnostic panel (AA-QAT study section 16):
        the SAME metric set logged whatever objective is trained, on the
        held-out validation windows. Per-depth means over the chunk."""
        tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
        a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
        teach = torch.stack([w["teacher_tokens"][:args.K].long()
                             for w in ws]).to(dev)
        zT = torch.stack([w["teacher_logits"][:args.K].float()
                          for w in ws]).to(dev)
        has_ach = "a_chain" in ws[0]
        tgt = None
        if has_ach:
            ach = torch.stack([w["a_chain"][:args.K].float()
                               for w in ws]).to(dev)
            tgt = ((ach @ conv_R1.t()) * conv_g) @ rot.R()
        outs = model.forward_chain(tok, a, args.K, recur_tokens=teach)
        per = {m: [] for m in ("alpha", "kl", "tv", "hyb", "gce", "top1",
                               "logq", "rank", "sl1", "cos", "nmse",
                               "conv")}
        als, lqs = [], []
        for k, (lg, hk, _c) in enumerate(outs):
            zTk = zT[:, k]
            al = L.overlap_alpha(zTk, lg)
            hy, _lam, _a2 = L.hybrid_lk(zTk, lg, eta=args.eta)
            lq = L.teacher_token_logp(lg, teach[:, k])
            als.append(al)
            lqs.append(lq)
            per["alpha"].append(float(al.mean()))
            per["kl"].append(float(L.kl_full(zTk, lg).mean()))
            per["tv"].append(float(L.tv(zTk, lg).mean()))
            per["hyb"].append(float(hy.mean()))
            per["gce"].append(float(L.greedy_ce(zTk, lg).mean()))
            per["top1"].append(float((lg.argmax(-1) == teach[:, k])
                                     .float().mean()))
            per["logq"].append(float(lq.mean()))
            zy = lg.gather(-1, teach[:, k].unsqueeze(-1))
            per["rank"].append(float((lg >= zy).sum(-1).float().mean()))
            if has_ach:
                hf = hk.squeeze(1).float()
                tk_ = tgt[:, k]
                sl1 = F.smooth_l1_loss(hf, tk_,
                                       reduction="none").mean(-1)
                per["sl1"].append(float(sl1.mean()))
                per["cos"].append(float(
                    F.cosine_similarity(hf, tk_, dim=-1).mean()))
                per["nmse"].append(float(
                    ((hf - tk_).pow(2).sum(-1)
                     / tk_.pow(2).sum(-1).clamp_min(1e-9)).mean()))
                tp = model.head_logits(tk_).softmax(-1)
                lp = model.head_logits(hk.squeeze(1)).log_softmax(-1)
                per["conv"].append(float(
                    (sl1 + 0.1 * (-(tp * lp).sum(-1))).mean()))
        a_st = torch.stack(als, dim=-1)
        per["expected_tau"] = float(L.expected_tau(a_st).mean())
        per["survival"] = float(L.expected_tau(a_st).mean() - 1.0)
        per["greedy_survival"] = float(L.prefix_survival(
            torch.stack(lqs, dim=-1)).mean())
        per["n"] = len(ws)
        return per

    core_steps = (sorted({int(x) for x in args.save_core_steps.split(",")
                          if x.strip() != ""})
                  if args.save_core_steps else [])
    if core_steps:
        assert args.train_draft_core, "--save-core-steps needs core"

    def save_core_ckpt(nstep):
        """Original-basis draft state dict in the conv-QAT export format
        (eval-compatible: {'draft_state_dict': ..., 'meta': ...})."""
        ex = model.export_original_state()
        sd_out = {k: v.clone() for k, v in sd.items()}
        sd_out["fc.weight"] = torch.cat([ex["W_e"], ex["W_h"]], dim=1)
        sd_out["fc.bias"] = ex["b_fc"]
        m = {"Wq": "self_attn.q_proj.weight",
             "Wk": "self_attn.k_proj.weight",
             "Wv": "self_attn.v_proj.weight",
             "Wo": "self_attn.o_proj.weight",
             "Wgate": "mlp.gate_proj.weight",
             "Wup": "mlp.up_proj.weight",
             "Wdown": "mlp.down_proj.weight"}
        for k, v in m.items():
            sd_out[f"layers.0.{v}"] = ex[k]
        sd_out["layers.0.post_attention_layernorm.weight"] = ex["gl"]
        p = f"{args.out}.step{nstep:04d}.pt"
        torch.save({"draft_state_dict":
                    {k: v.half() for k, v in sd_out.items()},
                    "meta": dict(step=nstep, objective=args.objective,
                                 seed=args.seed, core_lr=args.core_lr,
                                 lr=args.lr, K=args.K,
                                 alpha=(model.alpha_exact
                                        if not args.train_alpha
                                        else float(model.log_alpha.exp())),
                                 alpha_rec=model.alpha_rec_exact,
                                 rot_fixed_ckpt=args.rot_fixed_ckpt,
                                 out=args.out)}, p)
        print(f"[lk] core ckpt step {nstep} -> {p}", flush=True)

    best_val = None
    best_state = None
    for step in range(args.steps):
        if step in core_steps:
            save_core_ckpt(step)
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
        if r2_rot is not None:
            pen = pen + r2_rot.penalty()
        if float(pen) != 0.0:
            pen.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for gp in groups for p in gp["params"]], args.clip)
        opt.step()
        sched.step()
        rot.post_step()
        if r2_rot is not None:
            r2_rot.post_step()
        if hasattr(rot, "orth_error"):
            oe = rot.orth_error()
            assert oe < 1e-3, f"Gate G: orth {oe}"
        if r2_rot is not None:
            oe2 = r2_rot.orth_error()
            assert oe2 < 1e-3, f"Gate R2-D: orth {oe2}"
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
                va, vls = [], []
                for i0 in range(0, min(len(W_va), 64), 16):
                    lw_, aw = batch_loss(W_va[i0:i0 + 16], False)
                    va.append(aw)
                    vls.append(float(lw_))
                vam = torch.cat(va).mean(0)
            et = float(L.expected_tau(torch.cat(va)).mean())
            vloss = sum(vls) / max(len(vls), 1)
            r2_oe = (r2_rot.orth_error() if r2_rot is not None else None)
            r2_gen = (r2_rot.generator_frob_norm()
                      if r2_rot is not None else None)
            print(f"[lk] VAL step {step}: alpha_by_depth="
                  f"{[round(float(x),4) for x in vam]} "
                  f"expected_tau={et:.4f} val_loss={vloss:.5f}"
                  + (f" r2_orth={r2_oe:.2e} r2_genB={r2_gen:.4f}"
                     if r2_rot is not None else ""), flush=True)
            log.append(dict(step=step, val_alpha=[float(x) for x in vam],
                            val_expected_tau=et, val_loss=vloss,
                            r2_orth_error=r2_oe,
                            r2_generator_frob=r2_gen))
            # AA-QAT study: objective-independent diagnostic panel on the
            # same held-out windows (chunked; equal chunk sizes -> mean of
            # chunk means == pooled mean)
            pans = [diag_panel(W_va[i0:i0 + 16])
                    for i0 in range(0, min(len(W_va), 64), 16)]
            panel = {}
            for key in pans[0]:
                if key == "n":
                    panel["n"] = sum(p["n"] for p in pans)
                elif isinstance(pans[0][key], list):
                    if pans[0][key]:
                        panel[key] = [round(sum(p[key][i] for p in pans)
                                            / len(pans), 5)
                                      for i in range(len(pans[0][key]))]
                else:
                    panel[key] = round(sum(p[key] for p in pans)
                                       / len(pans), 5)
            log.append(dict(step=step, panel=panel))
            print(f"[lk] PANEL step {step}: alpha={panel['alpha']} "
                  f"top1={panel['top1']} exp_tau="
                  f"{panel['expected_tau']:.4f}"
                  + (f" sl1={panel['sl1']}" if panel.get("sl1") else ""),
                  flush=True)
            metric = vloss if args.val_metric == "loss" else et
            cand_ok = not math.isnan(metric)
            incumbent_nan = (best_val is not None
                             and math.isnan(best_val["metric"]))
            better = cand_ok and (best_val is None or incumbent_nan or
                                  (metric < best_val["metric"]
                                   if args.val_metric == "loss"
                                   else metric > best_val["metric"]))
            if not cand_ok:
                print(f"[lk] WARN: val metric NaN at step {step} — "
                      f"skipped for best-val selection", flush=True)
            if better:
                best_val = dict(metric=metric, step=step,
                                val_loss=vloss, val_expected_tau=et)
                best_state = dict(
                    step=step,
                    R_D=rot.R().detach().cpu(),
                    rot_W=(rot.W.detach().cpu()
                           if hasattr(rot, "W") else None),
                    R2_D=(r2_rot.R64().detach().cpu()
                          if r2_rot is not None else None),
                    R2_W=(r2_rot.W.detach().cpu()
                          if r2_rot is not None else None))

    if args.steps in core_steps:
        save_core_ckpt(args.steps)

    # ---- weight-freeze proof: model weights bit-identical before/after ----
    freeze_after = _weight_hashes()
    if not args.train_draft_core:
        assert freeze_after == freeze_before, \
            f"weight freeze violated: {freeze_before} -> {freeze_after}"

    # ---- checkpoint selection: TRUE best-val vs legacy final-step ---------
    final_state = dict(
        step=args.steps - 1,
        R_D=rot.R().detach().cpu(),
        rot_W=(rot.W.detach().cpu() if hasattr(rot, "W") else None),
        R2_D=(r2_rot.R64().detach().cpu() if r2_rot is not None else None),
        R2_W=(r2_rot.W.detach().cpu() if r2_rot is not None else None))
    use_best = bool(args.save_best_val and best_state is not None)
    sel = best_state if use_best else final_state
    ckpt_policy = dict(policy=("best_val" if use_best else "final_step"),
                       metric=args.val_metric, selected_step=sel["step"],
                       best_val=best_val)

    # The 4096x4096 geodesic (complex eigvals) dominates post-training
    # wall clock, so snapshot what the save needs from the model, drop
    # the model, and release the card BEFORE computing it — otherwise a
    # finished job holds a GPU for tens of minutes of pure CPU work.
    save_alpha = (model.alpha_exact if not args.train_alpha
                  else float(model.log_alpha.exp()))
    save_alpha_rec = model.alpha_rec_exact
    save_counters = dict(model.counters)
    save_core = (model.export_original_state()
                 if args.train_draft_core else None)
    del model
    torch.cuda.empty_cache()
    _geo_cache = {}

    def _mk_save(state):
        key = hashlib.sha256(
            state["R_D"].numpy().tobytes()).hexdigest()
        if key not in _geo_cache:
            _geo_cache[key] = rotation_geometry(state["R_D"], R_T)
        geo = _geo_cache[key]
        r2_geo = None
        if state["R2_D"] is not None:
            r2_geo = rotation_geometry(state["R2_D"].float(),
                                       r2_rot.R2_B64.cpu().float())
            B = state["R2_W"] - state["R2_W"].t()
            r2_geo["generator_frob_norm"] = float(B.norm())
        sv = dict(R_D=state["R_D"],
                  alpha=save_alpha,
                  alpha_rec=save_alpha_rec,
                  meta=dict(vars(args), corpus_n=len(windows),
                            corpus_manifest_sha=hashlib.sha256(
                                open(args.corpus, "rb").read())
                            .hexdigest()[:16],
                            geometry=geo, r2_geometry=r2_geo,
                            best_val=best_val,
                            checkpoint_policy=dict(ckpt_policy,
                                                   saved_step=state["step"]),
                            weight_freeze=dict(before=freeze_before,
                                               after=freeze_after),
                            divergence_rates=div.rates(),
                            counters=save_counters),
                  log=log)
        if state["R2_D"] is not None:
            sv["R2_D"] = state["R2_D"]          # fp64, runtime fold input
            sv["R2_W"] = state["R2_W"]          # generator (for controls)
        if save_core is not None:
            sv["core_weights"] = save_core
        return sv

    torch.save(_mk_save(sel), args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    open(args.out + ".sha256", "w").write(sha + "\n")
    if use_best and sel["step"] != final_state["step"]:
        torch.save(_mk_save(final_state), args.out + ".final.pt")
    print(f"[lk] saved {args.out} sha={sha[:16]} "
          f"policy={ckpt_policy['policy']} step={sel['step']} "
          f"best_val={best_val}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
