#!/usr/bin/env python
"""Multi-step quantized-rollout QAT (study spec §19) — NON-OFFICIAL
extension, run only after the single-step baseline.

Dense K-step rollout from the fresh FP16 anchor: depth 0 is the official
single-step pass (interface fold, W_first); depths k=1..K-1 feed the
draft's OWN predicted hidden back through the deployed RECURRENT fold
(W_rec, no interface fold, exact contract), with teacher-forced target
tokens and teacher hidden targets shifted by k. Attention at each depth
is causal over position-aligned rows (dense-rollout approximation of the
deployment KV — declared; this extension is not the official recipe).

Loss (declared extension):
  sum_k w_k * [ SmoothL1(pred_k, h_{t+k+1}) + 0.1 * softCE_k ]
  M1: w_k uniform (1/K) · M2: w_k ∝ gamma^(k-1), gamma in {0.7, 0.8}

Reuses the validated single-step machinery (data split, teacher builds,
leaf-grad accumulation is NOT used here — rollout graphs are chained, so
plain per-step backward with once-per-optimizer-step quantized weights).
"""
import argparse, importlib.util, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

spec = importlib.util.spec_from_file_location(
    "sst", os.path.join(PROJECT_ROOT, "scripts",
                        "train_eagle_draft_int4_qat.py"))
sst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sst)

from eagle_spinquant import experiment, study
from eagle_spinquant.exact_quantized_rotation_forward import (
    ExactQuantizedRotationForward)

KIND = "learned_chat_w4a4kv16"
D = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["Q0", "Q1"])
    ap.add_argument("--weights", required=True,
                    choices=["M1", "M2g07", "M2g08"])
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-ckpt-every", type=int, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"MS{args.weights}_{args.variant}_s{args.seed}"
    done = os.path.join(args.run_dir, "manifests", f"train_{tag}.json")
    if os.path.exists(done):
        print(f"[{tag}] already complete; skip")
        return 0
    dev = "cuda:0"
    t0 = time.time()
    torch.manual_seed(1000 + args.seed)

    if args.weights == "M1":
        w = [1.0] * args.K
    else:
        g = 0.7 if args.weights == "M2g07" else 0.8
        w = [g ** k for k in range(args.K)]
    w = [x / sum(w) for x in w]

    int4_teacher = args.variant == "Q1"
    rotated = int4_teacher
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, tq = (("full", "w4a4") if int4_teacher else ("none", "none"))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, tq, 0, device=dev, rotations_root=rr)
    model.base_model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    from eagle_spinquant import eagle_bridge
    tok = eagle_bridge.get_tokenizer(model)
    blob = sst.build_tokenized_split(tok)
    rows = blob["rows"]
    train_rows = rows[:sst.TRAIN_N]
    val_rows = rows[sst.TRAIN_N:sst.TRAIN_N + sst.VAL_N]

    ea_sd = {k: v.detach().cpu() for k, v in
             model.ea_layer.state_dict().items()}
    init = torch.load(args.anchor, map_location="cpu",
                      weights_only=False)
    init = init.get("draft_state_dict", init.get("model", init))
    for k in ea_sd:
        if k in init:
            ea_sd[k] = init[k].to(ea_sd[k].dtype)

    if "R1" in stash and stash["R1"] is not None:
        R1 = stash["R1"].float()
    else:
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                        rr),
                       map_location="cpu", weights_only=False)
        R1 = R["R1"].float()
    W_lm = stash["lm_head_weight"].float()
    rot_shim = sst._FixedRot(R1.to(dev))
    gamma = (stash["gamma_f"].float() if rotated else torch.ones(D))
    ffold = (R1 if rotated else torch.eye(D))
    core = ExactQuantizedRotationForward(
        ea_sd, R1, gamma, W_lm, rot_shim, alpha_init=args.alpha,
        train_alpha=False, w_bits=4, a_bits=4, draft_kv_bits=16,
        train_draft_core=True, r2_seed=0, device=dev,
        first_fold_R=ffold).to(dev)
    core.log_alpha.requires_grad_(False)
    trainable = [p for p in core.parameters() if p.requires_grad]

    R1d = R1.to(dev).float()
    gd = (stash["gamma_f"].float().to(dev) if "gamma_f" in stash
          else model.base_model.model.norm.weight.detach().float()
          .to(dev))

    def teacher(ids, am):
        with torch.no_grad():
            B, T = ids.shape
            c = max(1, 4096 // max(T, 1))
            hs = [model.base_model.model(
                input_ids=ids[i:i + c].to(dev),
                attention_mask=am[i:i + c].to(dev)).last_hidden_state
                .float() for i in range(0, B, c)]
            h = torch.cat(hs, dim=0) if len(hs) > 1 else hs[0]
        if not rotated:
            return h, h @ R1d
        return h, ((h @ R1d.t()) * gd) @ R1d

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, args.warmup, args.steps)
    crit = torch.nn.SmoothL1Loss(reduction="none")
    gsh = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_rows), generator=gsh).tolist()
    it = sst.make_batches(train_rows, order, 2)   # teacher batch of 2
    os.makedirs(os.path.join(args.run_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "manifests"), exist_ok=True)
    logf = open(os.path.join(args.run_dir, "logs",
                             f"train_{tag}.jsonl"), "a")

    def export(path, meta):
        ex = core.export_original_state()
        sd_out = dict(ea_sd)
        sd_out["fc.weight"] = torch.cat([ex["W_e"], ex["W_h"]], dim=1)
        sd_out["fc.bias"] = ex["b_fc"]
        m = {"Wq": "self_attn.q_proj.weight",
             "Wk": "self_attn.k_proj.weight",
             "Wv": "self_attn.v_proj.weight",
             "Wo": "self_attn.o_proj.weight",
             "Wgate": "mlp.gate_proj.weight",
             "Wup": "mlp.up_proj.weight",
             "Wdown": "mlp.down_proj.weight"}
        for kk, v in m.items():
            sd_out[f"layers.0.{v}"] = ex[kk]
        sd_out["layers.0.post_attention_layernorm.weight"] = ex["gl"]
        torch.save({"draft_state_dict":
                    {k: v.half() for k, v in sd_out.items()},
                    "meta": meta}, path)

    def rollout_loss(ids, lm, am, lens, train=True, qw=None):
        """Per-conversation backward (train mode) with leaf-grad
        accumulation; softCE rows capped at 512/depth via unbiased
        random subsample (full-vocab logit graphs for K depths x convs
        do not fit 24 GB otherwise; SmoothL1 keeps all rows)."""
        x, t = teacher(ids, am)
        total_v = total_p = 0.0
        did = 0
        for b, L in enumerate(lens):
            if train:
                noise = (torch.rand(L, D, device=x.device) - 0.5) \
                    * 0.2 * 512 / L
                x[b, :L] += noise
        for b, L in enumerate(lens):
            if L < args.K + 8:
                continue
            hcur = x[b:b + 1, :L - 1].half()      # depth-0 input (teacher)
            loss = 0.0
            for k in range(args.K):
                Tk = L - 1 - k
                tokk = ids[b:b + 1, k:L].to(dev)   # tokens t_{i+k+1} rows
                h = core.forward_train(tokk, hcur[:, :Tk], qw=qw,
                                       proj="first" if k == 0 else "rec")
                sel = (lm[b:b + 1, k:L - 1]).to(dev).reshape(-1)
                nsel = sel.sum() + 1e-5
                hf = h.reshape(-1, D)[sel]
                tf = t[b:b + 1, k + 1:L].reshape(-1, D)[sel].float()
                v = crit(hf.float(), tf).mean(-1).sum() / nsel
                nh = hf.shape[0]
                if nh > 512:
                    ridx = torch.randperm(nh, device=hf.device)[:512]
                    hf_p, tf_p = hf[ridx], tf[ridx]
                else:
                    hf_p, tf_p = hf, tf
                with torch.no_grad():
                    tp = core.head_logits(tf_p).softmax(-1)
                lp = core.head_logits(hf_p).log_softmax(-1)
                pl = -(tp * lp).sum(-1).sum() / (hf_p.shape[0] + 1e-5)
                loss = loss + w[k] * (v + 0.1 * pl) / len(lens)
                total_v += float(v) * w[k] / len(lens)
                total_p += float(pl) * w[k] / len(lens)
                hcur = h.detach() if not train else h
                del tp, lp
            if torch.is_tensor(loss):
                did += 1
                if train:
                    loss.backward()        # frees this conv's graph
        return did, total_v, total_p

    best3, step = -1.0, 0
    core.train()
    while step < args.steps:
        try:
            ids, lm, am, lens = next(it)
        except StopIteration:
            order = torch.randperm(len(train_rows),
                                   generator=gsh).tolist()
            it = sst.make_batches(train_rows, order, 2)
            ids, lm, am, lens = next(it)
        opt.zero_grad(set_to_none=True)
        # leaf-grad scheme (validated in the single-step trainer):
        # quantize ONCE per optimizer step; grads accumulate on detached
        # leaves across depths/convs; fold graph traversed once (STE-
        # exact). Avoids K x convs fold-graph rebuilds (OOM on 24 GB).
        leaves, twR = core.qw_leaves(exact=False)
        did, tv, tp_ = rollout_loss(ids, lm, am, lens,
                                    qw=(leaves, twR))
        if not did:
            continue
        core.fold_backward(leaves)
        del leaves
        torch.nn.utils.clip_grad_value_(trainable, 0.5)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            logf.write(json.dumps(dict(step=step, vloss=round(tv, 5),
                                       ploss=round(tp_, 5),
                                       lr=sched.get_last_lr()[0],
                                       hours=round((time.time() - t0)
                                                   / 3600, 3))) + "\n")
            logf.flush()
        if args.save_ckpt_every and step % args.save_ckpt_every == 0:
            export(os.path.join(args.run_dir, "ckpts",
                                f"{tag}_step{step}.pt"),
                   dict(tag=tag, step=step))
        if step % args.val_every == 0 or step == args.steps:
            core.eval()
            with torch.no_grad():
                vqw = core.quantized_weights(exact=False)
                vs = []
                vit = sst.make_batches(val_rows[:60], list(range(60)), 2)
                for vids, vlm, vam, vlens in vit:
                    _, vv, _ = rollout_loss(vids, vlm, vam, vlens,
                                            train=False, qw=vqw)
                    vs.append(vv)
            del vqw                      # free val fold tensors before
            torch.cuda.empty_cache()     # resuming training (OOM)
            core.train()
            vrec = dict(step=step, kind="val",
                        vloss=round(sum(vs) / max(len(vs), 1), 5))
            logf.write(json.dumps(vrec) + "\n")
            logf.flush()
            print(f"[{tag}] VAL {vrec}", flush=True)
            export(os.path.join(args.run_dir, "ckpts",
                                f"{tag}_last.pt"),
                   dict(tag=tag, step=step, val=vrec))
    with open(done, "w") as f:
        json.dump(dict(tag=tag, steps=args.steps, K=args.K,
                       weights=args.weights, w=w, lr=args.lr,
                       gpu_hours=round((time.time() - t0) / 3600, 3)),
                  f, indent=1)
    print(f"[{tag}] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
