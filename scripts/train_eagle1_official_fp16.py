#!/usr/bin/env python
"""Official-recipe EAGLE-1 FP16 draft training from a FRESH initialization
(study: exp/eagle1-official-fromscratch-ptq-vs-qat; recipe audit:
docs/EAGLE1_OFFICIAL_RECIPE_AUDIT.md).

Faithful to the executable official v1 code:
  - cnets.Model(config, load_emb=True, path=<target>): fresh PyTorch-
    default-init decoder layer + fc, target embedding copied & frozen,
    frozen target LM head for the soft-CE loss, gradient checkpointing.
  - records: row i input = (hidden[i]+noise, token[i+1]), target =
    hidden[i+1]; loss = 1.0*SmoothL1(masked) + 0.1*softCE(masked);
    AdamW(lr 3e-5, betas (0.9,0.95), DEFAULT weight_decay 0.01),
    clip_grad_value_(0.5); linear warmup 2000 with FIXED 800000-step
    horizon; bs 4/process; bf16 autocast; 21 epochs (official
    range(num_epochs+1)); official 95/5 split by deterministic order;
    validation top-1/2/3 on the test split each epoch.
  - teacher: FUSED frozen FP16 target forward at bs=1 per conversation
    (bitwise-identical records to the audited official pre-generation;
    declared deviation D1 - storage).

DDP: torchrun --nproc_per_node=N; per-rank bs stays 4 (official code
does not scale LR with world size; effective global batch = 4N is
recorded). --bench N runs a fixed number of optimizer steps and exits
(for the pre-registered 1/2/4/8-GPU benchmark).
"""
import argparse, hashlib, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.distributed as dist
import torch.nn.functional as F

TOK_CACHE = ("/data/thahn1230/datasets/eagle1_official/"
             "official_tok_68k_v1.pt")
CFG_JSON = os.path.join(PROJECT_ROOT,
                        "third_party/EAGLE/eagle/train/"
                        "llama_2_chat_7B_config.json")
TC = dict(lr=3e-5, bs=4, num_epochs=20, num_warmup_steps=2000,
          total_steps=800000, p_w=0.1, v_w=1.0, noise_std=0.2,
          max_len=2048, b1=0.9, b2=0.95, grad_clip=0.5)


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def build_models(dev, seed):
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    from eagle_spinquant import experiment
    from transformers import AutoModelForCausalLM
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    target = AutoModelForCausalLM.from_pretrained(
        paths["target_path"], torch_dtype=torch.float16).to(dev).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    head = target.lm_head            # frozen fp16 target head (official)
    econf = EConfig.from_pretrained(CFG_JSON)
    torch.manual_seed(seed)          # fresh init determinism per seed
    model = Model(econf, load_emb=True, path=paths["target_path"],
                  bias=True)
    model = model.to(dev).float()    # fp32 masters, bf16 autocast compute
    model.embed_tokens.weight.data = \
        model.embed_tokens.weight.data.float()
    return target, head, model, paths


def official_batch(rows, idxs, target, dev, noise_gen):
    """Build the official padded batch record with the fused teacher
    (bs=1 teacher forwards -> bitwise parity with pre-generation)."""
    hs, ids_l, lm_l = [], [], []
    with torch.no_grad():
        for i in idxs:
            r = rows[i]
            ids = r["input_ids"].long()[None].to(dev)
            h = target.model(input_ids=ids).last_hidden_state[0].float()
            hs.append(h)
            ids_l.append(r["input_ids"].long())
            lm_l.append(r["loss_mask"].float())
    T = max(x.shape[0] for x in hs)
    B = len(hs)
    hidden = torch.zeros(B, T, 4096, device=dev)
    target_h = torch.zeros(B, T, 4096, device=dev)
    input_ids = torch.zeros(B, T, dtype=torch.long, device=dev)
    loss_mask = torch.zeros(B, T, device=dev)
    attn = torch.zeros(B, T, dtype=torch.long, device=dev)
    for b in range(B):
        L = hs[b].shape[0]
        # official CustomDataset: input token row i = ids[i+1] (0-pad
        # tail), target row i = hidden[i+1] (zero-pad), mask[-1]=0
        input_ids[b, :L - 1] = ids_l[b][1:].to(dev)
        target_h[b, :L - 1] = hs[b][1:]
        lm = lm_l[b].clone()
        lm[-1] = 0
        loss_mask[b, :L] = lm.to(dev)
        attn[b, :L] = 1
        # official AddUniformNoise: input hidden only, (rand-.5)*std*512/L
        noise = (torch.rand(L, 4096, device=dev,
                            generator=noise_gen) - 0.5) \
            * TC["noise_std"] * 512 / L
        hidden[b, :L] = hs[b] + noise
    return hidden, input_ids, target_h, loss_mask, attn


def compute_loss(model, head, batch, crit):
    """Official main.py loss on masked rows only — mathematically
    identical to the full-tensor form (masked terms are zero in the
    official sums; verified by test_eagle1_official_loss) but avoids
    materializing (B,T,V) logits, which OOMs on 24 GB cards."""
    hidden, input_ids, target_h, loss_mask, attn = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predict = model(hidden, input_ids=input_ids,
                        attention_mask=attn)
        sel = loss_mask.bool().reshape(-1)
        nsel = loss_mask.sum() + 1e-5
        pf = predict.reshape(-1, predict.shape[-1])[sel]
        tf = target_h.reshape(-1, target_h.shape[-1])[sel]
        with torch.no_grad():
            target_head = head(tf.half()).float()
            target_p = torch.softmax(target_head, dim=-1)
        out_head = head(pf.half()).float()
        out_logp = torch.log_softmax(out_head, dim=-1)
        ploss = -(target_p * out_logp).sum() / nsel
        v = crit(pf.float(), tf.float())
        vloss = v.mean(-1).sum() / nsel
        loss = TC["v_w"] * vloss + TC["p_w"] * ploss
    return loss, vloss.detach(), ploss.detach(), out_head.detach(), \
        target_head


@torch.no_grad()
def validate(model, head, rows, test_idx, target, dev, crit, bs, cap):
    model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    vl = correct = c2 = c3 = tot = 0.0
    nb = 0
    for s in range(0, min(len(test_idx), cap), bs):
        batch = official_batch(rows, test_idx[s:s + bs], target, dev, g)
        loss, vloss, ploss, out_head, target_head = compute_loss(
            model, head, batch, crit)
        # out_head/target_head are already masked-row (N, V)
        pk = out_head.topk(3, dim=-1).indices
        gt = target_head.argmax(-1)
        eq = pk.eq(gt.unsqueeze(-1))
        correct += eq[..., 0].sum().item()
        c2 += eq[..., :2].any(-1).sum().item()
        c3 += eq.any(-1).sum().item()
        tot += out_head.shape[0]
        vl += vloss.item()
        nb += 1
    model.train()
    return dict(vloss=vl / max(nb, 1), top1=correct / max(tot, 1),
                top2=c2 / max(tot, 1), top3=c3 / max(tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0,
                    help="run N optimizer steps then exit (benchmark)")
    ap.add_argument("--pilot", type=int, default=0,
                    help="pilot: N steps + frozen-param + resume checks")
    ap.add_argument("--epochs", type=int,
                    default=TC["num_epochs"] + 1)   # official off-by-one
    ap.add_argument("--save-every-steps", type=int, default=2000)
    ap.add_argument("--val-every-steps", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    t0 = time.time()

    blob = torch.load(TOK_CACHE, weights_only=False)
    rows, split = blob["rows"], blob["split"]
    train_idx_all = list(range(split))
    test_idx = list(range(split, len(rows)))

    target, head, model, paths = build_models(dev, args.seed)
    frozen_emb_sha = hashlib.sha256(
        model.embed_tokens.weight.data.half().cpu().numpy().tobytes()
    ).hexdigest()[:24]
    frozen_head_sha = hashlib.sha256(
        head.weight.data.cpu().numpy().tobytes()).hexdigest()[:24]

    raw = model
    if world > 1:
        # static_graph: cnets uses reentrant gradient checkpointing,
        # which needs a static DDP graph to avoid double-ready errors
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], static_graph=True)
        raw = model.module
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=TC["lr"],
                            betas=(TC["b1"], TC["b2"]))  # wd default .01
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(
        opt, TC["num_warmup_steps"], TC["total_steps"])
    crit = torch.nn.SmoothL1Loss(reduction="none")

    start_step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu",
                        weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step, start_epoch = ck["step"], ck["epoch"]
        log(rank, f"[fp16] resumed step {start_step} ep {start_epoch}")

    log(rank, f"[fp16] world={world} bs/rank={TC['bs']} eff_batch="
              f"{TC['bs']*world} trainable={n_tr/1e6:.1f}M "
              f"train={len(train_idx_all)} test={len(test_idx)} "
              f"epochs={args.epochs} emb_sha={frozen_emb_sha} "
              f"head_sha={frozen_head_sha}")
    os.makedirs(os.path.join(args.run_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "logs"), exist_ok=True)
    logf = (open(os.path.join(args.run_dir, "logs",
                              f"fp16_train_s{args.seed}.jsonl"), "a")
            if rank == 0 else None)

    step = start_step
    limit = args.bench or args.pilot or 10**12
    noise_gen = torch.Generator(device=dev)
    noise_gen.manual_seed(args.seed * 100003 + rank)
    t_bench = None
    for epoch in range(start_epoch, args.epochs):
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        perm = torch.randperm(len(train_idx_all), generator=g).tolist()
        my = perm[rank::world]
        # equalize per-rank step counts (avoids end-of-epoch DDP hang)
        n_common = (min(len(perm[r::world]) for r in range(world))
                    // TC["bs"]) * TC["bs"]
        my = my[:n_common]
        model.train()
        for s in range(0, len(my) - TC["bs"] + 1, TC["bs"]):
            idxs = [train_idx_all[j] for j in my[s:s + TC["bs"]]]
            batch = official_batch(rows, idxs, target, dev, noise_gen)
            opt.zero_grad(set_to_none=True)
            loss, vloss, ploss, _, _ = compute_loss(model, head, batch,
                                                    crit)
            loss.backward()
            gn = torch.nn.utils.clip_grad_value_(trainable,
                                                 TC["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step == 10:
                t_bench = time.time()
            if rank == 0 and step % 25 == 0:
                rec = dict(step=step, epoch=epoch,
                           vloss=round(float(vloss), 5),
                           ploss=round(float(ploss), 5),
                           lr=sched.get_last_lr()[0],
                           hours=round((time.time() - t0) / 3600, 3))
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if step % 200 == 0:
                    log(rank, f"[fp16] {rec}")
            if rank == 0 and step % args.val_every_steps == 0:
                v = validate(raw, head, rows, test_idx, target, dev,
                             crit, TC["bs"], cap=200)
                v.update(step=step, epoch=epoch, kind="val")
                logf.write(json.dumps(v) + "\n")
                logf.flush()
                log(rank, f"[fp16] VAL {v}")
            if rank == 0 and step % args.save_every_steps == 0:
                torch.save(dict(model=raw.state_dict(),
                                opt=opt.state_dict(),
                                sched=sched.state_dict(), step=step,
                                epoch=epoch, seed=args.seed),
                           os.path.join(args.run_dir, "ckpts",
                                        "fp16_fresh_last.pt"))
            if step - start_step >= limit:
                break
        if step - start_step >= limit:
            break
        if world > 1:
            dist.barrier()

    if args.bench and rank == 0:
        n = step - start_step - 10
        dt = time.time() - t_bench
        print(f"[bench] world={world} steps={n} s/step={dt/max(n,1):.3f}"
              f" eff_tokens/s="
              f"{n*TC['bs']*world*1200/dt:.0f}", flush=True)
    if args.pilot and rank == 0:
        emb2 = hashlib.sha256(raw.embed_tokens.weight.data.half().cpu()
                              .numpy().tobytes()).hexdigest()[:24]
        head2 = hashlib.sha256(head.weight.data.cpu().numpy()
                               .tobytes()).hexdigest()[:24]
        p = os.path.join(args.run_dir, "ckpts", "fp16_fresh_last.pt")
        torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step, epoch=0,
                        seed=args.seed), p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m2 = {k: v for k, v in ck["model"].items()}
        ok = all(torch.equal(m2[k].cpu(), v.detach().cpu())
                 for k, v in raw.state_dict().items())
        print(json.dumps(dict(
            pilot="PASS" if (emb2 == frozen_emb_sha and
                             head2 == frozen_head_sha and ok) else "FAIL",
            emb_frozen=emb2 == frozen_emb_sha,
            head_frozen=head2 == frozen_head_sha,
            save_load_roundtrip=ok, steps=step)), flush=True)
    if rank == 0 and not (args.bench or args.pilot):
        torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step,
                        epoch=args.epochs, seed=args.seed),
                   os.path.join(args.run_dir, "ckpts",
                                "fp16_fresh_final.pt"))
        log(rank, f"[fp16] DONE step={step} "
                  f"hours={(time.time()-t0)/3600:.2f}")
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
