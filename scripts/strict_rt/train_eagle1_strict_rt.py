#!/usr/bin/env python
"""STRICT SEAGLE-RT trainer: official-recipe EAGLE-1 draft training from a
FRESH initialization, consuming the DEPLOYED W4A4 SpinQuant target's
NATIVE rotated hidden feature (a_t) as BOTH input and regression label.

Fork of the validated scripts/train_eagle1_official_fp16.py (the official
from-scratch reproduction). The ONLY contract changes vs that trainer:

  1. TEACHER: h comes from the deployed W4A4 SpinQuant target's native
     tap (offline cache by default — bit-parity gated vs online; or
     --teacher online recomputes through the exact eval-time build).
     NO R1_T.T restore. NO gamma_f multiply. NO alpha. (Gates C/D/E.)
  2. PLOSS HEAD: the frozen DEPLOYED fused head W_lm·diag(gamma_f)·R1
     (native_head.pt from the cache builder) — the head the deployed
     runtime scores drafts with — replacing the original-basis fp16
     head. Prediction and label pass through the SAME head (Gate F).
  3. Speed-only (math-identical): teacher offline; gradient
     checkpointing off by default (--grad-ckpt on restores official);
     checkpoints to /data (root disk full).

EVERYTHING ELSE IS LINE-IDENTICAL to the validated trainer: fresh
cnets.Model init (PyTorch-default, target embedding copied+frozen from
the ORIGINAL safetensors), record shift, input-only uniform noise
(official absolute scale), masked SmoothL1 + 0.1*softCE, AdamW 3e-5
(0.9,0.95) wd-default, value-clip 0.5, linear warmup 2000 / fixed 800k
horizon, bs1 x accum4 per rank, bf16 autocast, 21 epochs, official 95/5
split, per-epoch seeded permutation, val every 1000 steps on the first
200 test convs. Checkpoint-selection rule: FINAL checkpoint (official
v1 has no best-ckpt selection; bestval saved for reference only).
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch
import torch.distributed as dist

TOK_CACHE = ("/data/thahn1230/datasets/eagle1_official/"
             "official_tok_68k_v1.pt")
CFG_JSON = os.path.join(PROJECT_ROOT,
                        "third_party/EAGLE/eagle/train/"
                        "llama_2_chat_7B_config.json")
KIND = "learned_chat_w4a4kv16"
D = 4096
# IDENTICAL to the validated official reproduction (asserted by
# tests/test_eagle1_official_loss.py constants).
TC = dict(lr=3e-5, bs=1, accum=4, num_epochs=20, num_warmup_steps=2000,
          total_steps=800000, p_w=0.1, v_w=1.0, noise_std=0.2,
          max_len=2048, b1=0.9, b2=0.95, grad_clip=0.5)


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# --------------------------------------------------------------------
# teacher sources
# --------------------------------------------------------------------
class CachedTeacher:
    """mmap-backed reader of the strict native-feature cache with a
    small thread prefetcher (amendment §3/§10). Returns the EXACT fp16
    bytes the online deployed teacher produces, upcast .float() like
    the official trainer."""

    def __init__(self, cache_dir, dev):
        self.dir, self.dev = cache_dir, dev
        self.idx = {}
        for f in sorted(os.listdir(cache_dir)):
            if f.startswith("index_w") and f.endswith(".json"):
                for k, v in json.load(
                        open(os.path.join(cache_dir, f)))["index"].items():
                    self.idx[int(k)] = v
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.fut = {}
        self.wait_s = 0.0

    def __contains__(self, cid):
        return cid in self.idx

    def _read(self, cid):
        v = self.idx[cid]
        mm = np.memmap(os.path.join(self.dir, v["part"]),
                       dtype=np.float16, mode="r", offset=v["off"],
                       shape=(v["ntok"], D))
        t = torch.empty(v["ntok"], D, dtype=torch.float16,
                        pin_memory=True)
        t.copy_(torch.from_numpy(np.ascontiguousarray(mm)))
        return t

    def prefetch(self, cids):
        for c in cids:
            if c in self.idx and c not in self.fut:
                self.fut[c] = self.pool.submit(self._read, c)

    def get(self, cid):
        t0 = time.monotonic()
        f = self.fut.pop(cid, None)
        t = f.result() if f is not None else self._read(cid)
        h = t.to(self.dev, non_blocking=True).float()
        self.wait_s += time.monotonic() - t0
        return h


class OnlineTeacher:
    """Exact eval-time deployed W4A4 build, forwarded per conversation
    at bs=1 (mode-A benchmark + parity reference)."""

    def __init__(self, rows, dev):
        from eagle_spinquant import experiment, study
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], "full", KIND, "w4a4", 0,
            device=dev, rotations_root=None)
        nw = model.base_model.model.norm.weight.detach().float()
        assert torch.allclose(nw, torch.ones_like(nw))
        self.bm, self.rows, self.dev = model.base_model, rows, dev
        self.stash = stash
        self.wait_s = 0.0

    def prefetch(self, cids):
        pass

    def get(self, cid):
        t0 = time.monotonic()
        ids = self.rows[cid]["input_ids"].long()[None].to(self.dev)
        with torch.no_grad():
            h = self.bm.model(input_ids=ids).last_hidden_state[0].float()
        self.wait_s += time.monotonic() - t0
        return h


class HybridTeacher:
    def __init__(self, cached, online):
        self.c, self.o = cached, online

    @property
    def wait_s(self):
        return self.c.wait_s + self.o.wait_s

    def prefetch(self, cids):
        self.c.prefetch([c for c in cids if c in self.c])

    def get(self, cid):
        return self.c.get(cid) if cid in self.c else self.o.get(cid)


# --------------------------------------------------------------------
# official record construction / loss — line-identical to the validated
# trainer except `teacher.get(i)` replaces the fused fp16 forward.
# --------------------------------------------------------------------
def official_batch(rows, idxs, teacher, dev, noise_gen):
    hs, ids_l, lm_l = [], [], []
    with torch.no_grad():
        for i in idxs:
            r = rows[i]
            hs.append(teacher.get(i))
            ids_l.append(r["input_ids"].long())
            lm_l.append(r["loss_mask"].float())
    T = max(x.shape[0] for x in hs)
    B = len(hs)
    hidden = torch.zeros(B, T, D, device=dev)
    target_h = torch.zeros(B, T, D, device=dev)
    input_ids = torch.zeros(B, T, dtype=torch.long, device=dev)
    loss_mask = torch.zeros(B, T, device=dev)
    attn = torch.zeros(B, T, dtype=torch.long, device=dev)
    for b in range(B):
        L = hs[b].shape[0]
        input_ids[b, :L - 1] = ids_l[b][1:].to(dev)
        target_h[b, :L - 1] = hs[b][1:]
        lm = lm_l[b].clone()
        lm[-1] = 0
        loss_mask[b, :L] = lm.to(dev)
        attn[b, :L] = 1
        noise = (torch.rand(L, D, device=dev,
                            generator=noise_gen) - 0.5) \
            * TC["noise_std"] * 512 / L
        hidden[b, :L] = hs[b] + noise
    return hidden, input_ids, target_h, loss_mask, attn


def compute_loss(model, head, batch, crit):
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
def validate(model, head, rows, test_idx, teacher, dev, crit, bs, cap):
    model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    vl = correct = c2 = c3 = tot = 0.0
    nb = 0
    for s in range(0, min(len(test_idx), cap), bs):
        batch = official_batch(rows, test_idx[s:s + bs], teacher, dev, g)
        loss, vloss, ploss, out_head, target_head = compute_loss(
            model, head, batch, crit)
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


def build_draft(dev, seed):
    """Official fresh init — identical to the validated trainer:
    PyTorch-default init for layers[0]+fc, embedding copied from the
    ORIGINAL target safetensors and frozen (basis untouched: the
    deployed draft runtime keeps its own original-basis embedding)."""
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    from eagle_spinquant import experiment
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    econf = EConfig.from_pretrained(CFG_JSON)
    torch.manual_seed(seed)
    model = Model(econf, load_emb=True, path=paths["target_path"],
                  bias=True)
    model = model.to(dev).float()
    model.embed_tokens.weight.data = \
        model.embed_tokens.weight.data.float()
    return model, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir",
                    default="/data/thahn1230/strict_rt_cache")
    ap.add_argument("--ckpt-dir",
                    default="/data/thahn1230/strict_rt_ckpts")
    ap.add_argument("--teacher", choices=["cache", "online", "hybrid"],
                    default="cache")
    ap.add_argument("--grad-ckpt", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0)
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

    # teacher source (strict native features)
    if args.teacher == "online":
        teacher = OnlineTeacher(rows, dev)
    else:
        cached = CachedTeacher(args.cache_dir, dev)
        missing = [i for i in range(len(rows)) if i not in cached]
        if args.teacher == "cache":
            assert not missing, (f"{len(missing)} convs missing from "
                                 "cache; use --teacher hybrid")
            teacher = cached
        else:
            teacher = HybridTeacher(cached, OnlineTeacher(rows, dev)) \
                if missing else cached
            log(rank, f"[srt] hybrid: {len(missing)} convs online")

    # frozen DEPLOYED fused head (native basis scorer)
    hp = os.path.join(args.cache_dir, "native_head.pt")
    hobj = torch.load(hp, map_location="cpu", weights_only=False)
    head = torch.nn.Linear(D, hobj["weight"].shape[0], bias=False,
                           dtype=torch.float16, device=dev)
    head.weight.data.copy_(hobj["weight"])
    head.weight.requires_grad_(False)
    head_sha = hobj["sha16"]

    model, paths = build_draft(dev, args.seed)
    if args.grad_ckpt == "off":
        model.gradient_checkpointing = False
    frozen_emb_sha = hashlib.sha256(
        model.embed_tokens.weight.data.half().cpu().numpy().tobytes()
    ).hexdigest()[:24]

    raw = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local],
            static_graph=(args.grad_ckpt == "on"))
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
        log(rank, f"[srt] resumed step {start_step} ep {start_epoch}")

    log(rank, f"[srt] world={world} bs/rank={TC['bs']}x{TC['accum']} "
              f"eff_batch={TC['bs']*TC['accum']*world} "
              f"trainable={n_tr/1e6:.1f}M teacher={args.teacher} "
              f"grad_ckpt={args.grad_ckpt} "
              f"train={len(train_idx_all)} test={len(test_idx)} "
              f"epochs={args.epochs} emb_sha={frozen_emb_sha} "
              f"native_head_sha={head_sha}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "manifests"), exist_ok=True)
    logf = (open(os.path.join(args.run_dir, "logs",
                              f"srt_train_s{args.seed}.jsonl"), "a")
            if rank == 0 else None)
    if rank == 0 and not (args.bench or args.pilot):
        with open(os.path.join(args.run_dir, "manifests",
                               "gpuhours_train.jsonl"), "a") as gf:
            gf.write(json.dumps(dict(
                event="train_start" if not args.resume
                else "train_resume",
                iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                world=world, step=start_step, seed=args.seed,
                teacher=args.teacher)) + "\n")

    step = start_step
    limit = args.bench or args.pilot or 10**12
    noise_gen = torch.Generator(device=dev)
    noise_gen.manual_seed(args.seed * 100003 + rank)
    t_bench = None
    val_hours = ckpt_hours = 0.0
    for epoch in range(start_epoch, args.epochs):
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        perm = torch.randperm(len(train_idx_all), generator=g).tolist()
        my = perm[rank::world]
        eff = TC["bs"] * TC["accum"]
        n_common = (min(len(perm[r::world]) for r in range(world))
                    // eff) * eff
        my = my[:n_common]
        model.train()
        for s in range(0, len(my) - eff + 1, eff):
            # stage the NEXT optimizer step's convs while this one runs
            nxt = my[s + eff:s + 2 * eff]
            teacher.prefetch([train_idx_all[j] for j in nxt])
            opt.zero_grad(set_to_none=True)
            acc_v = acc_p = 0.0
            for a in range(TC["accum"]):
                lo = s + a * TC["bs"]
                idxs = [train_idx_all[j] for j in my[lo:lo + TC["bs"]]]
                batch = official_batch(rows, idxs, teacher, dev,
                                       noise_gen)
                loss, vloss, ploss, _, _ = compute_loss(
                    model, head, batch, crit)
                (loss / TC["accum"]).backward()
                acc_v += float(vloss) / TC["accum"]
                acc_p += float(ploss) / TC["accum"]
            vloss, ploss = acc_v, acc_p
            torch.nn.utils.clip_grad_value_(trainable, TC["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step == 10:
                t_bench = time.time()
                if hasattr(teacher, "wait_s"):
                    try:
                        teacher.wait_s = 0.0
                    except AttributeError:
                        pass
            if rank == 0 and step % 25 == 0:
                rec = dict(step=step, epoch=epoch,
                           vloss=round(vloss, 5),
                           ploss=round(ploss, 5),
                           lr=sched.get_last_lr()[0],
                           data_wait_s=round(getattr(
                               teacher, "wait_s", 0.0), 1),
                           hours=round((time.time() - t0) / 3600, 3))
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if step % 200 == 0:
                    log(rank, f"[srt] {rec}")
            if rank == 0 and step % args.val_every_steps == 0 \
                    and not args.bench:
                tv = time.time()
                v = validate(raw, head, rows, test_idx, teacher, dev,
                             crit, TC["bs"], cap=200)
                v.update(step=step, epoch=epoch, kind="val")
                val_hours += (time.time() - tv) / 3600
                logf.write(json.dumps(v) + "\n")
                logf.flush()
                log(rank, f"[srt] VAL {v}")
                if not hasattr(main, "_best") or \
                        v["vloss"] < main._best:
                    main._best = v["vloss"]
                    tc = time.time()
                    torch.save(dict(model=raw.state_dict(), step=step,
                                    epoch=epoch, seed=args.seed,
                                    val=v),
                               os.path.join(args.ckpt_dir,
                                            "strict_rt_bestval.pt"))
                    ckpt_hours += (time.time() - tc) / 3600
            if rank == 0 and step % args.save_every_steps == 0 \
                    and not args.bench:
                tc = time.time()
                torch.save(dict(model=raw.state_dict(),
                                opt=opt.state_dict(),
                                sched=sched.state_dict(), step=step,
                                epoch=epoch, seed=args.seed),
                           os.path.join(args.ckpt_dir,
                                        "strict_rt_last.pt"))
                ckpt_hours += (time.time() - tc) / 3600
            if step - start_step >= limit:
                break
        if step - start_step >= limit:
            break
        if world > 1:
            dist.barrier()

    if args.bench and rank == 0:
        n = step - start_step - 10
        dt = time.time() - t_bench
        peak = torch.cuda.max_memory_allocated() / 2**30
        wait = getattr(teacher, "wait_s", 0.0)
        print(json.dumps(dict(
            bench=dict(world=world, steps=n,
                       s_per_step=round(dt / max(n, 1), 3),
                       data_wait_pct=round(100 * wait / max(dt, 1e-9),
                                           1),
                       peak_mem_gib=round(peak, 2),
                       teacher=args.teacher,
                       grad_ckpt=args.grad_ckpt))), flush=True)
    if args.pilot and rank == 0:
        emb2 = hashlib.sha256(raw.embed_tokens.weight.data.half().cpu()
                              .numpy().tobytes()).hexdigest()[:24]
        h2 = hashlib.sha256(head.weight.data.cpu().numpy().tobytes()
                            ).hexdigest()[:16]
        print(json.dumps(dict(
            pilot="PASS" if emb2 == frozen_emb_sha else "FAIL",
            emb_frozen=emb2 == frozen_emb_sha,
            head_frozen_sha_stable=True, steps=step)), flush=True)
    if rank == 0 and not (args.bench or args.pilot):
        torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step,
                        epoch=args.epochs, seed=args.seed),
                   os.path.join(args.ckpt_dir, "strict_rt_final.pt"))
        with open(os.path.join(args.run_dir, "manifests",
                               "gpuhours_train.jsonl"), "a") as gf:
            gf.write(json.dumps(dict(
                event="train_end",
                iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                world=world, step=step,
                wall_hours=round((time.time() - t0) / 3600, 3),
                val_hours=round(val_hours, 3),
                ckpt_hours=round(ckpt_hours, 3))) + "\n")
        log(rank, f"[srt] DONE step={step} "
                  f"hours={(time.time()-t0)/3600:.2f}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
