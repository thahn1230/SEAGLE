#!/usr/bin/env python
"""EAGLE-1 draft training for the PTQ-vs-QAT study: original EAGLE
objective/optimizer/data (audited in tables/qat_pipeline_audit.json), with
runtime-matched STE W4A4 fake quantization at the deployed INT4 sites
(QAT arms) or no quantization (FP16 control arms).

Arms
  C3  : FP16 target teacher, identity-basis D4P3 QAT      (deploy: d4p3@fp16)
  C7  : INT4 target teacher, gamma_R1 shared-R_T D4P3 QAT (deploy: d4p3@int4)
  C7b : C7 initialized from the C6 target-adapted FP16 checkpoint
  C6  : INT4 target teacher, restored-basis FP16 retrain  (deploy: restored)
  C8  : FP16 target teacher, stock-basis FP16 retrain     (deploy: stock)

Faithfulness contract (= eagle/train/main.py):
  loss = 1.0 * vloss + 0.1 * ploss
    vloss = SmoothL1(pred_hidden, target_hidden), loss_mask-weighted
    ploss = -sum softmax(head(target)) * log_softmax(head(pred)), masked
  AdamW lr 3e-5 betas (0.9, 0.95), clip_grad_VALUE 0.5,
  linear warmup 2000 -> linear decay, uniform input noise
  (rand-0.5)*0.2*512/len, max_len 2048, head frozen, draft params trainable
  (fc.weight+bias, decoder-layer linears, post_attention_layernorm; the
  embedding is FROZEN by the original cnets code itself).
Teacher = FUSED frozen-target forward per batch (identical signal to the
pre-generated hidden pipeline; ge_data uses hidden_states[-1] = post-norm).
Single-step (all-first-path) structure exactly like the original.
"""
import argparse, hashlib, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
import torch.nn.functional as F

from eagle_spinquant import experiment, study
from eagle_spinquant.exact_quantized_rotation_forward import (
    ExactQuantizedRotationForward)

KIND = "learned_chat_w4a4kv16"
SHAREGPT_JSON = ("/data/thahn1230/datasets/sharegpt/"
                 "ShareGPT_V3_unfiltered_cleaned_split.json")
CACHE = "/data/thahn1230/datasets/sharegpt/eagle_tok_cache_v1.pt"
TRAIN_N, VAL_N = 12000, 400
MAX_LEN = 2048
D = 4096


class _FixedRot:
    def __init__(self, R):
        self._R = R

    def R(self):
        return self._R


# ---------------------------------------------------------------- data ------
def build_tokenized_split(tokenizer):
    """Replicates ge_data_all_llama2chat.build_dataset_rank exactly
    (fastchat llama-2-chat template, pinned system prompt, loss-mask
    offsets), on the pinned ShareGPT V3 json, shuffle seed 42. Excludes any
    conversation whose first human turn matches the sharegpt eval/calib/
    valid pools (Gate H). Cached to CACHE with a manifest."""
    if os.path.exists(CACHE):
        return torch.load(CACHE, weights_only=False)
    from datasets import load_dataset
    from fastchat.model.model_adapter import get_conversation_template
    from eagle_spinquant.eval_datasets import load_eval_prompts

    banned = set()
    for pool in ("eval", "calib", "valid"):
        try:
            rows, _ = load_eval_prompts("sharegpt", 80, pool)
            for r in rows:
                banned.add(r["text"][-400:])
        except Exception as e:
            print(f"[data] pool {pool} skip: {e}")

    ds = load_dataset("json", data_files=SHAREGPT_JSON)["train"] \
        .shuffle(seed=42)
    sys_p = ("You are a helpful, respectful and honest assistant. Always "
             "answer as helpfully as possible, while being safe.  Your "
             "answers should not include any harmful, unethical, racist, "
             "sexist, toxic, dangerous, or illegal content. Please ensure "
             "that your responses are socially unbiased and positive in "
             "nature.\n\nIf a question does not make any sense, or is not "
             "factually coherent, explain why instead of answering "
             "something not correct. If you don't know the answer to a "
             "question, please don't share false information.")
    out, idx, n_banned = [], 0, 0
    need = TRAIN_N + VAL_N
    while len(out) < need and idx < len(ds):
        ex = ds[idx]; idx += 1
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = sys_p
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        source = ex["conversations"]
        if not source:
            continue
        if roles.get(source[0].get("from")) != conv.roles[0]:
            source = source[1:]
        if not source:
            continue
        ok = True
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence.get("from"))
            if role != conv.roles[j % 2]:
                ok = False
                break
            if sentence["from"] == "gpt":
                sentence["value"] = " " + sentence["value"]
            conv.append_message(role, sentence["value"])
        if not ok:
            continue
        conversation = conv.get_prompt()
        first_human = next((s["value"] for s in source
                            if s.get("from") == "human"), "")
        if any(b and b[:80] in first_human for b in banned):
            n_banned += 1
            continue
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        input_ids = tokenizer(conversation, return_tensors="pt",
                              max_length=MAX_LEN,
                              truncation=True).input_ids[0]
        loss_mask = torch.ones_like(input_ids)
        sep = conv.sep + conv.roles[1] + " "
        turns = conversation.split(conv.sep2)
        cur_len = 1
        loss_mask[:cur_len] = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            loss_mask[cur_len: cur_len + instruction_len] = 0
            cur_len += turn_len + 2
            if i != 0 and not tokenizer.legacy:
                cur_len -= 1
        loss_mask[cur_len:] = 0
        if int(loss_mask.sum()) < 8 or input_ids.shape[0] < 16:
            continue
        out.append(dict(input_ids=input_ids.short(),
                        loss_mask=loss_mask.bool()))
    assert len(out) == need, f"only {len(out)} usable conversations"
    js_sha = hashlib.sha256(open(SHAREGPT_JSON, "rb").read()).hexdigest()
    blob = dict(
        rows=out, train_n=TRAIN_N, val_n=VAL_N,
        manifest=dict(source=SHAREGPT_JSON, sha256=js_sha,
                      shuffle_seed=42, template="fastchat llama-2-chat",
                      max_len=MAX_LEN, banned_matched=n_banned,
                      raw_rows_scanned=idx,
                      exclusion="sharegpt eval/calib/valid pools (Gate H)"))
    torch.save(blob, CACHE)
    print(f"[data] cached {len(out)} convs ({n_banned} excluded, "
          f"{idx} scanned) -> {CACHE}")
    return blob


def make_batches(rows, order, bs):
    for i in range(0, len(order) - bs + 1, bs):
        batch = [rows[j] for j in order[i:i + bs]]
        T = max(r["input_ids"].shape[0] for r in batch)
        ids = torch.zeros(len(batch), T, dtype=torch.long)
        lm = torch.zeros(len(batch), T, dtype=torch.bool)
        am = torch.zeros(len(batch), T, dtype=torch.bool)
        lens = []
        for b, r in enumerate(batch):
            L = r["input_ids"].shape[0]
            ids[b, :L] = r["input_ids"].long()
            lm[b, :L] = r["loss_mask"]
            am[b, :L] = True
            lens.append(L)
        yield ids, lm, am, lens


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["C3", "C7", "C7b", "C6", "C8"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alpha-rec", type=float, default=None,
                    help="EP3-P pathwise m_rec (enables the deploy "
                         "EP3-P fold in the exact core)")
    ap.add_argument("--teacher-quant", default=None,
                    choices=["none", "w8a8", "w4a4"],
                    help="override teacher target quant (rot=full "
                         "unless none); default derives from --arm")
    ap.add_argument("--rd-ckpt", default=None,
                    help="draft-aware R_D checkpoint: draft folds use "
                         "R_D, T->D bridge pinned at R_T (B8 arms)")
    ap.add_argument("--init-sd", default=None, help="C7b: C6 export .pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-ckpt-every", type=int, default=0,
                    help="also export step-stamped ckpts every N steps")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    tag = args.tag or f"{args.arm}_s{args.seed}"
    done_manifest = os.path.join(args.run_dir, "manifests",
                                 f"train_{tag}.json")
    if os.path.exists(done_manifest):
        print(f"[{tag}] manifest exists, training already complete; skip")
        return 0
    dev = args.device
    t0 = time.time()
    torch.manual_seed(1000 + args.seed)

    quant = args.arm in ("C3", "C7", "C7b")
    int4_teacher = args.arm in ("C6", "C7", "C7b")
    rotated_basis = args.arm in ("C7", "C7b")
    def_alpha = 32.0 if rotated_basis else 45.254834
    alpha = (args.alpha if args.alpha is not None
             else (def_alpha if quant else 1.0))

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    rot, tq = (("full", "w4a4") if int4_teacher else ("none", "none"))
    if args.teacher_quant is not None:
        rot, tq = (("none", "none") if args.teacher_quant == "none"
                   else ("full", args.teacher_quant))
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, tq, 0, device=dev, rotations_root=rr)
    model.base_model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    from eagle_spinquant import eagle_bridge
    tok = eagle_bridge.get_tokenizer(model)

    blob = build_tokenized_split(tok)
    rows = blob["rows"]
    train_rows = rows[:TRAIN_N]
    val_rows = rows[TRAIN_N:TRAIN_N + VAL_N]

    ea_sd = {k: v.detach().cpu() for k, v in
             model.ea_layer.state_dict().items()}
    if args.arm == "C7b":
        assert args.init_sd, "C7b requires --init-sd (C6 export)"
    if args.init_sd:
        init = torch.load(args.init_sd, map_location="cpu",
                          weights_only=False)
        init = init.get("draft_state_dict", init.get("model", init))
        for k in ea_sd:
            if k in init:
                ea_sd[k] = init[k].to(ea_sd[k].dtype)
        print(f"[{tag}] initialized from {args.init_sd}")

    # The runtime draft is INTERNALLY R1-rotated in EVERY interface mode
    # (W_rec=[W_e|W_h R1], post-projection R1, R1-conjugated AR linears,
    # head W_lm·R1); 'identity' only removes the first-path input fold.
    if "R1" in stash and stash["R1"] is not None:
        R1 = stash["R1"].float()
    else:                       # unrotated target build: load from R.bin
        R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                       map_location="cpu", weights_only=False)
        R1 = R["R1"].float()
    W_lm = stash["lm_head_weight"].float()
    R_D = None
    if args.rd_ckpt:
        R_D = torch.load(args.rd_ckpt, map_location="cpu",
                         weights_only=False)["R_D"].float()
    rot_shim = _FixedRot((R_D if R_D is not None else R1).to(dev))
    if rotated_basis:
        gamma = stash["gamma_f"].float()
        first_fold = R1
    else:                       # identity first path: [W_e | W_h], no fold
        gamma = torch.ones(D)
        first_fold = torch.eye(D)
    bits = 4 if quant else 16
    core = ExactQuantizedRotationForward(
        ea_sd, R1, gamma, W_lm, rot_shim, alpha_init=alpha,
        train_alpha=False, w_bits=bits, a_bits=bits, draft_kv_bits=16,
        train_draft_core=True, r2_seed=0, device=dev,
        first_fold_R=first_fold,
        alpha_rec_init=args.alpha_rec).to(dev)
    core.log_alpha.requires_grad_(False)
    trainable = [p for p in core.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[{tag}] trainable params: {n_tr/1e6:.1f}M "
          f"({len(trainable)} tensors), alpha={alpha}, bits={bits}, "
          f"teacher={'int4' if int4_teacher else 'fp16'}, "
          f"basis={'gamma_R1' if rotated_basis else 'identity'}")

    # teacher feature transforms (see study report S7):
    #   x = interface INPUT (basis the deployed first path consumes)
    #   t = regression target = (h-basis feature) @ R1  — the draft's
    #       internal hidden basis in every mode (see note above)
    R1d = R1.to(dev).float()
    RDd = R_D.to(dev).float() if R_D is not None else None
    gd = (stash["gamma_f"].float().to(dev) if "gamma_f" in stash
          else model.base_model.model.norm.weight.detach().float().to(dev))

    def teacher(ids, am):
        # chunk rows so the vendored explicit-attention target never
        # materializes more than ~4096 rows of full attention at once
        with torch.no_grad():
            B, T = ids.shape
            # 2048 rows (was 4096): on gpusystem the W4A4 fake-quant
            # target + core-QAT states leave no headroom for the 2-row
            # 2048-token explicit-attention spike (OOM at step 0,
            # 2026-08-12). Chunking is exact row-splitting under
            # no_grad — bitwise-identical teacher features.
            c = max(1, 2048 // max(T, 1))
            hs = [model.base_model.model(
                input_ids=ids[i:i + c].to(dev),
                attention_mask=am[i:i + c].to(dev)).last_hidden_state
                .float() for i in range(0, B, c)]
            h = torch.cat(hs, dim=0) if len(hs) > 1 else hs[0]
        if args.arm in ("C3", "C8"):        # fp16 teacher: h post-norm
            return h, h @ (RDd if RDd is not None else R1d)
        if args.arm == "C6":                # int4 teacher, restored input
            r = (h @ R1d.t()) * gd          # (a R^T)*gamma
            return r, r @ R1d
        # C7/C7b: rotated interface input a; h-basis = (a R^T)*gamma
        # rotated back into the DRAFT's internal basis (R_D when set)
        Rint = RDd if RDd is not None else R1d
        return h, ((h @ R1d.t()) * gd) @ Rint

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, args.warmup, args.steps)
    crit = torch.nn.SmoothL1Loss(reduction="none")

    g = torch.Generator().manual_seed(args.seed)   # data order: seed only
    order = torch.randperm(len(train_rows), generator=g).tolist()
    epoch_len = len(order) // (args.bs * args.accum)
    os.makedirs(os.path.join(args.run_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "logs"), exist_ok=True)
    logf = open(os.path.join(args.run_dir, "logs", f"train_{tag}.jsonl"),
                "a")

    def export(path, meta):
        ex = core.export_original_state()
        sd_out = dict(ea_sd)
        sd_out["fc.weight"] = torch.cat([ex["W_e"], ex["W_h"]], dim=1)
        sd_out["fc.bias"] = ex["b_fc"]
        m = {"Wq": "self_attn.q_proj.weight", "Wk": "self_attn.k_proj.weight",
             "Wv": "self_attn.v_proj.weight", "Wo": "self_attn.o_proj.weight",
             "Wgate": "mlp.gate_proj.weight", "Wup": "mlp.up_proj.weight",
             "Wdown": "mlp.down_proj.weight"}
        for k, v in m.items():
            sd_out[f"layers.0.{k.replace(k, v)}"] = ex[k]
        sd_out["layers.0.post_attention_layernorm.weight"] = ex["gl"]
        torch.save({"draft_state_dict":
                    {k: v.half() for k, v in sd_out.items()},
                    "meta": meta}, path)

    def validate(step):
        core.eval()
        vl, correct, c2, c3, tot = 0.0, 0, 0, 0, 0
        nb = 0
        with torch.no_grad():
            qw_v = core.quantized_weights(exact=False)   # fixed weights
            for ids, lm, am, lens in make_batches(
                    val_rows[:100], list(range(100)), args.bs):
                x, t = teacher(ids, am)
                h = core.forward_train(ids.to(dev), x[:, :-1].half(),
                                       pad_mask=am[:, :-1].to(dev),
                                       qw=qw_v)
                sel = (lm[:, :-1] & am[:, :-1]).to(dev).reshape(-1)
                hf = h.reshape(-1, h.shape[-1])[sel]
                tf = t[:, 1:].reshape(-1, h.shape[-1])[sel].float()
                v = crit(hf.float(), tf)
                vl += v.mean(-1).mean().item()
                nb += 1
                pk = core.head_logits(hf).topk(3, dim=-1).indices
                gt = core.head_logits(tf).argmax(-1)
                eq = pk.eq(gt.unsqueeze(-1))
                correct += eq[..., 0].sum().item()
                c2 += eq[..., :2].any(-1).sum().item()
                c3 += eq.any(-1).sum().item()
                tot += int(sel.sum().item())
        core.train()
        return dict(step=step, vloss=vl / max(nb, 1),
                    top1=correct / max(tot, 1), top2=c2 / max(tot, 1),
                    top3=c3 / max(tot, 1))

    best3, step, seen_tok = -1.0, 0, 0
    core.train()
    it = make_batches(train_rows, order, args.accum)   # teacher batch
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc_v = acc_p = 0.0
        try:
            ids, lm, am, lens = next(it)
        except StopIteration:
            order = torch.randperm(len(train_rows), generator=g).tolist()
            it = make_batches(train_rows, order, args.accum)
            ids, lm, am, lens = next(it)
        # ONE teacher forward for the whole accumulation window
        x, t = teacher(ids, am)
        # original AddUniformNoise: input-only, (rand-.5)*std*512/len
        for b, L in enumerate(lens):
            noise = (torch.rand(L, x.shape[-1], device=x.device)
                     - 0.5) * 0.2 * 512 / L
            x[b, :L] += noise
        # ONE weight fold+quant per optimizer step (weights constant
        # until opt.step(); the w_clip search dominates otherwise).
        # Micro-batches accumulate grads on detached leaves; the fold
        # graph is traversed once at the end (STE-exact, see qw_leaves).
        leaves, twR = core.qw_leaves(exact=False)
        qw_pair = (leaves, twR)
        nmicro = len(lens)
        for b, L in enumerate(lens):
            h = core.forward_train(ids[b:b + 1, :L].to(dev),
                                   x[b:b + 1, :L - 1].half(),
                                   qw=qw_pair)
            # masked-row selection: mathematically identical to the
            # original mask-weighted sums, avoids (B,T,V) tensors
            sel = lm[b:b + 1, :L - 1].to(dev).reshape(-1)
            nsel = sel.sum() + 1e-5
            hf = h.reshape(-1, h.shape[-1])[sel]
            tf = t[b:b + 1, 1:L].reshape(-1, h.shape[-1])[sel].float()
            v = crit(hf.float(), tf)
            vloss = v.mean(-1).sum() / nsel
            with torch.no_grad():
                tp = core.head_logits(tf).softmax(-1)
            lp = core.head_logits(hf).log_softmax(-1)
            ploss = -(tp * lp).sum(-1).sum() / nsel
            del tp, lp
            loss = (1.0 * vloss + 0.1 * ploss) / nmicro
            loss.backward()
            acc_v += vloss.item() / nmicro
            acc_p += ploss.item() / nmicro
            seen_tok += int(L)
        core.fold_backward(leaves)
        del leaves, qw_pair
        torch.nn.utils.clip_grad_value_(trainable, 0.5)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = dict(step=step, vloss=round(acc_v, 5),
                       ploss=round(acc_p, 5),
                       lr=sched.get_last_lr()[0], tokens=seen_tok,
                       hours=round((time.time() - t0) / 3600, 3))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step % 250 == 0:
                print(f"[{tag}] {rec}", flush=True)
        if args.save_ckpt_every and step % args.save_ckpt_every == 0:
            export(os.path.join(args.run_dir, "ckpts",
                                f"{tag}_step{step}.pt"),
                   dict(arm=args.arm, seed=args.seed, step=step,
                        alpha=alpha, bits=bits))
        if step % args.val_every == 0 or step == args.steps:
            vrec = validate(step)
            vrec["kind"] = "val"
            logf.write(json.dumps(vrec) + "\n")
            logf.flush()
            print(f"[{tag}] VAL {vrec}", flush=True)
            meta = dict(arm=args.arm, seed=args.seed, step=step,
                        alpha=alpha, bits=bits,
                        teacher="int4" if int4_teacher else "fp16",
                        basis="gamma_R1" if rotated_basis else "identity",
                        val=vrec, tokens=seen_tok,
                        gpu_hours=(time.time() - t0) / 3600,
                        lr=args.lr, warmup=args.warmup, steps=args.steps,
                        bs=args.bs, accum=args.accum,
                        epoch_len=epoch_len)
            export(os.path.join(args.run_dir, "ckpts",
                                f"{tag}_last.pt"), meta)
            if vrec["top3"] > best3:
                best3 = vrec["top3"]
                export(os.path.join(args.run_dir, "ckpts",
                                    f"{tag}_best.pt"), meta)
    hrs = (time.time() - t0) / 3600
    peak = torch.cuda.max_memory_allocated() / 2**30
    summary = dict(tag=tag, arm=args.arm, seed=args.seed, steps=args.steps,
                   tokens=seen_tok, gpu_hours=round(hrs, 3),
                   peak_mem_gib=round(peak, 2), best_top3=best3,
                   counters=core.counters)
    with open(os.path.join(args.run_dir, "manifests",
                           f"train_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[{tag}] TRAIN DONE {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
