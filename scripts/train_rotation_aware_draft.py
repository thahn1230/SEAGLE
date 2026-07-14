#!/usr/bin/env python
"""Variant G: train a rotation-NATIVE draft — consumes h_hat, recycles f_hat,
scored by the rotated head, with NO runtime basis conversion and ONE fc path.

Init options (--init):
  f_gamma  : F_R_gamma algebraic conversion (exact everywhere except the
             RMSNorm stage — training must repair exactly that).   [default]
  b_fold   : original draft with only the fc h-block folded (B init; the
             recycled path starts fully wrong).

Objective (teacher = frozen ROTATED target, quant OFF):
  input  : h_hat[t] (+ token ids), teacher for position t: h_hat[t+1]
  L = lambda_feat * SmoothL1(f_hat_pred, h_hat_next)
    + lambda_kl   * softCE(head_rot(f_hat_pred), head_rot(h_hat_next))
    + lambda_cos  * (1 - cosine(f_hat_pred, h_hat_next))

Stages (spec §4-G): stage0 = 100 seqs/1 epoch smoke; stage1 = 1200 seqs/2
epochs. Training is fp32 (fp16 NaNs, known from Variant C).

Trainable (--trainable): 'all' (fc + layer; embed stays frozen as in EAGLE)
or 'fc_norm' (fc + post_attention_layernorm only, G1-lightweight).

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/train_rotation_aware_draft.py \
      --stage 0 --out-name rotnative_stage0
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (experiment, logging_utils,  # noqa: E402
                             rotation_aware as ra, study)

DEV = "cuda:0"
STAGES = {0: dict(num_sequences=100, epochs=1),
          1: dict(num_sequences=1200, epochs=2),
          2: dict(num_sequences=5000, epochs=2)}


@torch.no_grad()
def gen_data(paths, cfg, num_sequences, seqlen, rotations_root):
    """Frozen rotated target over wikitext-2: (token_ids, h_hat) per sequence."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    tok = AutoTokenizer.from_pretrained(paths["target_path"], use_fast=False)
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"],
                             rotations_root)
    m = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                low_cpu_mem_usage=True).eval()
    stash = study.apply_rotation_quant(m, "full", r_bin, "none",
                                       cfg["model"]["target"], DEV)
    m.to(DEV)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(train["text"]), return_tensors="pt").input_ids
    data = []
    for i in range(num_sequences):
        chunk = ids[:, i * seqlen:(i + 1) * seqlen].to(DEV)
        if chunk.shape[1] < seqlen:
            break
        hh = m.model(input_ids=chunk, use_cache=False)[0]
        data.append((chunk.cpu(), hh.float().cpu()))
    del m
    torch.cuda.empty_cache()
    return data, stash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--init", default="f_gamma", choices=["f_gamma", "b_fold"])
    ap.add_argument("--trainable", default="all", choices=["all", "fc_norm"])
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss-basis", default="s", choices=["s", "original"],
                    help="'original': measure feat/cos losses after mapping "
                         "pred/target back with S^-1 (well-conditioned; the "
                         "S basis amplifies 1/gamma outlier channels ~300x "
                         "and destabilizes regression — Stage-0 attempt 1 "
                         "diverged in 's'). Model I/O stays S-basis either way.")
    ap.add_argument("--lambda-feat", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=0.5)
    ap.add_argument("--lambda-cos", type=float, default=0.2)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    ra.verify_fold_algebra()
    torch.manual_seed(0)

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    st = STAGES[args.stage]
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "draft_ckpts", args.out_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "command.txt"), "w") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")

    print(f"[G] stage {args.stage}: generating {st['num_sequences']} seqs "
          f"x {args.seqlen} of rotated-target h_hat...", flush=True)
    t0 = time.time()
    data, stash = gen_data(paths, cfg, st["num_sequences"], args.seqlen,
                           rotations_root)
    print(f"[G] {len(data)} sequences in {time.time()-t0:.0f}s", flush=True)
    R1 = stash["R1"].cpu().double()
    gamma = stash["gamma_f"].cpu().double()

    # draft init
    draft = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    sd_cpu = {k: v.detach().cpu() for k, v in draft.state_dict().items()}
    if args.init == "f_gamma":
        conv, _ = ra.convert_draft_state(sd_cpu, R1, gamma, "gamma")
        draft.load_state_dict({k: v.float() for k, v in conv.items()}, strict=True)
    else:
        M = study.fold_matrix(R1.float(), gamma.float()).to(draft.fc.weight.device)
        draft.fc.weight.data[:, 4096:] = (
            draft.fc.weight.data[:, 4096:].double() @ M.double()).float()
    draft.to(DEV).train()
    draft.gradient_checkpointing = False

    head_rot = ra.build_rotated_head(stash["lm_head_weight"], R1, gamma,
                                     DEV, torch.float32)
    for p in draft.embed_tokens.parameters():
        p.requires_grad = False
    if args.trainable == "fc_norm":
        for n_, p in draft.named_parameters():
            p.requires_grad = ("fc." in n_ or "post_attention_layernorm" in n_)
        draft.embed_tokens.weight.requires_grad = False
    params = [p for p in draft.parameters() if p.requires_grad]
    print(f"[G] trainable params: {sum(p.numel() for p in params)/1e6:.1f}M "
          f"(init={args.init}, trainable={args.trainable})", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    smoothl1 = nn.SmoothL1Loss()
    S_inv32 = ra.s_inv_matrix(R1, gamma).float().to(DEV)

    loss_rows, step = [], 0
    for epoch in range(st["epochs"]):
        for ids_cpu, hh_cpu in data:
            ids = ids_cpu.to(DEV)
            hh = hh_cpu.to(DEV)              # [1, L, D] h_hat
            # input h_hat[0:L-1] predicts h_hat[1:L]; ids aligned like EAGLE:
            # draft sees ids[1:] with hidden[:-1]
            x_h = hh[:, :-1]
            y_h = hh[:, 1:]
            out = draft(x_h, input_ids=ids[:, 1:], use_cache=False)
            pred = out[0] if isinstance(out, tuple) else out
            if args.loss_basis == "original":
                pred_l, y_l = pred @ S_inv32, y_h @ S_inv32
            else:
                pred_l, y_l = pred, y_h
            l_feat = smoothl1(pred_l, y_l)
            with torch.no_grad():
                t_log = F.log_softmax(head_rot(y_h), dim=-1)
            p_log = F.log_softmax(head_rot(pred), dim=-1)
            l_kl = F.kl_div(p_log, t_log, log_target=True, reduction="batchmean") \
                / pred.shape[1]
            l_cos = (1 - F.cosine_similarity(pred_l, y_l, dim=-1)).mean()
            loss = (args.lambda_feat * l_feat + args.lambda_kl * l_kl
                    + args.lambda_cos * l_cos)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                row = dict(step=step, epoch=epoch, loss=loss.item(),
                           l_feat=l_feat.item(), l_kl=l_kl.item(),
                           l_cos=l_cos.item())
                loss_rows.append(row)
                print(f"  step {step:5d} loss={loss.item():.4f} "
                      f"feat={l_feat.item():.4f} kl={l_kl.item():.4f} "
                      f"cos={l_cos.item():.4f}", flush=True)

    ckpt = os.path.join(ckpt_dir, "draft_rotnative.pt")
    torch.save({k: v.detach().cpu() for k, v in draft.state_dict().items()}, ckpt)
    logging_utils.write_csv(os.path.join(ckpt_dir, "loss_curve.csv"), loss_rows)
    meta = dict(stage=args.stage, init=args.init, trainable=args.trainable,
                sequences=len(data), seqlen=args.seqlen, steps=step,
                loss_basis=args.loss_basis, lr=args.lr,
                r_bin="outputs/rotations/random_hadamard/R.bin (seed 0; "
                      "ckpt is ONLY valid against this rotation)",
                first_loss=loss_rows[0] if loss_rows else None,
                final_loss=loss_rows[-1] if loss_rows else None, gpu=gpu)
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[G] ckpt -> {ckpt}")
    print(json.dumps({k: meta[k] for k in ('steps', 'first_loss', 'final_loss')},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
